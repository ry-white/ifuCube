#!/usr/bin/env python

"""
mapcat_build.py
---------------
Builds high quality DAP emission line maps
And catalogs wanted DAP quantities

Wires together:
    - io_reader.FiberSet
    - geometry.make_grid_from_points
    - geometry.fit_celestial_wcs
    - kernels.build_projection_A / splat_cube / ivar_through_A / neff_from_A

Units of the output
-------------------
The projection is flux-conserving (columns of A sum to 1), so a pixel value
is an *amount of flux per output pixel*, and is therefore proportional to the
pixel solid angle. save_map_to_fits records this in the FLUX header as

    BUNIT    = 'flux/pixel'
    PIXAR_A2 = <pixel solid angle in arcsec**2>

so that anything downstream which averages pixels (HEALPix binning,
reprojection to a different grid) can divide by PIXAR_A2 first and work in
surface brightness instead. See the kernels.py module docstring.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .io_readers import FiberSet
from .geometry import (make_grid_from_points, fit_celestial_wcs, make_wcs3d,
                       pixel_area_arcsec2)
from .kernels import (build_projection_A, splat_cube, ivar_through_A,
                      neff_from_A, coverage_from_A)

def build_map(
    fibers: FiberSet,
    pixel_size_arcsec: float,
    fwhm_arcsec: float,
    fiber_radius_arcsec: float,
    r_sigma_mult: float = 3.0,
    kernel_mode: str = "gauss",
    wcs_strategy: str = "fit", # "fit" or "none"
    max_dropped_frac: float = 0.02,
) -> Dict[str, Any]:
    """
    Build flux-conserving emission-line maps from a FiberSet

    Steps
    -----
    1. Build a grid + inside_IFU mask
    2. Build the projection matrix A (pixels x fibers)
    3. Forward project fiber line fluxes -> maps
    4. Propagate IVAR through A
    5. Compute Neff and coverage maps.
    6. Optionally fit a celestial WCS.

    Parameters
    ----------
    fibers : FiberSet
        Input fiber data
    pixel_size_arcsec : float
        spaxel size
    fwhm_arcsec : float
        Kernel FWHM used for splatting. Set this from the fiber spacing
        (~37 arcsec for LVM), not from the pixel size -- see kernels.py.
    fiber_radius_arcsec : float
        Fiber radius; sets the grid padding and hull expansion.
    r_sigma_mult : float
        Sigma multiplier for truncation radius
    kernel_mode : {"gauss", "tophat"}
        Kernel type
    wcs_strategy : {"fit", "none"}
        If "fit", use fibers.meta["ra_deg"], ["dec_deg"] to fit a celestial WCS
    max_dropped_frac : float
        Forwarded to build_projection_A: refuse to build if more than this
        fraction of fibers would have their flux discarded.

    Returns
    -------
    results : dict with keys
        "map" : (N_lines, ny, nx) emission maps, or (ny, nx) for a single line
        "ivar" : same shape, inverse variance map
        "neff" : (ny, nx) effective number of fibers per pixel
        "coverage" : (ny, nx) summed kernel weight per pixel (row sums of A);
                     divide "map" by this to get a weighted average instead
                     of a flux-conserving sum
        "A"    : (P, K) projection matrix
        "xg", "yg", "inside" : grid and IFU mask
        "wcs2d" : WCS or None
        "wave" : wavelength array
        "line_names" : list[str] if present in fibers.meta
        "pixel_size_arcsec" : echoed back, so save_map_to_fits can record it
    """

    # Scrub non-finite input. This matters more than it looks: the projection
    # is a matrix multiply, so one NaN fiber flux propagates to EVERY pixel in
    # that fiber's kernel support. DAP line fits fail on individual fibers
    # routinely, and at fwhm=35" a few percent of bad fibers NaN out a fifth
    # of the map. (cube_build.py has always done this; build_map did not.)
    flux = np.nan_to_num(np.asarray(fibers.flux, float),
                         nan=0.0, posinf=0.0, neginf=0.0)   # (K, N)
    ivar = np.nan_to_num(np.asarray(fibers.ivar, float),
                         nan=0.0, posinf=0.0, neginf=0.0)   # (K, N)
    wave = np.asarray(fibers.wave, float)   # (N,)
    x    = np.asarray(fibers.x_arcsec, float)
    y    = np.asarray(fibers.y_arcsec, float)

    # grid and mask
    xg, yg, inside = make_grid_from_points(
        x_arcsec=x,
        y_arcsec=y,
        pixel_size_arcsec=pixel_size_arcsec,
        fiber_radius_arcsec=fiber_radius_arcsec,
    )

    ny, nx = inside.shape

    # build projection matrix A
    A = build_projection_A(
        x_arcsec=x,
        y_arcsec=y,
        xg=xg,
        yg=yg,
        inside_mask=inside,
        fwhm_arcsec=fwhm_arcsec,
        fiber_radius_arcsec=fiber_radius_arcsec,
        r_sigma_mult=r_sigma_mult,
        mode=kernel_mode,
        max_dropped_frac=max_dropped_frac,
    )

    # Forward project, get a cube shape (L, ny, nx)
    map_cube = splat_cube(flux, A, ny=ny, nx=nx)

    # propagate uncertainties through A
    ivar_cube = ivar_through_A(ivar, A, ny=ny, nx=nx)

    # effective number of fibers per pixel, and raw coverage
    neff = neff_from_A(A, ny=ny, nx=nx)
    coverage = coverage_from_A(A, ny=ny, nx=nx)

    # L=1 for a single emission line
    if map_cube.shape[0] == 1:
        emap = map_cube[0]          # (ny, nx)
        eivar = ivar_cube[0]    # (ny, nx)
    else:
        # if multiple lines are given
        emap = map_cube
        eivar = ivar_cube

    # Fit 2D WCS.
    # x0/y0 MUST be the grid origin: array element [0,0] sits at
    # (xg[0], yg[0]), not at (0, 0) arcsec. Omitting them offsets the WCS
    # from the array by about a third of a degree. See geometry.py.
    wcs2d: Optional[WCS]
    if wcs_strategy == "fit" and "ra_deg" in fibers.meta and "dec_deg" in fibers.meta:
        wcs2d = fit_celestial_wcs(
            x_arcsec=x,
            y_arcsec=y,
            ra_deg=np.asarray(fibers.meta["ra_deg"], float),
            dec_deg=np.asarray(fibers.meta["dec_deg"], float),
            pixel_size_arcsec=pixel_size_arcsec,
            x0_arcsec=xg[0],
            y0_arcsec=yg[0],
        )
    else:
        wcs2d = None

    line_names = fibers.meta.get("line_names", None)

    return dict(
        map=emap,
        ivar=eivar,
        neff=neff,
        coverage=coverage,
        A=A,
        xg=xg,
        yg=yg,
        inside=inside,
        wave=wave,
        wcs2d=wcs2d,
        line_names=line_names,
        pixel_size_arcsec=float(pixel_size_arcsec),
        flux_in=flux,
    )

def save_map_to_fits(
    result: Dict[str, Any],
    path: str,
    extra_header: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save an emission-line map result (from build_map) into a FITS file.

    Layout
    ======
    HDU 0 : PRIMARY
    HDU 1 : FLUX   (Nlines, ny, nx) or (ny, nx)
    HDU 2 : IVAR  (same shape)
    HDU 3 : NEFF  (ny, nx)
    HDU 4 : LINES (BinTable: LINE_NAME, WAVE)
    HDU 5 : COVERAGE (ny, nx)   -- summed kernel weight, if present

    WCS (if present) is written to the FLUX, IVAR, NEFF and COVERAGE headers
    (they all share the same grid) and applies to the last two axes; the first
    axis of FLUX/IVAR is "line index".

    BUNIT / PIXAR_A2 are written so that downstream code knows these values
    are flux *per pixel* and can convert to surface brightness.
    """

    from pathlib import Path as _P
    path = _P(path)

    M = np.asarray(result["map"], float)
    IV = np.asarray(result["ivar"], float)
    neff = np.asarray(result["neff"], float)
    wave = np.asarray(result["wave"], float)
    wcs2d: Optional[WCS] = result.get("wcs2d", None)
    line_names: Optional[Sequence[str]] = result.get("line_names", None)
    coverage = result.get("coverage", None)

    # ensure MAP and IVAR have identical shape
    if M.shape != IV.shape:
        raise ValueError("map and ivar must have the same shape")

    if M.ndim == 2:
        # promote to (1, ny, nx) for single line case
        M = M[None, ...]
        IV = IV[None, ...]

    n_lines, ny, nx = M.shape

    # Primary HDU
    primary_hdu = fits.PrimaryHDU()
    if extra_header:
        for k, v in extra_header.items():
            primary_hdu.header[k] = v

    # attach spatial WCS to the image headers
    if wcs2d is not None:
        map_hdr = wcs2d.to_header(relax=True)
        pix_area = pixel_area_arcsec2(wcs2d)
    else:
        map_hdr = fits.Header()
        ps = result.get("pixel_size_arcsec", None)
        pix_area = float(ps) ** 2 if ps else None

    # Record the units convention. These two keywords are what let the
    # HEALPix binning (or any other averaging step) convert to surface
    # brightness instead of silently inheriting this pixel size.
    map_hdr["BUNIT"] = ("flux/pixel", "flux-conserving: amount per output pixel")
    if pix_area is not None:
        map_hdr["PIXAR_A2"] = (pix_area, "pixel solid angle [arcsec**2]")
    map_hdr["FLUXCONS"] = (True, "columns of projection matrix A sum to 1")

    ivar_hdr = map_hdr.copy()
    ivar_hdr["BUNIT"] = ("1/(flux/pixel)**2", "inverse variance of FLUX")
    aux_hdr = wcs2d.to_header(relax=True) if wcs2d is not None else fits.Header()

    map_hdu = fits.ImageHDU(M.astype("f4"), header=map_hdr, name="FLUX")
    ivar_hdu = fits.ImageHDU(IV.astype("f4"), header=ivar_hdr, name="IVAR")
    neff_hdu = fits.ImageHDU(neff.astype("f4"), header=aux_hdr, name="NEFF")

    # Line metadata table
    if line_names is None:
        line_names = [f"LINE{i}" for i in range(n_lines)]

    line_names = np.asarray(line_names, dtype="U32")
    if wave.size != n_lines:
        # sanity
        raise ValueError(f"wave.size ({wave.size}) != n_lines ({n_lines})")

    col_name = fits.Column(name="LINE_NAME", format="32A", array=line_names)
    col_wave = fits.Column(name="WAVE", format="E", unit="Angstrom", array=wave.astype("f4"))
    lines_hdu = fits.BinTableHDU.from_columns([col_name, col_wave], name="LINES")

    hdus = [primary_hdu, map_hdu, ivar_hdu, neff_hdu, lines_hdu]
    if coverage is not None:
        hdus.append(fits.ImageHDU(np.asarray(coverage, float).astype("f4"),
                                  header=aux_hdr, name="COVERAGE"))

    fits.HDUList(hdus).writeto(path, overwrite=True)
