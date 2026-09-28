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

Units of the output -- read this before comparing map values
-------------------------------------------------------------
The FLUX image is stored as *flux per pixel* (BUNIT = 'flux/pixel').

Why: the projection is flux-conserving. Each fiber's measured flux is shared
out over nearby pixels with weights that sum to 1, so summing the pixels of
a region gives the total flux in that region, exactly -- the natural unit
for photometry. The price is that a pixel value depends on how big the
pixel is: build the same data with 5" instead of 15" pixels and each value
drops 9x (the light is split over 9x as many pixels), while totals stay the
same.

To compare maps built with different pixel sizes, or to average pixels
(which the HEALPix step does), divide by the pixel solid angle first. That
gives a surface brightness, flux per arcsec**2, which does not depend on the
pixel size:

    surface_brightness = FLUX / PIXAR_A2          [flux / arcsec**2]
    ivar_sb            = IVAR * PIXAR_A2**2       (variance scales with the square)

PIXAR_A2 is written in the FLUX header. read_map() below does this for you,
and wcs_fits_to_healpix_table does it by default.

A second, constant factor: fiber fill factor
--------------------------------------------
DAP fluxes are totals per fiber aperture (35.3" diameter), but LVM fibers
sit on a 37" hexagonal grid, so apertures cover only

    APERFILL = pi * (35.3/2)**2 / ((sqrt(3)/2) * 37**2) = 0.8255

of the sky; light falling between fibers is never measured. The map spreads
each fiber's flux over its whole grid cell, so on smooth emission the
surface brightness above reads ~0.83x the true sky value. That is the same
for every pixel size and kernel, so it does not affect comparisons between
LVM maps, but divide by APERFILL (read_map(..., correct_fill_factor=True))
when comparing against other surveys or a model in absolute units.
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

# LVM science IFU geometry (Drory et al. 2024): 35.3" apertures on a 37" hex grid
LVM_FIBER_DIAMETER_ARCSEC = 35.3
LVM_FIBER_PITCH_ARCSEC = 37.0
LVM_FILL_FACTOR = (np.pi * (LVM_FIBER_DIAMETER_ARCSEC / 2) ** 2
                   / (np.sqrt(3) / 2 * LVM_FIBER_PITCH_ARCSEC ** 2))   # 0.8255

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
        "fibercat" : per-fiber catalog from the reader (dict of (K,) arrays)
                     or None; written by save_map_to_fits as FIBERCAT
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
        fibercat=fibers.meta.get("fibercat"),
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
    HDU 6 : FIBERCAT (BinTable) -- one row per fiber used, if present

    WCS (if present) is written to the FLUX, IVAR, NEFF and COVERAGE headers
    (they all share the same grid) and applies to the last two axes; the first
    axis of FLUX/IVAR is "line index".

    BUNIT / PIXAR_A2 are written so that downstream code knows these values
    are flux *per pixel* and can convert to surface brightness; the FLUX
    header also carries COMMENT cards spelling out the conversion. See the
    module docstring, and read_map() for a reader that applies it.
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
    map_hdr["APERFILL"] = (round(float(LVM_FILL_FACTOR), 4),
                           "LVM fiber aperture fill factor, see COMMENT")
    for line in (
        "FLUX is flux PER PIXEL (flux-conserving: summing pixels gives the",
        "total flux). Values scale with pixel area, so to compare maps with",
        "different pixel sizes, or to average pixels, convert first:",
        "  surface brightness [flux/arcsec**2] = FLUX / PIXAR_A2",
        "  its inverse variance                = IVAR * PIXAR_A2**2",
        "Fibers cover APERFILL of the sky (35.3in apertures, 37in hex grid);",
        "divide surface brightness by APERFILL for true sky brightness.",
        "ifuCube.mapcat_build.read_map() applies both conversions.",
    ):
        map_hdr.add_comment(line)

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

    # Per-fiber catalog: one row per fiber that went into this map, in the
    # same order as the columns of A. Missing numeric values are stored as NaN.
    fibercat = result.get("fibercat")
    if fibercat:
        from astropy.table import Table
        cat_hdu = fits.table_to_hdu(Table(fibercat))
        cat_hdu.name = "FIBERCAT"
        cat_hdu.header["COMMENT"] = ("One row per fiber used to build FLUX; "
                                     "flux_* are per-fiber aperture totals.")
        hdus.append(cat_hdu)

    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_map(
    path: str,
    units: str = "surface_brightness",
    correct_fill_factor: bool = False,
    hdu: str = "FLUX",
) -> Dict[str, Any]:
    """
    Read a map written by save_map_to_fits, converting units on the way in.

    Parameters
    ----------
    path : str
        Map FITS file.
    units : {"surface_brightness", "flux_per_pixel"}
        "surface_brightness" (default): FLUX / PIXAR_A2, in flux/arcsec**2.
        Independent of the pixel size the map was built with, so maps built
        with different settings can be compared pixel by pixel.
        "flux_per_pixel": the stored values; sum pixels for total flux.
    correct_fill_factor : bool
        Also divide by APERFILL (~0.83), so smooth emission matches the true
        sky surface brightness. Only valid with units="surface_brightness".
    hdu : str
        Image HDU to read (default "FLUX").

    Returns
    -------
    dict with
        "data"     : (N_lines, ny, nx) or (ny, nx) array in the chosen units
        "ivar"     : matching inverse variance
        "wcs"      : celestial WCS
        "bunit"    : units string of "data"
        "header"   : header of the image HDU
        "fibercat" : astropy Table of the FIBERCAT extension, or None

    Example
    -------
        m = read_map("map_exp00006116.fits")
        plt.subplot(projection=m["wcs"]); plt.imshow(m["data"][0], origin="lower")
    """
    from astropy.table import Table

    with fits.open(path) as hdul:
        hdr = hdul[hdu].header.copy()
        data = np.asarray(hdul[hdu].data, float)
        ivar = np.asarray(hdul["IVAR"].data, float) if "IVAR" in hdul else None
        cat = Table.read(hdul["FIBERCAT"]) if "FIBERCAT" in hdul else None
    wcs2d = WCS(hdr).celestial
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
        ivar = ivar[0] if ivar is not None else None

    bunit = str(hdr.get("BUNIT", "flux/pixel"))
    if units == "surface_brightness":
        if bunit != "flux/pixel":
            raise ValueError(f"{hdu} has BUNIT={bunit!r}, not 'flux/pixel'; "
                             "use units='flux_per_pixel' to read it as stored")
        area = float(hdr.get("PIXAR_A2", pixel_area_arcsec2(wcs2d)))
        data = data / area
        ivar = ivar * area ** 2 if ivar is not None else None
        bunit = "flux/arcsec**2"
        if correct_fill_factor:
            fill = float(hdr.get("APERFILL", LVM_FILL_FACTOR))
            data = data / fill
            ivar = ivar * fill ** 2 if ivar is not None else None
            bunit += " (aperture-fill corrected)"
    elif units == "flux_per_pixel":
        if correct_fill_factor:
            raise ValueError("correct_fill_factor needs units='surface_brightness'")
    else:
        raise ValueError(f"units must be 'surface_brightness' or 'flux_per_pixel', got {units!r}")

    return dict(data=data, ivar=ivar, wcs=wcs2d, bunit=bunit, header=hdr, fibercat=cat)
