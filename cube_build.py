#!/usr/bin/env python

"""
cube_build.py
-------------
High level cube builder

Wires together:
    - io_reader.FiberSet
    - geometry.make_grid_from_points
    - geometry.fit_celestial_wcs / make_wcs3d
    - kernels.build_projection_A / splat_cube / ivar_through_A / neff_from_A

Units of the output
-------------------
Flux-conserving: a voxel value is an amount of flux per output pixel, and is
proportional to the pixel solid angle. save_cube_to_fits records BUNIT and
PIXAR_A2 so downstream averaging steps can convert to surface brightness.
See the kernels.py module docstring.

Memory note
-----------
The output is L x ny x nx floats, twice over (FLUX and IVAR). An LVM SFrame
has several thousand wavelengths, so at fine pixel scales a full-field cube
runs to tens of GB. Cubes are practical at coarse pixel scales or over a
restricted wavelength range; slice `fibers` before calling if needed.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .io_readers import FiberSet
from .geometry import (make_grid_from_points, fit_celestial_wcs, make_wcs3d,
                       pixel_area_arcsec2)
from .kernels import (build_projection_A, splat_cube, ivar_through_A,
                      neff_from_A, coverage_from_A)

def build_cube(
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
    Build a 3D flux-conserving cube from FiberSet

    Steps
    -----
    1. Build a grid + inside_IFU mask
    2. Build the projection matrix A (pixels x fibers)
    3. Forward project fiber spectra -> cube
    4. Propagate IVAR through A
    5. Compute Neff and coverage maps.
    6. Optionally fit a celestial WCS and embed into 3D.

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
        "cube" : (L, ny, nx) flux cube
        "ivar" : same shape, inverse variance cube
        "neff" : (ny, nx) effective number of fibers per pixel
        "coverage" : (ny, nx) summed kernel weight per pixel (row sums of A)
        "A"    : (P, K) projection matrix
        "xg", "yg", "inside" : grid and IFU mask
        "wcs3d": WCS or None
        "wcs2d": the celestial part, or None
        "wave" : wavelength array
        "pixel_size_arcsec" : echoed back for save_cube_to_fits
    """
    wave = np.asarray(fibers.wave, float)
    x = np.asarray(fibers.x_arcsec, float)
    y = np.asarray(fibers.y_arcsec, float)

    # A single NaN fiber flux propagates to every pixel in that fiber's
    # kernel support, so scrub before projecting.
    flux = np.nan_to_num(np.asarray(fibers.flux, float),
                         nan=0.0, posinf=0.0, neginf=0.0)
    ivar = np.nan_to_num(np.asarray(fibers.ivar, float),
                         nan=0.0, posinf=0.0, neginf=0.0)

    # 1) Grid + mask.
    # fiber_radius_arcsec was previously not forwarded here, so the grid
    # padding and hull expansion silently used the 18.5" default regardless
    # of what the caller asked for.
    xg, yg, inside = make_grid_from_points(
        x_arcsec=x,
        y_arcsec=y,
        pixel_size_arcsec=pixel_size_arcsec,
        fiber_radius_arcsec=fiber_radius_arcsec,
    )
    ny, nx = inside.shape

    # 2) Projection matrix
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

    # 3) Forward Projection
    cube = splat_cube(flux, A, ny=ny, nx=nx)

    # 4) IVAR propagation
    ivar_cube = ivar_through_A(ivar, A, ny=ny, nx=nx)

    # 5) Neff + coverage
    neff = neff_from_A(A, ny=ny, nx=nx)
    coverage = coverage_from_A(A, ny=ny, nx=nx)

    # 6) WCS.
    # x0/y0 must be the grid origin -- see geometry.py module docstring.
    wcs2d: Optional[WCS] = None
    wcs3d: Optional[WCS] = None
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
        wcs3d = make_wcs3d(wcs2d, wave)

    return dict(
        cube=cube,
        ivar=ivar_cube,
        neff=neff,
        coverage=coverage,
        A=A,
        xg=xg,
        yg=yg,
        inside=inside,
        wcs3d=wcs3d,
        wcs2d=wcs2d,
        wave=wave,
        pixel_size_arcsec=float(pixel_size_arcsec),
        flux_in=flux,
    )

def save_cube_to_fits(
    result: Dict[str, Any],
    path: str,
    extra_header: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a cube result (from build_cube) to a FITS file

    Layout:
        HDU 0 : PRIMARY (optional metadata)
        HDU 1 : FLUX (L, ny, nx)
        HDU 2 : IVAR (L, ny, nx)
        HDU 3 : NEFF (ny, nx)
        HDU 4 : WAVE (BinTable, column 'WAVE')
        HDU 5 : COVERAGE (ny, nx), if present

    WCS is attached to FLUX (3D) and to the 2D auxiliary images.
    The WAVE table holds the exact wavelength grid; the CD3_3 in the header
    can only describe a linear axis, so trust the table if they disagree.
    """
    cube = result["cube"]
    ivar = result["ivar"]
    neff = result["neff"]
    wave = np.asarray(result["wave"], float)
    wcs3d: Optional[WCS] = result.get("wcs3d", None)
    wcs2d: Optional[WCS] = result.get("wcs2d", None)
    coverage = result.get("coverage", None)

    if cube.shape != ivar.shape:
        raise ValueError("cube and ivar must have same shape")

    primary_hdu = fits.PrimaryHDU()
    if extra_header:
        for k, v in extra_header.items():
            primary_hdu.header[k] = v

    if wcs3d is not None:
        flux_header = wcs3d.to_header(relax=True)
    else:
        flux_header = fits.Header()

    if wcs2d is not None:
        pix_area = pixel_area_arcsec2(wcs2d)
    elif wcs3d is not None:
        pix_area = pixel_area_arcsec2(wcs3d)
    else:
        ps = result.get("pixel_size_arcsec", None)
        pix_area = float(ps) ** 2 if ps else None

    flux_header["BUNIT"] = ("flux/pixel", "flux-conserving: amount per output pixel")
    if pix_area is not None:
        flux_header["PIXAR_A2"] = (pix_area, "pixel solid angle [arcsec**2]")
    flux_header["FLUXCONS"] = (True, "columns of projection matrix A sum to 1")

    ivar_header = flux_header.copy()
    ivar_header["BUNIT"] = ("1/(flux/pixel)**2", "inverse variance of FLUX")
    aux_hdr = wcs2d.to_header(relax=True) if wcs2d is not None else fits.Header()

    cube_hdu = fits.ImageHDU(cube.astype("f4"), header=flux_header, name="FLUX")
    ivar_hdu = fits.ImageHDU(ivar.astype("f4"), header=ivar_header, name="IVAR")
    neff_hdu = fits.ImageHDU(np.asarray(neff, float).astype("f4"),
                             header=aux_hdr, name="NEFF")

    col = fits.Column(name="WAVE", format="E", unit="Angstrom", array=wave.astype("f4"))
    wave_hdu = fits.BinTableHDU.from_columns([col], name="WAVE")

    hdus = [primary_hdu, cube_hdu, ivar_hdu, neff_hdu, wave_hdu]
    if coverage is not None:
        hdus.append(fits.ImageHDU(np.asarray(coverage, float).astype("f4"),
                                  header=aux_hdr, name="COVERAGE"))

    fits.HDUList(hdus).writeto(path, overwrite=True)
