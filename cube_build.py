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

"""

from __future__ import annotations
from typing import Optional, Dict, Any 

import numpy as np
from astropy.io import fits 
from astropy.wcs import WCS

from .io_readers import FiberSet
from .geometry import make_grid_from_points, fit_celestial_wcs, make_wcs3d
from .kernels import build_projection_A, splat_cube, ivar_through_A, neff_from_A

def build_cube(
    fibers: FiberSet,
    pixel_size_arcsec: float,
    fwhm_arcsec: float, 
    fiber_radius_arcsec: float,
    r_sigma_mult: float = 3.0,
    kernel_mode: str = "gauss",
    wcs_strategy: str = "fit", # "fit" or "none"
) -> Dict[str, Any]:
    """
    Build a 3D flux-conserving cube from FiberSet

    Steps
    -----
    1. Build a grid + inside_IFU mask
    2. Build the projection matrix A (pixels x fibers)
    3. Forward project fiber spectar -> cube
    4. Propagate IVAR through A 
    5. Compute Neff map.
    6. Optionally fit a celestial WCS and embed into 3D. 

    Parameters
    ----------
    fibers : FiberSet
        Input fiber data
    pixel_size_arcsec : float
        spaxel size
    fwhm_arcsec : float 
        Kernel FWHM used for splatting
    fiber_radius_arcsec : float
        Fiber radius (for truncation cap)
    r_sigma_mult : float 
        Sigma multiplier for truncation radius
    kernel_mode : {"gauss", "tophat}
        Kernel type
    wcs_strategy : {"fit", "none"}
        If "fit", try to use fiber.meta["ra_deg"], ["dec_deg"]
        to fit a celestial WCS
    
    Returns
    -------
    results : dict with keys 
        "cube" : (L, ny, nx) flux cube 
        "ivar" : same shape, inverse variance cube
        "neff" : (ny, nx) effective number of fibers per pixel
        "A"    : (P, K) projection matrix
        "xg", "yg", "inside" : grid and IFU mask
        "wcs3d": WCSor None
        "wave" : wavelength array
    """
    flux = fibers.flux
    ivar = fibers.ivar
    wave = fibers.wave
    x = fibers.x_arcsec
    y = fibers.y_arcsec

    flux = np.nan_to_num(fibers.flux, nan=0.0, posinf=0.0, neginf=0.0)
    ivar = np.nan_to_num(fibers.ivar, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) Grid + mask
    xg, yg, inside = make_grid_from_points(x, y, pixel_size_arcsec)
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
    )

    # 3) Forward Projection
    cube = splat_cube(flux, A, ny=ny, nx=nx)

    # 4) IVAR propagation
    ivar_cube = ivar_through_A(ivar, A, ny=ny, nx=nx)

    # 5) Neff
    neff = neff_from_A(A, ny=ny, nx=nx)

    # 6) WCS 
    wcs3d: Optional[WCS]
    if wcs_strategy == "fit" and "ra_deg" in fibers.meta and "dec_deg" in fibers.meta:
        wcs2d = fit_celestial_wcs(
            x_arcsec=x,
            y_arcsec=y,
            ra_deg=np.asarray(fibers.meta["ra_deg"], float),
            dec_deg=np.asarray(fibers.meta["dec_deg"], float),
            pixel_size_arcsec=pixel_size_arcsec,
        )
        wcs3d = make_wcs3d(wcs2d, wave)
    else:
        wcs3d = None 
    
    return dict(
        cube=cube,
        ivar=ivar_cube,
        neff=neff,
        A=A,
        xg=xg,
        yg=yg,
        inside=inside,
        wcs3d=wcs3d,
        wave=wave,
        flux_in=flux
    )

def save_cube_to_fits(
    result: Dict[str, Any],
    path: str,
    extra_header: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a cube result (from cube_build) to a FITS file

    Layout:
        HDU 0 : PRIMARY (optional metadata)
        HDU 1 : FLUX (L, ny, nx)
        HDU 2 : IVAR (L, ny, nx)
        HDU 3 : NEFF (ny, nx)
        HDU 4 : WAVE (BinTable, column 'WAVE')

    WCS is attached to FLUX header
    """
    cube = result["cube"]
    ivar = result["ivar"]
    neff = result["neff"]
    wave = result["wave"]
    wcs3d: Optional[WCS] = result.get("wcs3d", None)

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
    cube_hdu = fits.ImageHDU(cube.astype("f4"), header=flux_header, name="FLUX") 
    ivar_hdu = fits.ImageHDU(ivar.astype("f4"), name="IVAR")
    neff_hdu = fits.ImageHDU(neff.astype("f4"), name="NEFF")

    col = fits.Column(name="WAVE", format="E", unit="Angstrom", array=wave.astype("f4"))
    wave_hdu = fits.BinTableHDU.from_columns([col], name="WAVE")

    fits.HDUList([primary_hdu, cube_hdu, ivar_hdu, neff_hdu, wave_hdu]).writeto(
        path, overwrite=True
    )