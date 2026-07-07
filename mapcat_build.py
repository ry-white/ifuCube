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
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
from astropy.io import fits 
from astropy.wcs import WCS

from .io_readers import FiberSet
from .geometry import make_grid_from_points, fit_celestial_wcs, make_wcs3d
from .kernels import build_projection_A, splat_cube, ivar_through_A, neff_from_A

def build_map(
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
        "map" : (N_lines, ny, nx) emission maps
        "ivar" : same shape, inverse variance map
        "neff" : (ny, nx) effective number of fibers per pixel
        "A"    : (P, K) projection matrix
        "xg", "yg", "inside" : grid and IFU mask
        "wcs2d" : WCSor None
        "wave" : wavelength array
        "line_names" : list[str] if present in fibers.meta
    """

    flux = np.asarray(fibers.flux, float)   # (K, N)
    ivar = np.asarray(fibers.ivar, float)   # (K, N)
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
    )

    # Forward project, get a cube shape (L, ny, nx)
    map_cube = splat_cube(flux, A, ny=ny, nx=nx)

    # propogate uncertainties through A 
    ivar_cube = ivar_through_A(ivar, A, ny=ny, nx=nx)

    # effective number of fibers per pixel
    neff = neff_from_A(A, ny=ny, nx=nx)

    # L=1 for a single emission line
    if map_cube.shape[0] == 1:
        emap = map_cube[0]          # (ny, nx)
        eivar = ivar_cube[0]    # (ny, nx)
    else:
        # if multiple lines are given
        emap = map_cube
        eivar = ivar_cube
    
    # Fit 2D WCS
    wcs2d: Optional[WCS]
    if wcs_strategy == "fit" and "ra_deg" in fibers.meta and "dec_deg" in fibers.meta:
        ra_deg = np.asarray(fibers.meta["ra_deg"], float)
        dec_deg = np.asarray(fibers.meta["dec_deg"], float)
        wcs2d = fit_celestial_wcs(
            x_arcsec=x,
            y_arcsec=y,
            ra_deg=np.asarray(fibers.meta["ra_deg"], float),
            dec_deg=np.asarray(fibers.meta["dec_deg"], float),
            pixel_size_arcsec=pixel_size_arcsec,
        )
    else:
        wcs2d = None
    
    line_names = fibers.meta.get("line_names", None)
    
    return dict(
        map=emap,
        ivar=eivar,
        neff=neff,
        A=A,
        xg=xg,
        yg=yg,
        inside=inside,
        wave=wave,
        wcs2d=wcs2d,
        line_names=line_names,
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

    WCS (if present) is attached to MAP header and applies
    to the last two axes; the first axis is "line index".
    """

    from pathlib import Path as _P
    path = _P(path)

    M = np.asarray(result["map"], float)
    IV = np.asarray(result["ivar"], float)
    neff = np.asarray(result["neff"], float)
    wave = np.asarray(result["wave"], float)
    wcs2d: Optional[WCS] = result.get("wcs2d", None)
    line_names: Optional[Sequence[str]] = result.get("line_names", None)

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

    # attach spatial WCS to MAP header
    if wcs2d is not None:
        map_hdr = wcs2d.to_header(relax=True)
    else:
        map_hdr = fits.Header()

    map_hdu = fits.ImageHDU(M.astype("f4"), header=map_hdr, name="FLUX")
    ivar_hdu = fits.ImageHDU(IV.astype("f4"), name="IVAR")
    neff_hdu = fits.ImageHDU(neff.astype("f4"), name="NEFF")

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

    hdul = fits.HDUList([primary_hdu, map_hdu, ivar_hdu, neff_hdu, lines_hdu])
    hdul.writeto(path, overwrite=True)
