#!/usr/bin/env python

"""
regions.py
----------
Region-based spectrum extraction

Use-cases:
- select fibers in side a polygon/circle in the IFU plane
- combin selected fibers into a single "region spectrum"
- Optionally use A^T to mapy a *cube region* back to fiber space
"""

from __future__ import annotations
from typing import Sequence, Tuple

import numpy as np 
from matplotlib.path import Path as MplPath
from scipy import sparse

def fibers_in_polygon(
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    poly_x: Sequence[float],
    poly_y: Sequence[float],
) -> np.ndarray:
    """
    Boolean mask of fibers inside a given polygon in the IFU (x,y) plane

    Parameters
    ----------
    x_arcsec, y_arcsec : (K,)
        Fiber position in arcsec
    poly_x, poly_y : sequence of float
        Polygon vertices in same coordinate system
    
    Returns
    -------
    mask : (K,) bool
        True for fibers inside the polygon
    """

    x = np.asarray(x_arcsec, float)
    y = np.asarray(y_arcsec, float)
    poly = np.column_stack([poly_x, poly_y])
    path = MplPath(poly, closed=True)
    pts = np.column_stack([x,y])
    return path.contains_points(pts)

def fibers_in_circle(
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    x0_arcsec: float,
    y0_arcsec: float,
    radius_arcsec: float,
) -> np.ndarray:
    """
    Boolean mask of fibers inside circle centered at (x0, y0)

    Parameters
    ----------
    x_arcsec, y_arcsec : (K,)
        Fiber positions
    x0_arcsec, y0_arcsec : float
        Circle in center in arcsec
    radius_arcsec : float
        Radius in arcsec
    """
    x = np.asarray(x_arcsec, float)
    y = np.asarray(y_arcsec, float)
    r2 = (x - x0_arcsec) ** 2 + (y - y0_arcsec) ** 2
    return r2 <= radius_arcsec**2

def region_spectrum_from_fibers(
    flux_fib: np.ndarray,
    ivar_fib: np.ndarray,
    mask_fibers: np.ndarray,
    combine: str = "ivar",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine spectra of selected fibers into a single "region spectrum"

    Parameters
    ----------
    flux_fib : (K, L)
        Fiber flux array
    ivar_fib : (K, L)
        Fiber inverse variance array
    mask_fibers : (K,) bool
        True for fibers to include
    combine : {"ivar", "mean"}
        - "ivar": inverse variance weighted mean per lambda
        - "mean": simple arithmetic mean per lambda
    
    Returns
    -------
    flux_reg, ivar_reg : (L,) , (L,)
        Region spectrum and its inverse variance
    """
    flux_fib = np.asarray(flux_fib, float)
    ivar_fib = np.asarray(ivar_fib, float)
    mask_fibers = np.asarray(mask_fibers, bool)

    if flux_fib.shape != ivar_fib.shape:
        raise ValueError("flux_fib and ivar_fib must have the same shape")
    
    sel = np.where(mask_fibers)[0]
    if sel.size == 0:
        raise ValueError("No fibers selected in region")
    
    F = flux_fib[sel, :] # (N_sel, L)
    I = ivar_fib[sel, :] # ditto

    if combine == "mean":
        flux_reg = np.nanmean(F, axis=0)
        # for a plain mean the variance roughly is mean(var)/N_sel
        with np.errstate(divide="ignore", invalid="ignore"):
            var_reg = np.nanmean(np.where(I > 0, 1.0 / I, np.nan), axis=0) / sel.size
            ivar_reg = np.where(np.isfinite(var_reg) & (var_reg > 0),
                                1.0 / var_reg, 0.0)
    
    elif combine == "ivar":
        # weighted mean with weights = ivar
        wsum = np.nansum(I, axis=0)
        num = np.nansum(I * F, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_reg = np.where(wsum > 0, num / wsum, np.nan)
            ivar_reg = wsum
    
    else:
        raise ValueError("combine must be set to 'mean' or 'ivar'")
    
    return flux_reg, ivar_reg
