#!/usr/bin/env python

"""
projection.py
-------------
The setup step that every builder needs and none of them owns.

Both build_cube() and build_map() used to open with the same ~40 lines:
scrub non-finite input, lay down a grid, build the projection matrix A, fit
a celestial WCS. Measured against each other, 68% of the two functions' code
lines were identical, and the identical part was exactly this prologue. That
is not a sign that cubes and maps are the same operation -- what follows the
prologue is genuinely different (cubes chunk over ~12,000 wavelength planes
and carry a 3D WCS; maps carry line names, kinematics and coverage
normalisation). It is a sign that the shared step had no home.

It has one now. `prepare_grid()` returns everything a builder needs before
it projects anything, so the builders differ only in what they then do with
A -- which is the actual difference between them.

Dependency direction: this module imports geometry and kernels; neither
imports this one. kernels imports nothing from the package at all, so the
graph stays a DAG.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
from astropy.wcs import WCS

from .io_readers import FiberSet
from .geometry import make_grid_from_points, fit_celestial_wcs
from .kernels import build_projection_A


def prepare_grid(
    fibers: FiberSet,
    pixel_size_arcsec: float,
    fwhm_arcsec: float,
    fiber_radius_arcsec: float,
    r_sigma_mult: float = 3.0,
    kernel_mode: str = "gauss",
    wcs_strategy: str = "fit",
    max_dropped_frac: float = 0.02,
    dtype=float,
    scrub: bool = True,
) -> Dict[str, Any]:
    """
    Grid, inside-IFU mask, projection matrix and celestial WCS.

    Parameters
    ----------
    fibers : FiberSet
        Input fiber data. `meta["ra_deg"]` and `meta["dec_deg"]` are required
        for wcs_strategy="fit".
    pixel_size_arcsec : float
        Output spaxel size.
    fwhm_arcsec : float
        Kernel FWHM for the splat. Set this from the FIBER SPACING (~37" for
        LVM), not from the pixel size -- a kernel narrower than the fiber
        spacing leaves holes between fibers no matter how fine the grid is.
        See kernels.py.
    fiber_radius_arcsec : float
        Fiber radius; sets the grid padding and the convex-hull expansion.
    r_sigma_mult : float
        Kernel truncation radius in units of sigma. 3.0 keeps 98.9% of a
        Gaussian's 2D volume; the rest is renormalised away by the column
        scaling, so this changes the kernel shape slightly but never breaks
        flux conservation.
    kernel_mode : {"gauss", "tophat"}
        Kernel type.
    wcs_strategy : {"fit", "none"}
        "fit" fits a celestial WCS from the fiber RA/Dec. "none" skips it,
        which is what the test suite uses.
    max_dropped_frac : float
        Forwarded to build_projection_A: refuse to build if more than this
        fraction of fibers would have their flux discarded entirely (a fiber
        whose kernel lands wholly outside the mask). Guards against a silent
        flux leak from a mis-specified grid.
    dtype : numpy dtype
        Precision for the scrubbed flux/ivar arrays. float32 halves peak
        memory and is what gets written to FITS anyway; maps use float64
        because L is a handful of lines rather than ~12,000 planes.
    scrub : bool
        Replace non-finite flux/ivar with 0.0. Leave this True.

    Returns
    -------
    dict with
        "x", "y"     : (K,) fiber positions in arcsec
        "flux"       : (K, L) scrubbed flux, cast to `dtype`
        "ivar"       : (K, L) scrubbed inverse variance, cast to `dtype`
        "wave"       : (L,) wavelength array
        "xg", "yg"   : 1D grid coordinate vectors
        "inside"     : (ny, nx) bool, True inside the expanded fiber hull
        "ny", "nx"   : grid shape
        "A"          : (P, K) sparse projection matrix, columns summing to 1
        "wcs2d"      : celestial WCS, or None
        "K"          : number of fibers

    Notes
    -----
    WHY THE SCRUB MATTERS MORE THAN IT LOOKS. The projection is a matrix
    multiply, so a single NaN fiber flux propagates to EVERY pixel in that
    fiber's kernel support. DAP line fits fail on individual fibers
    routinely, and at fwhm=35" a few percent of bad fibers will NaN out
    roughly a fifth of the map. Zeroing is correct here because flux is
    extensive: a zero contributes zero flux and the column normalisation is
    unaffected.

    This is NOT the right treatment for an intensive quantity. 0 km/s is a
    perfectly valid velocity, so a kinematic map must zero the WEIGHT of a
    bad fiber, not its value -- see build_kinematic_map in mapcat_build.py.

    WHY x0/y0 ARE PASSED TO THE WCS FIT. Array element [0, 0] sits at
    (xg[0], yg[0]) in arcsec, not at (0, 0). Omitting the origin offsets the
    fitted WCS from the array by about a third of a degree, which looks
    plausible on a single field and ruins any mosaic. See geometry.py.
    """
    dtype = np.dtype(dtype)

    wave = np.asarray(fibers.wave, float)
    x = np.asarray(fibers.x_arcsec, float)
    y = np.asarray(fibers.y_arcsec, float)

    if scrub:
        flux = np.nan_to_num(np.asarray(fibers.flux, dtype),
                             nan=0.0, posinf=0.0, neginf=0.0)
        ivar = np.nan_to_num(np.asarray(fibers.ivar, dtype),
                             nan=0.0, posinf=0.0, neginf=0.0)
    else:
        flux = np.asarray(fibers.flux, dtype)
        ivar = np.asarray(fibers.ivar, dtype)

    # 1) Grid + inside-hull mask. fiber_radius_arcsec must be forwarded: it
    # was once omitted here, so the padding and hull expansion silently used
    # the 18.5" default regardless of what the caller asked for.
    xg, yg, inside = make_grid_from_points(
        x_arcsec=x,
        y_arcsec=y,
        pixel_size_arcsec=pixel_size_arcsec,
        fiber_radius_arcsec=fiber_radius_arcsec,
    )
    ny, nx = inside.shape

    # 2) Projection matrix, columns normalised to 1 (flux-conserving).
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

    # 3) Celestial WCS.
    wcs2d: Optional[WCS] = None
    if (wcs_strategy == "fit"
            and "ra_deg" in fibers.meta and "dec_deg" in fibers.meta):
        wcs2d = fit_celestial_wcs(
            x_arcsec=x,
            y_arcsec=y,
            ra_deg=np.asarray(fibers.meta["ra_deg"], float),
            dec_deg=np.asarray(fibers.meta["dec_deg"], float),
            pixel_size_arcsec=pixel_size_arcsec,
            x0_arcsec=xg[0],
            y0_arcsec=yg[0],
        )

    return dict(x=x, y=y, flux=flux, ivar=ivar, wave=wave,
                xg=xg, yg=yg, inside=inside, ny=ny, nx=nx,
                A=A, wcs2d=wcs2d, K=int(flux.shape[0]))


def coverage_plateau(n_fibers: int, inside) -> float:
    """
    The exact mean of the coverage map over the covered footprint.

    DERIVATION. `coverage` is the row sum of A, coverage(p) = sum_k A[p,k].
    The columns of A are normalised to 1 by construction (that is what makes
    the projection flux-conserving), so

        sum_p coverage(p) = sum_p sum_k A[p,k]
                          = sum_k sum_p A[p,k]        (swap the order)
                          = sum_k 1                   (columns sum to 1)
                          = K

    exactly, whatever the kernel width or the pixel size. The mean of
    coverage over the N_inside footprint pixels is therefore K / N_inside,
    with no approximation anywhere.

    WHY THIS FUNCTION EXISTS. That mean is ~0.19 for 15" LVM pixels, not
    ~1. `coverage` is not a fraction of anything and has no fixed ceiling: a
    narrow kernel piles weight into few pixels and pushes the maximum well
    above the mean (measured: 0.51 at FWHM 20", 0.22 at FWHM 60"). So a
    threshold like `coverage > 0.3` masks the entire map, and the value at
    which it would start to bite moves with both FWHM and pixel size.

    Dividing by this plateau gives `coverage_norm`, which is ~1.0 on the
    well-sampled interior and falls to 0 at the edge for ANY kernel width
    and pixel size. Threshold on that, or on `neff`, never on raw coverage.
    """
    n_inside = int(np.count_nonzero(inside))
    if n_inside == 0:
        return float("nan")
    return float(n_fibers) / n_inside


__all__ = ["prepare_grid", "coverage_plateau"]
