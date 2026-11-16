#!/usr/bin/env python

"""
kernels.py
----------
Flux-conserving projection kernels for IFU cube building

Core object: projection matrix A (shape P x K)
----------------------------------------------
- K = number of fibers
- P = number of spatial pixels = ny * nx

For a -single- wavelength:
    f : (K,) fiber fluxes
    c : (P,) cube pixel fluxes

    c = A @ f

for -all- wavelengths if,
    F : (K, L(ambda)) fiber spectra
    C : (P, L) pixel spectra

    then

    C = A @ F

and we reshape C -> (L, ny, nx) to obtain the cube

Flux conservation
-----------------
We ensure each column of A sums to 1:

    sum_p A[p,k] = 1 for each fiber k

so each fiber's total flux during the projection
is redistributed among pixels without net gain or loss

Call it "column-normalized" Shepard splat

What column normalization does and does not give you
----------------------------------------------------
Because each *column* sums to 1, the total flux out equals the total flux in
exactly, and a pixel value is an *amount of light per pixel* -- not a surface
brightness. A consequence that is easy to miss: a pixel value is therefore
proportional to the pixel's solid angle. Rebuild the same data at half the
pixel size and every value drops by a factor of four while the total stays
fixed. Anything that later *averages* pixel values (a HEALPix binning, a
reprojection onto a different grid) must divide by the pixel solid angle
first, or its output silently depends on the pixel size chosen here. See
`coverage_from_A` for the row sums, which are what you divide by to get the
other convention (a weighted average / surface brightness).

Two separate parameter constraints
----------------------------------
1. FLUX CONSERVATION requires the truncation radius r_cap to be large enough
   that every fiber's disc contains at least one pixel *center*. A fiber with
   no pixel center inside r_cap contributes nothing and its flux is lost.
   Worst case a point sits at the center of four pixels, so r_cap must exceed
   pixel_size / sqrt(2); allow r_cap >= pixel_size for margin, because the
   pixel also has to be inside the footprint mask. build_projection_A counts
   and reports every fiber it drops.
2. SPATIAL SAMPLING requires the kernel to be comparable to the fiber
   spacing (~37 arcsec for LVM). A kernel much narrower than the spacing
   conserves flux perfectly while painting isolated dots with empty gaps
   between them. Set fwhm_arcsec from the fiber spacing, not from the pixel
   size.
"""

from __future__ import annotations
from typing import Tuple
import warnings

import numpy as np
from scipy import sparse

def _sigma_from_fwhm(fwhm: float) -> float:
    """Convert Gaussian FWHM to sigma. 2.3548 = 2*sqrt(2*ln 2)."""
    return float(fwhm) / 2.3548

def build_projection_A(
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    inside_mask: np.ndarray,
    fwhm_arcsec: float,
    fiber_radius_arcsec: float,
    r_sigma_mult: float = 3.0,
    mode: str = "gauss",
    cap_at_fiber: bool = False,
    max_dropped_frac: float = 0.02,
    warn_undersampled: bool = True,
    verbose: bool = True,
) -> sparse.csr_matrix:
    """
    Build a sparse projection matrix A (shape P x K), where
        P = ny*nx (pixels) and K = number of fibers

    Column k holds all weights w_pk for fiber k, distributed among
    pixels p near that fiber. The weights are built from a radial kernel
    and then *normalized per column* so that the sum_p w_pk = 1
    i.e. the sum over all pixels p of the weights for each fiber k = 1

    Kernel shape:
    -------------
        - "gauss" : w(r) = exp(-0.5 * (r/sigma)^2)
        - "tophat": w(r) = 1 for r <= r_cap, else 0

    support: r <= r_cap where

        sigma = FWHM / 2.3548
        r_cap = r_sigma_mult * sigma                      (cap_at_fiber=False)
        r_cap = min(r_sigma_mult * sigma, 1.2 * R_fiber)   (cap_at_fiber=True)

    Note on `cap_at_fiber`: earlier versions of this docstring described the
    capped form while the code always used the uncapped one. The uncapped
    form is the behaviour every existing map in this project was built with,
    so it remains the default; set cap_at_fiber=True to restrict each fiber's
    support to roughly its own aperture. Normalization happens *after*
    truncation either way, so capping never loses flux -- it only makes the
    kernel narrower, which brings constraint 1 in the module docstring into
    play.

    Normalization is also applied after the `inside_mask` is imposed. For a
    fiber near the footprint edge, weights landing on masked-out pixels are
    dropped and the survivors rescaled to sum to 1, so the fiber's whole flux
    is deposited on pixels inside the mask. Total flux is conserved; the
    spatial distribution right at the boundary is pushed slightly inward.
    The NEFF map shows where this is happening.

    Parameters
    ----------
    x_arcsec, y_arcsec : (K,)
        Fiber centers in arcsec in the IFU plane
    xg, yg : 1d arrays
        Pixel center coordinates (arcsec) along x and y, ascending.
    inside_mask : (ny, nx) bool
        True inside the IFU footprint
    fwhm_arcsec : float
        Kernel FWHM (arcsec)
    fiber_radius_arcsec : float
        Physical fiber radius (arcsec). Only used when cap_at_fiber=True.
    r_sigma_mult : float
        Multiple of sigma to use for truncation radius
    mode : {"gauss", "tophat"}
        Kernel type.
    cap_at_fiber : bool
        See above. Default False = historical behaviour.
    max_dropped_frac : float
        If more than this fraction of fibers receive no pixels, raise a
        ValueError instead of silently returning a matrix that throws away
        their flux. Set to 1.0 to disable the guard (it will still warn).
    warn_undersampled : bool
        Warn when the kernel FWHM is much smaller than the typical fiber
        spacing, which yields a map of disconnected dots.
    verbose : bool
        Print a one-line summary of dropped fibers when any are dropped.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Projection matrix with column-normalized weights

    Raises
    ------
    ValueError
        If the fraction of fibers dropped exceeds max_dropped_frac.
    """

    x_arcsec = np.asarray(x_arcsec, float)
    y_arcsec = np.asarray(y_arcsec, float)
    xg = np.asarray(xg, float)
    yg = np.asarray(yg, float)
    inside_mask = np.asarray(inside_mask, bool)

    if x_arcsec.shape != y_arcsec.shape:
        raise ValueError("x_arcsec and y_arcsec must have the same shape")
    if mode not in ("gauss", "tophat"):
        raise ValueError("mode must be 'gauss' or 'tophat'")

    sigma = _sigma_from_fwhm(fwhm_arcsec)
    r_cap = r_sigma_mult * sigma
    if cap_at_fiber:
        r_cap = min(r_cap, 1.2 * float(fiber_radius_arcsec))
    if not (r_cap > 0.0) or not (sigma > 0.0):
        raise ValueError(
            f"non-positive kernel: fwhm={fwhm_arcsec}, sigma={sigma}, r_cap={r_cap}"
        )

    ny, nx = inside_mask.shape
    K = x_arcsec.size
    P = ny * nx

    ps_x = float(xg[1] - xg[0]) if xg.size > 1 else r_cap
    ps_y = float(yg[1] - yg[0]) if yg.size > 1 else r_cap
    ps = max(ps_x, ps_y)

    # --- constraint 2: is the kernel wide enough to fill the gaps? --------
    if warn_undersampled and K >= 4:
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(np.column_stack([x_arcsec, y_arcsec])).query(
                np.column_stack([x_arcsec, y_arcsec]), k=2
            )
            spacing = float(np.median(d[:, 1]))
        except Exception:
            spacing = np.nan
        if np.isfinite(spacing) and spacing > 0 and fwhm_arcsec < 0.5 * spacing:
            warnings.warn(
                f"build_projection_A: kernel FWHM {fwhm_arcsec:.2f}\" is much "
                f"narrower than the median fiber spacing {spacing:.2f}\". Flux "
                "will still be conserved, but each fiber paints an isolated "
                "spot and most of the footprint will stay empty. Consider "
                f"fwhm_arcsec ~ {spacing:.0f}.",
                RuntimeWarning,
            )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    n_dropped = 0
    r_cap2 = r_cap * r_cap
    inv_two_sigma2 = 0.5 / (sigma * sigma)

    for k in range(K):
        xk = x_arcsec[k]
        yk = y_arcsec[k]

        # Only look at the window of the grid that can possibly be within
        # r_cap of this fiber. xg/yg are ascending, so searchsorted finds the
        # window in O(log n); this replaces evaluating the distance to every
        # one of the P pixels for every fiber (which is K*P work, almost all
        # of it discarded).
        i0 = int(np.searchsorted(xg, xk - r_cap, side="left"))
        i1 = int(np.searchsorted(xg, xk + r_cap, side="right"))
        j0 = int(np.searchsorted(yg, yk - r_cap, side="left"))
        j1 = int(np.searchsorted(yg, yk + r_cap, side="right"))
        if i0 >= i1 or j0 >= j1:
            n_dropped += 1
            continue

        dx = xg[i0:i1] - xk                      # (nxw,)
        dy = yg[j0:j1] - yk                      # (nyw,)
        rr2 = dy[:, None] ** 2 + dx[None, :] ** 2   # (nyw, nxw), no sqrt needed
        msk = (rr2 <= r_cap2) & inside_mask[j0:j1, i0:i1]
        if not np.any(msk):
            n_dropped += 1
            continue

        if mode == "gauss":
            w = np.exp(-inv_two_sigma2 * rr2[msk])
        else:  # tophat
            w = np.ones(int(np.count_nonzero(msk)), float)

        s = w.sum()
        if s <= 0:
            n_dropped += 1
            continue

        # column normalization: sum_p w_pk = 1
        w = w / s

        jj, ii = np.nonzero(msk)
        pix_idx = (jj + j0) * nx + (ii + i0)   # flatten (j, i) -> pixel index

        rows.extend(pix_idx.tolist())
        cols.extend([k] * pix_idx.size)
        data.extend(w.astype(float).tolist())

    # --- constraint 1: did we silently throw any fiber's flux away? -------
    if n_dropped:
        frac = n_dropped / max(K, 1)
        msg = (
            f"build_projection_A: {n_dropped} of {K} fibers ({100*frac:.1f}%) "
            f"received no pixels, so their flux is LOST. r_cap={r_cap:.2f}\" "
            f"vs pixel size {ps:.2f}\" (need r_cap > {ps/np.sqrt(2):.2f}\" at "
            f"minimum, and >= {ps:.2f}\" for margin against the footprint "
            f"mask). Either raise fwhm_arcsec/r_sigma_mult or use smaller "
            f"pixels."
        )
        if frac > float(max_dropped_frac):
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)
        if verbose:
            print(f"[warn] {msg}")

    A = sparse.csr_matrix((data, (rows, cols)), shape=(P, K))
    return A

def splat_cube(
    flux_fib: np.ndarray,
    A: sparse.csr_matrix,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Apply projection operator A to fiber spectra to obtain cube

    Given:
        F : (K, L(ambda)) fiber spectra
        A : (P, K) projection matrix

    Compute:
        C = A @ F          -> shape (P, L)
        cube = reshape C^T -> (L, ny, nx)

    Parameters
    ----------
    flux_fib : (K, L)
        Fiber flux array. Must be finite: a single NaN here propagates to
        every pixel in that fiber's kernel support. The builders scrub the
        input with np.nan_to_num before calling this.
    A : (P, K)
        Projection matrix
    ny, nx : int
        spatial dimensions

    # Note this can also be used for building emission line maps too
    with a little trickery in array slicing

    Returns
    -------
    cube_flux : (L, ny, nx)
    """

    flux_fib = np.asarray(flux_fib, float)
    if flux_fib.ndim != 2:
        raise ValueError(f"flux_fib must be 2D (K, L), got shape {flux_fib.shape}")
    K, L = flux_fib.shape
    P, K_A = A.shape
    if K_A != K:
        raise ValueError(
            f"A.shape[1] ({K_A}) must equal no. of fibers ({K})"
        )
    if P != ny * nx:
        raise ValueError(f"A.shape[0] ({P}) must equal ny*nx ({ny*nx})")

    C = (A @ flux_fib).astype(float) # shape (P, L)
    cube_flux = C.T.reshape(L, ny, nx)
    return cube_flux


def ivar_through_A(
    ivar_fib: np.ndarray,
    A: sparse.csr_matrix,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Propagate per-fiber inverse variance through A

    For a single wavelength lambda, the variance of pixel p is:

        Var_out(p) = sum_k [ w_pk ^2 * Var_in(k)]

    where w_pk = A[p,k]. This assumes fiber errors are independent.
    Note the weights enter *squared*: scaling a random quantity by w scales
    its standard deviation by w and hence its variance by w^2. Using w rather
    than w^2 here would understate the output errors by roughly the number of
    contributing fibers.

    In matrix notation, for each lambda:

        Var_out = A^2 @ Var_in

    Where A^2 is the elementwise square of A

    A fiber with ivar = 0 carries no information, so its variance is infinite
    and any pixel it touches comes out with ivar = 0 even if other fibers also
    contributed. That is correct for a *sum*, but it means a few dead fibers
    can carve zero-weight holes into an otherwise good IVAR map. Dropping
    zero-ivar fibers from the FiberSet before A is built (see
    io_lvm.read_* drop_nonfinite) avoids that.

    Parameters
    ----------
    ivar_fib : (K, L)
        Inverse variance per fiber and wavelength
    A : (P, K)
        Projection matrix
    ny, nx : int
        spatial dimensions

    Returns
    -------
    ivar_out : (L, ny, nx)
    """

    ivar_fib = np.asarray(ivar_fib, float)
    if ivar_fib.ndim != 2:
        raise ValueError(f"ivar_fib must be 2D (K, L), got shape {ivar_fib.shape}")
    K, L = ivar_fib.shape
    P, K_A = A.shape
    if K_A != K:
        raise ValueError(
            f"A.shape[1] ({K_A}) must equal number of fibers ({K})"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        var_in = np.where(ivar_fib > 0, 1.0 / ivar_fib, np.inf)

    A2 = A.multiply(A) # element wise square
    Var_out = A2 @ var_in # shape (P, L)

    with np.errstate(divide="ignore", invalid="ignore"):
        ivar_out = np.where(np.isfinite(Var_out) & (Var_out > 0),
                            1.0 / Var_out, 0.0)

    ivar_cube = ivar_out.T.reshape(L, ny, nx)
    return ivar_cube

def coverage_from_A(A: sparse.csr_matrix, ny: int, nx: int) -> np.ndarray:
    """
    Row sums of A: the total kernel weight landing in each pixel.

        W(p) = sum_k A[p,k]

    This is the bridge between the two normalization conventions. The cube
    this package builds is flux-conserving (columns of A sum to 1), so a
    pixel value is flux per pixel. Dividing by W gives the row-normalized
    quantity instead -- a weighted average of the contributing fibers, i.e.
    an estimate of surface brightness in fiber-aperture units, which does not
    depend on the pixel size:

        weighted_average(p) = flux_conserving(p) / W(p)

    W is 0 for pixels no fiber reached, so guard the division.

    Returns
    -------
    W : (ny, nx)
    """
    W = np.asarray(A.sum(axis=1)).ravel()
    return W.reshape(ny, nx)

def neff_from_A(A: sparse.csr_matrix, ny:int, nx:int) -> np.ndarray:
    """
    Effective number of fibers per pixel

        N_eff(p) = (sum_k w_pk)^2 / (sum_k w_pk^2)

    Standard effective sample size (Kish) for weighted means. Ten fibers
    contributing equally give N_eff = 10; one fiber dominating gives
    N_eff = 1. Pixels no fiber reached give 0.

    This is the coverage diagnostic to trust when judging a map: low N_eff
    marks the footprint edge, gaps between fibers, and regions where dead
    fibers were removed. It is also the honest count of independent
    measurements -- neighbouring pixels share fibers, so per-pixel IVAR
    understates the correlation between them.

    Returns
    -------
    Neff : (ny, nx)
    """
    wsum = np.asarray(A.sum(axis=1)).ravel()
    w2sum = np.asarray(A.multiply(A).sum(axis=1)).ravel() + 1e-300
    Neff = (wsum * wsum) / w2sum
    return Neff.reshape(ny, nx)

def backproject(A: sparse.csr_matrix, cube_flux: np.ndarray) -> np.ndarray:
    """
    Back-Project cube pixels onto fibers using A^T

    Forward:
        C = A @ F (pixels from fibers)

    Adjoint:
        F_hat = A^T @ C

    For all wavelengths at once, treat each wavelength slice
    as a vector and apply A^T to each slice (matrix mult.)

    This is the *adjoint*, not the inverse: applying A then A^T does not
    recover the input fiber fluxes, because information was destroyed when
    overlapping kernels added together. What it answers is "given values on
    the pixel grid, what does each fiber see through its own kernel?", which
    is the right operation for comparing a model image against fiber data,
    and the basis of the adjoint dot-product test in diagnostics.py.

    Parameters
    ----------
    A : (P, K)
        Projection matrix
    cube_flux : (L, ny, nx)
        Reconstructed cube

    Returns
    -------
    F_hat : (K, L)
        Backprojected fiber spectra
    """

    cube_flux = np.asarray(cube_flux, float)
    if cube_flux.ndim != 3:
        raise ValueError(
            f"cube_flux must be 3D (L, ny, nx), got shape {cube_flux.shape}"
        )
    L, ny, nx = cube_flux.shape
    P, K = A.shape
    if P != ny * nx:
        raise ValueError(f"A.shape[0] ({P}) must equal ny*nx ({ny*nx})")

    C = cube_flux.reshape(L, P)   # (L, P) lambda, pixel
    F_hat = A.T @ C.T             # (K, P) @ (P, L) -> (K, L) fiber, lambda
    return F_hat                  # (K, L), as documented
