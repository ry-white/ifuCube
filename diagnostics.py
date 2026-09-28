#!/usr/bin/env python

"""
diagnostics.py
--------------
Diagnostic tools to validate flux conservation and operator behaviour.

The quick version: after any parameter change, call `run_all_checks(result)`
on the dict returned by build_map / build_cube. Four cheap tests:

  1. total flux out / total flux in            -> want 1.0
  2. adjoint identity <Ax, y> vs <x, A^T y>    -> want ~1e-12
  3. partition-of-unity per-fiber ratios       -> want a tight cluster near 1
  4. column sums of A, and dropped-fiber count -> want 1.0 and 0

A note on check 3, because the obvious version of it is wrong: comparing the
input fluxes against `backproject(A, cube)` does NOT give 1. A^T is the
adjoint, not the inverse, so A^T A is not the identity and that ratio comes
out at whatever sum_p w_pk^2 happens to be (~0.19 for a typical LVM kernel).
The quantity that should be 1 is the partition-of-unity redistribution:
share each pixel's flux back among the fibers in proportion to their weight
in that pixel. `partition_of_unity_band_ratio` does that;
`per_fibre_residuals` is the raw adjoint comparison and is only meaningful
relative to itself (e.g. spotting outlier fibers).

Test 1 is the one that catches the failure mode this pipeline is most prone
to: a kernel too narrow for the pixel size drops fibers entirely and loses
their flux with no error raised. Test 2 catches index-ordering and flattening
mistakes in the operator itself.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional

import numpy as np
from scipy import sparse

# Single implementation of the adjoint, imported rather than duplicated.
try:                                   # package import
    from .kernels import backproject
except ImportError:                    # running as a loose script
    from kernels import backproject


def global_flux_ratio(flux_fib: np.ndarray, cube_flux: np.ndarray) -> float:
    """
    Compute total cube_flux / total input fiber flux

    For an ideal flux conserving mapping over the full IFU, this
    should be ~1 within floating-point noise and any masking.

    A value well below 1 means fibers were dropped -- almost always because
    the truncation radius r_cap was smaller than the pixel size, so a fiber's
    disc contained no pixel center. build_projection_A now warns about this,
    but this ratio is the end-to-end confirmation.
    """

    total_in = np.nansum(flux_fib)
    total_out = np.nansum(cube_flux)
    if total_in == 0:
        return np.nan
    return float(total_out / total_in)

def global_flux_ratio_from_A(
    flux_fib: np.ndarray,
    cube_flux: np.ndarray,
    A: sparse.csr_matrix,
    verbose: bool = True,
) -> float:
    """
    Compare total flux in the cube to the total input fibre flux,
    using the same projection operator A that built the cube.

    Parameters
    ----------
    flux_fib : (K, L)
        Fibre flux array used to build the cube.
    cube_flux : (L, ny, nx)
        Reconstructed cube.
    A : (P, K) sparse matrix
        Projection operator used in the forward splat.
    verbose : bool
        Print the intermediate sums.

    Returns
    -------
    ratio : float
        sum(cube) / sum(flux_fib)
    """
    flux0 = np.nan_to_num(flux_fib, nan=0.0, posinf=0.0, neginf=0.0)
    cube0 = np.nan_to_num(cube_flux, nan=0.0, posinf=0.0, neginf=0.0)

    K, L = flux0.shape
    P, K_A = A.shape
    if K_A != K:
        raise ValueError("A.shape[1] must match number of fibres in flux_fib")

    # recompute cube from A for safety
    C_from_A = (A @ flux0).astype(float)        # (P, L)

    # flatten your stored cube to (P, L)
    L_cube, ny, nx = cube0.shape
    if L_cube != L:
        raise ValueError("cube and flux_fib must share wavelength dimension")
    C_cube = cube0.reshape(L, ny*nx).T          # (P, L)

    sum_flux = np.sum(flux0)
    sum_cube = np.sum(C_cube)
    sum_reproj = np.sum(C_from_A)

    if verbose:
        print("sum_flux    :", sum_flux)
        print("sum_cube    :", sum_cube)
        print("sum(A@flux) :", sum_reproj)
        print("cube / (A@F):", sum_cube / sum_reproj if sum_reproj else np.nan)

    if sum_flux == 0:
        return float("nan")
    return float(sum_cube / sum_flux)

def adjoint_dot_product_test(
    A: sparse.csr_matrix,
    fiber_n: int,
    wave_n: int,
    trials_n: int = 3,
    rng_seed: int = 42,
    verbose: bool = True,
) -> float:
    """
    Check numerically that A^T is the adjoint of A via dot products

    For random X (fiber space) and Y (pixel space):

        <A X, Y> approx = <X, A^T Y>

    where <a, b> is the inner product (elementwise product, summed).

    If these agree to ~1e-12 the implementation of the projection operator A
    is self-consistent: this is the test that catches index-ordering errors,
    shape mix-ups, and row/column-major flattening mistakes, which otherwise
    produce plausible-looking maps that are subtly wrong.

    Returns
    -------
    worst : float
        Largest relative mismatch over the trials.
    """
    rng = np.random.default_rng(rng_seed)
    P, K = A.shape
    if K != fiber_n:
        raise ValueError(f"A.shape[1] ({K}) must equal fiber_n ({fiber_n})")

    worst = 0.0
    for t in range(trials_n):
        X = rng.normal(size=(K, wave_n))
        Y = rng.normal(size=(P, wave_n))

        AX = A @ X  # (P, L)
        left = float(np.sum(AX * Y))

        ATY = A.T @ Y
        right = float(np.sum(X * ATY))

        denom = max(1.0, abs(left), abs(right))
        rel = abs(left - right) / denom
        worst = max(worst, rel)
        if verbose:
            print(f"[Adjoint Test {t+1}/{trials_n}] relative mismatch = {rel:.3e}")
    return worst

def column_sum_check(A: sparse.csr_matrix, atol: float = 1e-12) -> Dict[str, Any]:
    """
    Verify the flux-conservation property directly on the operator.

    For a strictly flux-conserving splat, every column of A sums to exactly 1:

        sum_p A[p,k] = 1   for every fiber k

    A column summing to 0 is a fiber whose flux was discarded. A column
    summing to anything else means the normalization did not happen.

    Returns
    -------
    dict with keys n_fibers, n_dropped, min_nonzero, max, all_unit
    """
    col = np.asarray(A.sum(axis=0)).ravel()
    nonzero = col[col > 0]
    return dict(
        n_fibers=int(col.size),
        n_dropped=int(np.count_nonzero(col == 0)),
        min_nonzero=float(nonzero.min()) if nonzero.size else float("nan"),
        max=float(col.max()) if col.size else float("nan"),
        all_unit=bool(nonzero.size and np.allclose(nonzero, 1.0, atol=atol)),
    )

def partition_of_unity_band_ratio(
    A: sparse.csr_matrix,
    flux_fib: np.ndarray,
    cube_flux: np.ndarray,
    wave: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    """
    Partition-of-unity backprojection test on a wavelength band.

    Redistributes each pixel's band-integrated flux back to the fibers in
    proportion to their share of that pixel, and compares against the
    original per-fiber band flux. A per-fiber version of the global flux
    ratio: it localises *which* fibers lost flux.

    Computed entirely sparsely. The previous implementation called
    A.toarray(), which for a full field at 2 arcsec pixels is a
    ~1.4e9-element dense array (about 11 GB) and simply fails to allocate;
    the identity below gives the same answer without it:

        Fb_hat(k) = sum_p [A[p,k] / W(p)] * Cb(p)
                  = (A^T @ (Cb / W))(k)         where W(p) = sum_j A[p,j]

    Parameters
    ----------
    A : (P, K)
        Projection matrix (column normalized)
    flux_fib : (K, L)
        Input fibre spectra
    cube_flux : (L, ny, nx)
        Reconstructed cube
    wave : (L,)
        Wavelength grid
    band : (lo, hi)
        Band in same units as wave

    Returns
    -------
    ratio : (K,)
        For each fibre k, ratio = Fb_hat(k) / Fb_true(k)
        where Fb is band-integrated flux
    """
    cube_flux = np.asarray(cube_flux, float)
    flux_fib  = np.asarray(flux_fib, float)
    wave      = np.asarray(wave, float)

    L, ny, nx = cube_flux.shape
    P, K = A.shape
    if P != ny * nx:
        raise ValueError("A.shape[0] must equal ny*nx")

    lo, hi = band
    band_mask = (wave >= lo) & (wave <= hi)
    if not np.any(band_mask):
        return np.full(flux_fib.shape[0], np.nan)

    # band-integrated cube -> flatten
    Cb = np.nansum(cube_flux[band_mask], axis=0).reshape(P)  # (P,)

    # band-integrated fibre flux
    Fb = np.nansum(flux_fib[:, band_mask], axis=1)           # (K,)

    # W(p) = sum_j A[p,j]; inf for empty pixels so the quotient is 0
    W = np.asarray(A.sum(axis=1)).ravel()                    # (P,)
    W = np.where(W > 0, W, np.inf)

    # Fb_hat(k) = sum_p Share[p,k] * Cb(p), Share[p,k] = A[p,k] / W(p)
    Fb_hat = np.asarray(A.T @ (Cb / W)).ravel()              # (K,)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Fb != 0, Fb_hat / Fb, np.nan)

    return ratio

def per_fibre_residuals(
    flux_fib: np.ndarray,
    flux_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute per-fibre integrated flux ratio F_hat/F_true.

    Parameters
    ----------
    flux_fib : (K, L)
    flux_hat : (K, L)

    Returns
    -------
    ratio : (K,) array
        Integrated over lambda: sum F_hat / sum F_true.
    """
    flux_fib = np.asarray(flux_fib, float)
    flux_hat = np.asarray(flux_hat, float)
    if flux_fib.shape != flux_hat.shape:
        raise ValueError(
            f"flux_fib {flux_fib.shape} and flux_hat {flux_hat.shape} must "
            "have the same shape (K, L). Note kernels.backproject returns "
            "(K, L)."
        )
    F_true_int = np.nansum(flux_fib, axis=1)
    F_hat_int  = np.nansum(flux_hat, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(F_true_int != 0,
                         F_hat_int / F_true_int,
                         np.nan)
    return ratio

# Kept as an alias: this used to be a byte-identical second copy of
# per_fibre_residuals under a different spelling of "fibre".
per_fiber_backprojection_ratios = per_fibre_residuals


def run_all_checks(
    result: Dict[str, Any],
    flux_fib: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run every cheap validation on a build_map / build_cube result.

    Parameters
    ----------
    result : dict
        The dict returned by build_map or build_cube. Must contain "A" and
        either "cube" or "map"; "flux_in" is used as the input fluxes if
        `flux_fib` is not given.
    flux_fib : (K, L), optional
        Input fiber fluxes, if not present in `result` as "flux_in".
    verbose : bool
        Print a readable report.

    Returns
    -------
    dict with keys flux_ratio, adjoint_worst, columns, neff_stats,
    per_fibre (array), passed (bool)
    """
    A = result["A"]
    data = result.get("cube", result.get("map", None))
    if data is None:
        raise KeyError("result must contain 'cube' or 'map'")
    data = np.asarray(data, float)
    if data.ndim == 2:                      # single-plane map
        data = data[None, ...]

    if flux_fib is None:
        flux_fib = result.get("flux_in", None)
    if flux_fib is None:
        raise KeyError(
            "no input fluxes: pass flux_fib= or ensure result['flux_in'] exists"
        )
    flux_fib = np.asarray(flux_fib, float)

    K, L = flux_fib.shape
    ny, nx = data.shape[1], data.shape[2]

    cols = column_sum_check(A)
    ratio = global_flux_ratio(flux_fib, data)
    adj = adjoint_dot_product_test(A, fiber_n=K, wave_n=min(L, 3),
                                   trials_n=2, verbose=False)

    # Partition-of-unity redistribution over the whole wavelength range. This
    # is the per-fiber quantity that should be 1; the raw adjoint ratio should
    # not be (see the module docstring).
    wave = np.asarray(result.get("wave", np.arange(L)), float)
    band = (float(np.min(wave)) - 1.0, float(np.max(wave)) + 1.0)
    per_fib = partition_of_unity_band_ratio(A, flux_fib, data, wave, band)

    neff = np.asarray(result.get("neff", np.array([np.nan])), float)
    inside = result.get("inside", None)
    if inside is not None and neff.shape == np.asarray(inside).shape:
        nin = neff[np.asarray(inside, bool)]
    else:
        nin = neff.ravel()
    neff_stats = dict(
        median=float(np.nanmedian(nin)) if nin.size else float("nan"),
        frac_zero=float(np.mean(nin == 0)) if nin.size else float("nan"),
    )

    pf_med = float(np.nanmedian(per_fib)) if np.isfinite(per_fib).any() else np.nan
    passed = (
        abs(ratio - 1.0) < 1e-6
        and cols["n_dropped"] == 0
        and cols["all_unit"]
        and adj < 1e-9
        and np.isfinite(pf_med) and abs(pf_med - 1.0) < 1e-2
    )

    if verbose:
        print("=" * 62)
        print("ifuCube validation")
        print("=" * 62)
        print(f"  fibers K={K}  planes L={L}  grid {nx} x {ny}  A.nnz={A.nnz:,}")
        print(f"  1. total flux out / in     : {ratio:.12f}   "
              f"{'OK' if abs(ratio-1) < 1e-6 else 'FAIL -- flux is being lost'}")
        print(f"  2. adjoint worst mismatch  : {adj:.3e}   "
              f"{'OK' if adj < 1e-9 else 'FAIL -- operator/transpose inconsistent'}")
        print(f"  3. per-fiber partition     : median {pf_med:.6f}, "
              f"5-95% {np.nanpercentile(per_fib,5):.4f}.."
              f"{np.nanpercentile(per_fib,95):.4f}   "
              f"{'OK' if np.isfinite(pf_med) and abs(pf_med-1) < 1e-2 else 'FAIL'}")
        print(f"  4. column sums             : {cols['n_dropped']} dropped of "
              f"{cols['n_fibers']}, all unit: {cols['all_unit']}")
        print(f"     N_eff inside footprint  : median {neff_stats['median']:.2f}, "
              f"{100*neff_stats['frac_zero']:.1f}% empty")
        print("=" * 62)
        print(f"  {'ALL CHECKS PASSED' if passed else 'CHECKS FAILED -- see above'}")
        print("=" * 62)

    return dict(flux_ratio=ratio, adjoint_worst=adj, columns=cols,
                neff_stats=neff_stats, per_fibre=per_fib, passed=passed)
