#!/usr/bin/env python

"""
diagnostics.py
--------------
Diagnostic tools to validate flux conservation and operation behavior
"""

from __future__ import annotations
from typing import Tuple

import numpy as np 
import matplotlib.pyplot as plt 
from scipy import sparse

def global_flux_ratio(flux_fib: np.ndarray, cube_flux: np.ndarray) -> float:
    """
    Compute total cube_flux / total input fiber flux

    For an ideal flux conserving mapping over the full IFU, this
    should obviously be ~1 within floating-point noise and any masking
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

    # sanity: this should be almost identical:
    sum_reproj = np.sum(C_from_A)
    print("sum_flux    :", sum_flux)
    print("sum_cube    :", sum_cube)
    print("sum(A@flux) :", sum_reproj)
    print("cube / (A@F):", sum_cube / sum_reproj)

    return float(sum_cube / sum_flux)

def adjoint_dot_product_test(
    A: sparse.csr_matrix,
    fiber_n: int,
    wave_n: int,
    trials_n: int = 3,
    rng_seed: int = 42,
) -> None:
    """
    Check numerically that A^T is the adjoint of A via dot products

    For random X (fiber space) and Y (pixel space):

        <A X, Y> approx = <X, A^T Y>

    where <x, x> is the inner product (or dot product)

    If these agree to 1e-12 ish the implementation of the projection operator
    A is trustworthy
    """
    rng = np.random.default_rng(rng_seed)
    P, K = A.shape
    if K != fiber_n:
        raise ValueError("A.shape[1] must equal fiber_n")
    
    for t in range(n_trials):
        X = rng.normal(size=(K, wave_n))
        Y = rng.normal(size=(P, wave_n))

        AX = A @ X  # (P, L)
        left = float(np.sum(AX * Y))

        ATY = A.T @ Y 
        right = float(np.sum(X * ATY))

        denom = max(1.0, abs(left), abs(right))
        rel = abs(left - right) / denom
        print(f"[Adjoint Test {t+1}/{n_trials}] relative mismatch = {rel:.3e}")

def partition_of_unity_band_ratio(
    A: sparse.csr_matrix,
    flux_fib: np.ndarray,
    cube_flux: np.ndarray,
    wave: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    """
    Partition-of-unity backprojection test on a wavelength band.

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

    # band-integrated cube → flatten
    Cb = np.nansum(cube_flux[band_mask], axis=0).reshape(P)  # (P,)

    # band-integrated fibre flux
    Fb = np.nansum(flux_fib[:, band_mask], axis=1)           # (K,)

    # Weights: W[p,k] = A[p,k]
    W = A.toarray()       # (P, K)
    Wsum = W.sum(axis=1)  # (P,)
    Wsum[Wsum <= 0] = np.inf

    # Share[p,k] = A[p,k] / Σ_j A[p,j]
    Share = (W.T / Wsum).T     # (P, K)

    # Fb_hat(k) = Σ_p Share[p,k] * Cb(p)
    Fb_hat = np.sum(Share * Cb[:, None], axis=0)   # (K,)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Fb != 0, Fb_hat / Fb, np.nan)

    return ratio

def backproject_fibres(
    cube_flux: np.ndarray,
    A: sparse.csr_matrix,
) -> np.ndarray:
    """
    Backproject cube onto fibre space using A^T.

    Parameters
    ----------
    cube_flux : (L, ny, nx)
        Reconstructed cube.
    A : (P, K) projection matrix

    Returns
    -------
    flux_hat : (K, L)
        Reconstructed fibre spectra.
    """
    cube_flux = np.asarray(cube_flux, float)
    L, ny, nx = cube_flux.shape
    P, K = A.shape
    if P != ny * nx:
        raise ValueError("A.shape[0] must equal ny*nx")

    C = cube_flux.reshape(L, P)       # (L, P)
    F_hat = (A.T @ C.T).T             # (L, K) → transpose
    return F_hat.T                    # (K, L)

def per_fiber_backprojection_ratios(flux_fib, flux_hat):
    """
    Compare fibre-integrated flux between original and backprojected spectra.

    flux_fib, flux_hat : (K, L)
    returns ratio : (K,)
    """
    F_true = np.nansum(flux_fib, axis=1)
    F_hat  = np.nansum(flux_hat, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(F_true != 0, F_hat / F_true, np.nan)
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
        Integrated over λ: ∫ F_hat / ∫ F_true.
    """
    F_true_int = np.nansum(flux_fib, axis=1)
    F_hat_int  = np.nansum(flux_hat, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(F_true_int != 0,
                         F_hat_int / F_true_int,
                         np.nan)
    return ratio

