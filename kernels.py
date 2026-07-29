#!/usr/bin/env python

"""
kernels.py
----------
Flux-conserving projection kernels for IFU cube building

Core object: projection matrix A (shape P x K)
----------------------------------------------
- K = number of fibers
- P = number of spatial pizels = ny * nx

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
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy import sparse 

def _sigma_from_fwhm(fwhm: float) -> float:
    """Convert Gaussian FWHM to sigma"""
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
) -> sparse.csr_matrix:
    """
    Build a sparse projection matrix A (shape P x K), where
        P = ny*nx (pixels) and K = number of fibers
    
    Column K holds all weights w_pk for fiber k, distributed among
    pixels p near that fiber. The weights are built from a radial kernel
    and then *normalized per column* so that the sum_p w_pk = 1
    i.e. the sum over all pixels p of the weights for each fiber k = 1

    Kernel shape:
    -------------
        - "gauss" : w(r) = exp(-0.5 * (r/sigma)^2)
        - "tophat": w(r) = 1 for r <= r_cap, else 0

    support: r<= r_cap where
        sigma = FWHM / 2.3548
        r_cap = min(r_sigma_mult * sigma, 1.2 * fiber_radius)

    Parameters
    ----------
    x_arcsec, y_arcsec : (K,)
        Fiber centers in arcsec in the IFU plane
    xg, yg : 1d arrays
        Pixel center coordinates (arcsec) along x and y
    inside_mask : (ny, nx) bool
        True inside the IFU footprint
    fwhm_arcsec : float
        Kernel FWHM (arcsec)
    fiber_radius_arcsec : float
        Physical fiber radius (arcsec), used as an upper cap on support
    r_sigma_mult : float
        Multiple of simga to use for truncation radius
    mode : {"gauss", "tophat"}
        Kernel type. 
    
    Returns
    -------
    A : scipy.sparse.csr_matrix
        Projection matrix with column-normalized weights
    """

    x_arcsec = np.asarray(x_arcsec, float)
    y_arcsec = np.asarray(y_arcsec, float)
    xg = np.asarray(xg, float)
    yg = np.asarray(yg, float)
    inside_mask = np.asarray(inside_mask, bool)

    if x_arcsec.shape != y_arcsec.shape:
        raise ValueError("x_arcsec and y_arcsec must have the same shapoe")

    sigma = _sigma_from_fwhm(fwhm_arcsec)
    r_cap = min(r_sigma_mult * sigma, 1.2 * float(fiber_radius_arcsec))

    ny, nx = inside_mask.shape
    X, Y = np.meshgrid(xg, yg)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for k in range(x_arcsec.size):
        rr = np.hypot(X - x_arcsec[k], Y - y_arcsec[k])
        msk = (rr <= r_cap) & inside_mask
        if not np.any(msk):
            continue

        if mode == "gauss":
            w = np.exp(-0.5 * (rr[msk] / sigma) ** 2)
        elif mode == "tophat":
            w = np.ones(np.count_nonzero(msk), float)
        else:
            raise ValueError("mode must be 'gauss' or 'tophat'")

        s = w.sum()
        if s <= 0:
            continue

        # column normalization: sum_p w_pk = 1
        w /= s

        jj, ii = np.nonzero(msk)
        pix_idx = jj * nx + ii # flatten (j, i) -> pixel index

        rows.extend(pix_idx.tolist())
        cols.extend([k] * pix_idx.size)
        data.extend(w.astype(float).tolist())

    P = ny * nx
    K = x_arcsec.size
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
        Fiber flux array
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
    K, L = flux_fib.shape
    P, K_A = A.shape
    if K_A != K:
        raise ValueError("A.shape[1] must equal no. of fibers")
    
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

    For a singel wavelength lambda, the variance of pixel p is:

        Var_out(p) = sum_k [ w_pk ^2 * Var_in(k)]
    
    where w_pk = A[p,k]. This assumes fiber errors are independent

    In matrix notation, for each lambda:

        Var_out = A^2 @ Var_in

    Where A^2 is the elementwise square of A 

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
    K, L = ivar_fib.shape
    P, K_A = A.shape
    if K_A != K:
        raise ValueError("A.shape[1] must equal number of fibers")

    with np.errstate(divide="ignore", invalid="ignore"):
        var_in = np.where(ivar_fib > 0, 1.0 / ivar_fib, np.inf)

    A2 = A.multiply(A) # element wise square
    Var_out = A2 @ var_in # shape (P, L)

    with np.errstate(divide="ignore", invalid="ignore"):
        ivar_out = np.where(np.isfinite(Var_out) & (Var_out > 0),
                            1.0 / Var_out, 0.0)
        
    ivar_cube = ivar_out.T.reshape(L, ny, nx)
    return ivar_cube

def neff_from_A(A: sparse.csr_matrix, ny:int, nx:int) -> np.ndarray:
    """
    Effective number of fibers per pixel

        N_eff(p) = (sum_k w_pk)^2 / (sum_k w_pk^2)
    
    Standard effetice sample size for weighted means 

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
    L, ny, nx = cube_flux.shape
    P, K = A.shape
    if P != ny * nx:
        raise ValueError("A.shape[0] must equal ny*nx")
    
    C = cube_flux.reshape(L, P) # (L, P) lambda, pixel
    F_hat = (A.T @ C.T)         # (L, K) lambda, fiber
    return F_hat.T             # (K, L) fiber, lambda
    
    

