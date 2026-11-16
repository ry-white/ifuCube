#!/usr/bin/env python

"""
io_readers.py

Generic I/O helpers and FiberSet container

Everything else in the package works with a FiberSet, which
wraps the fundamental IFU quantities needed to build cubes:

    - flux : 2D array, shape (fiber_n, wave_n)
    - ivar : 2D array, same shape as flux, (inverse variance)
    - wave : 1D array, shape (wave_n,) wavelength grid
    - x,y  : 1D array, shape (fiber_n,) fiber positions in arcsec
    - meta : dictionary with extra information (ra,dec, headers, etc.)

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np

@dataclass
class FiberSet: 
    """
    Container for IFU fiber data needed for cube construction.

    Attributes
    ----------
    flux : np.ndarray
        2D array with shape (fiber_n, wave_n). Each row is the spectrum
        of one fiber
    ivar : np.ndarray
        2D array with inverse variance for each flux value.
        Same shape as 'flux'. Zero or negative values mean "no info".
    wave : np.ndarray 
        1D array of observed wavelengths with shape (wave_n,)
    x_arcsec, y_arcsec : np.ndarray
        1D arrays of fiber positions in the IFU plane (arcsec)
    meta : dict
        Arbitrary metadata; useful keys might include:
            - ra, dec, header, etc. can be found. 
    """

    flux : np.ndarray
    ivar: np.ndarray
    wave: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Check the five arrays agree with each other.

        @dataclass runs this automatically after construction, so the checks
        apply however a FiberSet is made -- previously they lived only in
        fibers_from_arrays, which the LVM readers bypass by constructing
        FiberSet directly. A shape mismatch here would otherwise surface much
        later as an unrelated-looking error inside the projection.
        """
        flux = np.asarray(self.flux)
        ivar = np.asarray(self.ivar)
        wave = np.asarray(self.wave)
        x = np.asarray(self.x_arcsec)
        y = np.asarray(self.y_arcsec)

        if flux.ndim != 2:
            raise ValueError(f"flux must be 2D (fiber_n, wave_n), got {flux.shape}")
        if ivar.shape != flux.shape:
            raise ValueError(
                f"ivar shape {ivar.shape} != flux shape {flux.shape}")
        if wave.ndim != 1:
            raise ValueError(f"wave must be 1D (wave_n,), got {wave.shape}")
        if flux.shape[1] != wave.size:
            raise ValueError(
                f"flux.shape[1] ({flux.shape[1]}) != wave.size ({wave.size})")
        if x.shape != y.shape:
            raise ValueError(
                f"x_arcsec {x.shape} and y_arcsec {y.shape} must match")
        if x.ndim != 1:
            raise ValueError(f"x_arcsec/y_arcsec must be 1D, got {x.shape}")
        if x.size != flux.shape[0]:
            raise ValueError(
                f"{x.size} fiber positions but {flux.shape[0]} flux rows")

    @property
    def fiber_n(self) -> int:
        return int(self.flux.shape[0])

    @property
    def n_wave(self) -> int:
        return int(self.flux.shape[1])


def fibers_from_arrays(
    flux: np.ndarray,
    ivar: np.ndarray,
    wave: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
) -> FiberSet:
    """
    Wrap in-memory arrays into a FiberSet

    Generic entry point of RSS

    Parameters
    ----------
    flux, ivar, wave, x_arcsec, y_arcsec :
        Arrays with the shapes described in FiberSet docstring
    meta: dict, optional
        Any extra metadata that is wanted to attach

    Returns
    -------
    FiberSet
    """
    flux = np.asarray(flux, float)
    ivar = np.asarray(ivar, float)
    wave = np.asarray(wave, float)
    x_arcsec = np.asarray(x_arcsec, float)
    y_arcsec = np.asarray(y_arcsec, float)

    # Shape consistency is enforced by FiberSet.__post_init__, so it applies
    # to every FiberSet however it was built, not only to this entry point.

    if meta is None:
        meta = {}

    return FiberSet(
        flux=flux,
        ivar=ivar,
        wave=wave,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
        meta=meta,
    )
