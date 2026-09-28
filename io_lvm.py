#!/usr/bin/env python

"""
io_lvm.py
---------
Read LVM SFrame / DAP files and convert to a generic FiberSet

Assumes the SFrame FITS file has:
    HDU "FLUX" -> (wave_n, fiber_n) or vice versa
    HDU "IVAR" -> same
    HDU "WAVE" -> wavelength grid
    HDU "SLITMAP" -> table with fiberid, ra, dec

Also assumes the LVM fiducial fibermap YAML with the metrology
(targettype, fibstatus, etc.)

Fiber positions
---------------
Positions come from the SLITMAP's own ra/dec columns, converted to
tangent-plane arcsec by geometry.radec_to_tangent_arcsec. They are NOT
derived from xpmm/ypmm and a plate scale: those RA/Dec values are a
measurement that already includes the pointing solution and field
distortion, whereas a plate scale is an assumption about the instrument.
An incorrect assumed plate scale (8.6 arcsec/mm) previously made the field
come out about 13x too small. The plate-scale parameter has been removed
from these readers entirely rather than left in place unused.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import warnings

import yaml
import numpy as np
from astropy.io import fits
import polars as pl

from lvmdap.dap_tools import read_DAP_file
from .geometry import radec_to_tangent_arcsec
from .io_readers import FiberSet

# name : (flux_col, err_col, rest_wavelength_A)
# Module level so build.py and friends can validate --lines against it and
# print the available names.
LINE_DEFS = {
    "Halpha":      ("flux_Halpha_6562.85",      "e_flux_Halpha_6562.85",      6562.85),
    "Hbeta":       ("flux_Hbeta_4861.36",       "e_flux_Hbeta_4861.36",       4861.36),

    "[NII]6583":   ("flux_[NII]_6583.45",       "e_flux_[NII]_6583.45",       6583.45),
    "[NII]6548":   ("flux_[NII]_6548.05",       "e_flux_[NII]_6548.05",       6548.05),

    "[OIII]5007":  ("flux_[OIII]_5006.84",      "e_flux_[OIII]_5006.84",      5006.84),

    "[SII]6716":   ("flux_[SII]_6716.44",       "e_flux_[SII]_6716.44",       6716.44),
    "[SII]6731":   ("flux_[SII]_6730.82",       "e_flux_[SII]_6730.82",       6730.82),
}


def _good_science_mask(fiberid: np.ndarray, fibermap_yaml: str | Path) -> np.ndarray:
    """
    Boolean mask of fibers that are science targets with fibstatus == 0.

    Joins this exposure's fiber list against the fiducial fibermap. Any
    fiberid missing from the fibermap yields a null on the join, which would
    otherwise turn into an ambiguous value on conversion to numpy -- those
    are filled with False and reported.
    """
    with open(fibermap_yaml) as f:
        yml = yaml.safe_load(f)

    columns = [c["name"] for c in yml["schema"]]
    fibs = pl.DataFrame(yml["fibers"], schema=columns, orient="row")

    joined = pl.DataFrame({"fiberid": np.asarray(fiberid, int)}).join(
        fibs, on="fiberid", how="left"
    )

    n_unmatched = int(joined["targettype"].is_null().sum())
    if n_unmatched:
        warnings.warn(
            f"{n_unmatched} of {len(joined)} fiberids are absent from the "
            f"fibermap YAML and are being dropped.",
            RuntimeWarning,
        )

    ok = (
        (joined["targettype"].str.to_lowercase() == "science")
        & (joined["fibstatus"] == 0)
    ).fill_null(False).to_numpy()

    return np.asarray(ok, bool)


def _drop_nonfinite(flux, ivar, ra, dec, fiberid, label="fibers"):
    """
    Remove fibers whose flux or ivar is not finite anywhere useful.

    Worth doing at read time rather than patching later: the projection is a
    matrix multiply, so one NaN flux propagates to every pixel in that
    fiber's kernel support. Zeroing it instead (np.nan_to_num) stops the NaN
    spreading but leaves the fiber holding kernel weight while contributing
    no light, which shows up as a local flux deficit. Removing it from the
    FiberSet keeps it out of A altogether.
    """
    finite_any = np.isfinite(flux).any(axis=1) & np.isfinite(ivar).any(axis=1)
    keep = finite_any & np.isfinite(ra) & np.isfinite(dec)
    n_bad = int((~keep).sum())
    if n_bad:
        print(f"[info] dropped {n_bad} of {keep.size} {label} with non-finite "
              f"flux/ivar/position")
    return flux[keep], ivar[keep], ra[keep], dec[keep], fiberid[keep]


def read_lvm_sframe(
    sframe_path: str | Path,
    fibermap_yaml: str | Path,
    drop_nonfinite: bool = True,
) -> FiberSet:
    """
    Read an LVM SFrame file + fibermap YAML and return FiberSet

    Parameters
    ----------
    sframe_path : path to SFrame FITS
    fibermap_yaml : path to fiducial fibermap YAML
    drop_nonfinite : bool
        Remove fibers with non-finite flux/ivar/position before returning.

    Returns
    -------
    FiberSet
    """

    sframe_path = Path(sframe_path)
    fibermap_yaml = Path(fibermap_yaml)

    with fits.open(sframe_path) as hdul:
        flux = np.asarray(hdul["FLUX"].data)
        ivar = np.asarray(hdul["IVAR"].data)
        wave = np.asarray(hdul["WAVE"].data).astype("float32")
        slitmap = hdul["SLITMAP"].data

    # enforce (fiber_n, wave_n)
    if flux.shape[0] == wave.size and flux.shape[1] != wave.size:
        flux = flux.T
        ivar = ivar.T
    elif flux.shape[1] != wave.size:
        raise ValueError(
            f"SFrame FLUX shape {flux.shape} matches WAVE ({wave.size}) on "
            "neither axis"
        )

    # Fiber IDs from SFrame
    fiberid = np.asarray(slitmap["fiberid"], int)

    # The fibermap mask is built in SLITMAP row order, so it must line up
    # with the FLUX rows.
    if fiberid.size != flux.shape[0]:
        raise ValueError(
            f"SLITMAP has {fiberid.size} rows but FLUX has {flux.shape[0]} "
            "fibers; cannot align the fibermap mask"
        )

    ok = _good_science_mask(fiberid, fibermap_yaml)

    flux = flux[ok]
    ivar = ivar[ok]
    ra = np.asarray(slitmap["ra"], float)[ok]
    dec = np.asarray(slitmap["dec"], float)[ok]
    fid = fiberid[ok]

    if drop_nonfinite:
        flux, ivar, ra, dec, fid = _drop_nonfinite(
            np.asarray(flux, float), np.asarray(ivar, float), ra, dec, fid,
            label="SFrame fibers")

    x_arcsec, y_arcsec = radec_to_tangent_arcsec(ra, dec)

    return FiberSet(
        flux=flux,
        ivar=ivar,
        wave=wave,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
        meta=dict(ra_deg=ra, dec_deg=dec, fiberid=fid)
    )


def read_dap_multi(
    dap_path: str | Path,
    fibermap_yaml: str | Path,
    lines: list[str] | None = None,
    drop_nonfinite: bool = True,
) -> FiberSet:
    """
    Read an LVM dap file + fibermap YAML and return FiberSet
    with *one or more* emission lines.

    Parameters
    ----------
    dap_path : path to dap FITS

    fibermap_yaml : path to fiducial fibermap YAML

    lines : list of str, optional
        Logical line names to extract (keys of LINE_DEFS). Defaults to
        ["Halpha"].

    drop_nonfinite : bool
        Remove fibers with non-finite flux/ivar/position. DAP line fits fail
        on individual fibers routinely, and each bad fiber would otherwise
        NaN every pixel in its kernel support.

    Returns
    -------
    FiberSet
        flux : (K, N_lines)
        ivar : (K, N_lines)
        wave : (N_lines,)   rest wavelengths, used as plane labels
        x_arcsec, y_arcsec : (K,)
        meta : dict with ra_deg, dec_deg, fiberid, line_names
    """

    dap_path = Path(dap_path)
    fibermap_yaml = Path(fibermap_yaml)

    if lines is None:
        lines = ["Halpha"]

    # sanity check: make sure all lines exist in dict, before reading anything
    for ln in lines:
        if ln not in LINE_DEFS:
            raise KeyError(
                f"Line {ln!r} not in LINE_DEFS. "
                f"Available: {sorted(LINE_DEFS.keys())}"
            )

    # read DAP Table
    dap_tbl = read_DAP_file(dap_path)
    fiberid = np.asarray(dap_tbl["fiberid"].data, int)

    # Per-line flux, ivar, and wavelength
    flux_list = []
    ivar_list = []
    wave_list = []
    line_names = []

    for ln in lines:
        flux_col, err_col, lam = LINE_DEFS[ln]

        # 1D array length K (all fibers)
        f = np.asarray(dap_tbl[flux_col], float)
        e = np.asarray(dap_tbl[err_col], float)

        # err -> ivar: err is the 1-sigma error, var = err^2,
        # ivar = 1/var where err > 0, else 0 meaning "no information"
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = np.where(e > 0.0, 1.0 / (e*e), 0.0)

        flux_list.append(f)
        ivar_list.append(ivar)
        wave_list.append(lam)
        line_names.append(ln)

    # Stack to (K, N_lines)
    flux = np.stack(flux_list, axis=1)          # (K, N)
    ivar = np.stack(ivar_list, axis=1)          # (K, N)
    wave = np.array(wave_list, dtype="float32") # (N,)

    # RA/DEC from DAP
    ra = np.asarray(dap_tbl["ra"], float)
    dec = np.asarray(dap_tbl["dec"], float)

    if fiberid.size != flux.shape[0]:
        raise ValueError(
            f"DAP table has {fiberid.size} fiberids but {flux.shape[0]} flux "
            "rows; cannot align the fibermap mask"
        )

    ok = _good_science_mask(fiberid, fibermap_yaml)

    flux = flux[ok]
    ivar = ivar[ok]
    ra = ra[ok]
    dec = dec[ok]
    fid = fiberid[ok]

    if drop_nonfinite:
        flux, ivar, ra, dec, fid = _drop_nonfinite(
            flux, ivar, ra, dec, fid, label="DAP fibers")

    x_arcsec, y_arcsec = radec_to_tangent_arcsec(ra, dec)

    return FiberSet(
        flux=flux,      # (K_good, N_lines)
        ivar=ivar,      # (K_good, N_lines)
        wave=wave,      # (N_lines,)
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
        meta=dict(
            ra_deg=ra,
            dec_deg=dec,
            fiberid=fid,
            line_names=line_names,),
    )


def read_dap(
    dap_path: str | Path,
    fibermap_yaml: str | Path,
    drop_nonfinite: bool = True,
) -> FiberSet:
    """
    Single-line (Halpha) DAP reader.

    Kept for backward compatibility. This used to be a near-copy of
    read_dap_multi; it now delegates, so the two cannot drift apart.

    Returns
    -------
    FiberSet with flux/ivar of shape (K, 1) and wave of shape (1,).
    """
    return read_dap_multi(
        dap_path=dap_path,
        fibermap_yaml=fibermap_yaml,
        lines=["Halpha"],
        drop_nonfinite=drop_nonfinite,
    )
