#!/usr/bin/env python

"""
io_lvm.py
---------
Read LVM SFrame file and convert to a generic FiberSet

Assumes the SFrame FITS file has:
    HDU "FLUX" -> (wave_n, fiber_n) or vice verse
    HDU "IVAR" -> same
    HDU "WAVE" -> wavelength grid
    HDU "SLITMAP" -> table with fiberid, ra, dec, xpmm, ypmm

Also assumes that the LVM fiducial fibermap YAML with the metrology
(xpmm, ypmm, status flags, etc.)
"""

from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
from astropy.io import fits
import polars as pl

from lvmdap.dap_tools import read_DAP_file
from .geometry import radec_to_tangent_arcsec
from .io_readers import FiberSet

def read_lvm_sframe(
    sframe_path: str | Path,
    fibermap_yaml: str | Path,
    plate_scale_arcsec_per_mm: float = 8.6,
) -> FiberSet:
    """
    Read an LVM SFrame file + fibermap YAML and return FiberSet

    Parameters
    ----------
    sframe_path : path to SFrame Fits
    fibermap_yaml : path to fiducial fibermap YAML
    plate_scale_arcsec_per_mm : conversion factor for xpmm, ypmm

    Returns
    -------
    FiberSet
    """

    sframe_path = Path(sframe_path)
    fibermap_yaml = Path(fibermap_yaml)

    with fits.open(sframe_path) as hdul:
        flux = hdul["FLUX"].data
        ivar = hdul["IVAR"].data
        wave = hdul["wave"].data.astype("float32")
        slitmap = hdul["SLITMAP"].data
    
    # enforce (fiber_n, wave_n)
    if flux.shape[0] == wave.size:
        flux = flux.T
        ivar = ivar.T
    elif flux.shape[1] != wave.size:
        raise ValueError("SFrame FLUX dimensions do not match WAVE grid")
    
    # Fiber IDs from SFrame
    fiberid = np.asarray(slitmap["fiberid"], int)

    # load fibermap
    with open(fibermap_yaml) as f:
        yml = yaml.safe_load(f)
    
    columns = [c["name"] for c in yml["schema"]]
    fibs = pl.DataFrame(yml["fibers"], schema=columns, orient="row")

    # match by fiberid
    joined = pl.DataFrame({"fiberid": fiberid}).join(fibs, on="fiberid", how="left")

    # select good science fiber == "0"
    ok = (
        (joined["targettype"].str.to_lowercase() == "science")
        & (joined["fibstatus"] == 0)
    ).to_numpy()

    flux = flux[ok]
    ivar = ivar[ok]

    ra = np.asarray(slitmap["ra"], float)[ok]
    dec = np.asarray(slitmap["dec"], float)[ok]

    x_arcsec, y_arcsec = radec_to_tangent_arcsec(ra, dec)

    return FiberSet(
        flux=flux,
        ivar=ivar,
        wave=wave,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
        meta=dict(ra_deg=ra, dec_deg=dec, fiberid=fiberid[ok])
    )

def read_dap(
    dap_path: str | Path,
    fibermap_yaml: str | Path,
    plate_scale_arcsec_per_mm: float = 8.6,
   # wave: float -> want to eventually be able to input wanted wavelengths
   # for map construction
) -> FiberSet:
    """
    Read an LVM dap file + fibermap YAML and return FiberSet

    Parameters
    ----------
    dap_path : path to dap Fits
    fibermap_yaml : path to fiducial fibermap YAML
    plate_scale_arcsec_per_mm : conversion factor for xpmm, ypmm

    Returns
    -------
    FiberSet
    """

    dap_path = Path(dap_path)
    fibermap_yaml = Path(fibermap_yaml)

    dap_tbl = read_DAP_file(dap_path)
    fiberid = np.asarray(dap_tbl["fiberid"].data).astype(int)

    flux_ha = dap_tbl["flux_Halpha_6562.85"]
    flux_ha_err = dap_tbl["e_flux_Halpha_6562.85"] # turn into ivar

    # shape as (K, L) with L = 1 (single line) to use splat_cube
    flux = flux_ha[:, None] #

    # err to ivar conversion
    # err = sigmak (1sigma error), var = sigmak^2
    # ivar = 1/var = 1/sig^2 if sigk > 0, else its 0
    
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar_ha = np.where(flux_ha_err > 0, 1.0 / flux_ha_err**2, 0.0)[:, None]
    
    # define a trivial wavelength "array" with a single element (L=1)
    wave = np.array([6562.85], dtype="float32")

    ra = np.asarray(dap_tbl["ra"], float)
    dec = np.asarray(dap_tbl["dec"], float)

    # load fibermap
    with open(fibermap_yaml) as f:
        yml = yaml.safe_load(f)
    
    columns = [c["name"] for c in yml["schema"]]
    fibs = pl.DataFrame(yml["fibers"], schema=columns, orient="row")

    # match by fiberid
    joined = pl.DataFrame({"fiberid": fiberid}).join(fibs, on="fiberid", how="left")

    # select good science fiber == "0"
    ok = (
        (joined["targettype"].str.to_lowercase() == "science")
        & (joined["fibstatus"] == 0)
    ).to_numpy()

    flux = flux[ok]
    ivar_ha = ivar_ha[ok]
    ra = ra[ok]
    dec = dec[ok]

    x_arcsec, y_arcsec = radec_to_tangent_arcsec(ra, dec)

    return FiberSet(
        flux=flux,      # (K, 1)
        ivar=ivar_ha,      # (K, 1)
        wave=wave,      # (1,)
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
        meta=dict(ra_deg=ra, dec_deg=dec, fiberid=fiberid[ok])
    )

def read_dap_multi(
    dap_path: str | Path, 
    fibermap_yaml: str | Path, 
    plate_scale_arcsec_per_mm: float = 8.6, 
    lines: list[str] | None = None
) -> FiberSet:
    """
    Read an LVM dap file + fibermap YAML and return FiberSet
    with * one or more * emission lines

    Parameters
    ----------
    dap_path : path to dap Fits
    
    fibermap_yaml : path to fiducial fibermap YAML
    
    plate_scale_arcsec_per_mm : conversion factor for xpmm, ypmm

    lines : list of str, optional 
        Logical line names to extract, if None defaults to ["Halpha"]

    Returns
    -------
    FiberSet
        flux : (K, N_lines)
        ivar : (K, N_lines)
        wave : (N_lines,)
        x_arcsec, y_arcsec : (K,)
        meta : dict with ra_deg, dec_deg, fiberid, line_names
    """

    dap_path = Path(dap_path)
    fibermap_yaml = Path(fibermap_yaml)

    # --- Define which DAP columns correspond to which physical lines ---
    # name : (flux_col, err_col, rest_wavelength_A)

    LINE_DEFS = {
    "Halpha":      ("flux_Halpha_6562.85",      "e_flux_Halpha_6562.85",      6562.85),
    "Hbeta":       ("flux_Hbeta_4861.36",       "e_flux_Hbeta_4861.36",       4861.36),

    "[NII]6583":   ("flux_[NII]_6583.45",       "e_flux_[NII]_6583.45",       6583.45),
    "[NII]6548":   ("flux_[NII]_6548.05",       "e_flux_[NII]_6548.05",       6548.05),

    "[OIII]5007":  ("flux_[OIII]_5006.84",      "e_flux_[OIII]_5006.84",      5006.84),

    "[SII]6716":   ("flux_[SII]_6716.44",       "e_flux_[SII]_6716.44",       6716.44),
    "[SII]6731":   ("flux_[SII]_6730.82",       "e_flux_[SII]_6730.82",       6730.82),
}


    if lines is None:
        lines = ["Halpha"]

    # sanity check: make sure all lines exist in dict 
    for ln in lines: 
        if ln not in LINE_DEFS:
            raise KeyError(
                f"Line '{ln}' not in LINE_DEFS"
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

    # Join to fibermap for xpmm/ypmm, targettype and fibstatus
    with open(fibermap_yaml) as f:
        yml = yaml.safe_load(f)
    
    columns = [c["name"] for c in yml["schema"]]
    fibs = pl.DataFrame(yml["fibers"], schema=columns, orient="row")

    # match by fiberid
    joined = pl.DataFrame({"fiberid": fiberid}).join(fibs, on="fiberid", how="left")

    # select good science fiber == "0"
    ok = (
        (joined["targettype"].str.to_lowercase() == "science")
        & (joined["fibstatus"] == 0)
    ).to_numpy()

    flux = flux[ok]
    ivar = ivar[ok]
    ra = ra[ok]
    dec = dec[ok]

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
            fiberid=fiberid[ok],
            line_names=line_names,),
    )
