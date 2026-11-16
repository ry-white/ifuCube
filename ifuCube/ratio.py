#!/usr/bin/env python

"""
ratio.py
---------------
Builds ratio maps of emission lines
from high quality DAP emission line maps
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Sequence
from pathlib import Path
import numpy as np 
from astropy.io import fits
from astropy.wcs import WCS

def ratio_and_ivar(num, num_ivar, den, den_ivar, min_den=0.0):
    """
    R = num/den
    Var(R) = Var(num)/den^2 + num^2 * Var(den)/den^4
    """

    num = np.asarray(num, float)
    den = np.asarray(den, float)
    num_ivar = np.asarray(num_ivar, float)
    den_ivar = np.asarray(den_ivar, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        var_num = np.where(num_ivar > 0, 1.0/num_ivar, np.nan)
        var_den = np.where(den_ivar > 0, 1.0/den_ivar, np.nan)
    
        # both variances must be usable, not just the numerator's: otherwise a
        # ratio is computed but var_R comes out NaN and ivar_R comes out 0,
        # silently producing a value with zero weight.
        good = (np.isfinite(num) & np.isfinite(den)
                & np.isfinite(var_num) & np.isfinite(var_den))
        good &= (np.abs(den) > float(min_den))

        R = np.full_like(num, np.nan)
        var_R = np.full_like(num, np.nan)

        R[good] = num[good] / den[good]
        var_R[good] = var_num[good]/den[good]**2 + (num[good]**2)*var_den[good]/den[good]**4

        ivar_R = np.where(np.isfinite(var_R) & (var_R > 0), 1.0/var_R, 0.0)

    return R, ivar_R

def save_ratio_map_to_fits(
    result: Dict[str, Any],
    path: str,
    extra_header: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save emission-line ratio maps into a FITS file.

    HDU 0 : PRIMARY
    HDU 1 : RATIO   (Nratio, ny, nx)
    HDU 2 : IVAR
    HDU 3 : NEFF    (ny, nx)
    HDU 4 : RATIOS  (BinTable: NAME, NUM_WAVE, DEN_WAVE)
    """
    from pathlib import Path as _P
    path = _P(path)

    R  = np.asarray(result["ratio"], float)
    IV = np.asarray(result["ivar"], float)
    neff = np.asarray(result["neff"], float)
    wcs2d: Optional[WCS] = result.get("wcs2d", None)

    ratio_names: Optional[Sequence[str]] = result.get("ratio_names", None)
    wave_num = np.asarray(result.get("wave_num", []), float)
    wave_den = np.asarray(result.get("wave_den", []), float)

    # Promote 2D → (1, ny, nx)
    if R.ndim == 2:
        R = R[None, ...]
    if IV.ndim == 2:
        IV = IV[None, ...]

    if R.shape != IV.shape:
        raise ValueError("ratio and ivar must have the same shape")

    n_ratio, ny, nx = R.shape

    primary = fits.PrimaryHDU()
    if extra_header:
        for k, v in extra_header.items():
            primary.header[k] = v

    hdr = wcs2d.to_header(relax=True) if wcs2d is not None else fits.Header()

    hdr["BUNIT"] = ("ratio", "dimensionless: do NOT area-convert")
    ratio_hdu = fits.ImageHDU(R.astype("f4"), header=hdr, name="RATIO")
    ivar_hdu  = fits.ImageHDU(IV.astype("f4"), name="IVAR")
    neff_hdu  = fits.ImageHDU(neff.astype("f4"), name="NEFF")

    if ratio_names is None:
        ratio_names = [f"RATIO{i}" for i in range(n_ratio)]
    ratio_names = np.asarray(ratio_names, dtype="U64")

    if wave_num.size != n_ratio or wave_den.size != n_ratio:
        raise ValueError("wave_num and wave_den must have length = Nratio")

    col_name = fits.Column(name="RATIO_NAME", format="64A", array=ratio_names)
    col_num  = fits.Column(name="NUM_WAVE", format="E", unit="Angstrom", array=wave_num.astype("f4"))
    col_den  = fits.Column(name="DEN_WAVE", format="E", unit="Angstrom", array=wave_den.astype("f4"))

    ratio_table = fits.BinTableHDU.from_columns([col_name, col_num, col_den], name="RATIOS")

    fits.HDUList([primary, ratio_hdu, ivar_hdu, neff_hdu, ratio_table]).writeto(path, overwrite=True)

def build_ratio_from_map_fits(
    map_fits: str | Path,
    numerator: str,
    denominator: str,
    out_fits: str | Path,
    map_hdu: str = "FLUX",
    ivar_hdu: str = "IVAR",
    neff_hdu: str = "NEFF",
    lines_hdu: str = "LINES",
    min_den: float = 0.0,
) -> None:
    """
    Build a single ratio map from a multi-line emission map FITS produced by save_map_to_fits.

    Expects:
      - FLUX:  (Nlines, ny, nx)
      - IVAR: (Nlines, ny, nx)
      - NEFF: (ny, nx)  (optional but nice)
      - LINES table with LINE_NAME and WAVE
    """
    map_fits = Path(map_fits)
    out_fits = Path(out_fits)
    out_fits.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(map_fits, memmap=False) as hdul:
        if map_hdu not in hdul:
            raise ValueError(f"HDU '{map_hdu}' not found. Available: {[h.name for h in hdul]}")
        if ivar_hdu not in hdul:
            raise ValueError(f"HDU '{ivar_hdu}' not found. Available: {[h.name for h in hdul]}")

        M = np.asarray(hdul[map_hdu].data, float)
        IV = np.asarray(hdul[ivar_hdu].data, float)

        if M.ndim != 3:
            raise ValueError(f"{map_hdu} should be 3D. Got shape {M.shape}")
        if IV.shape != M.shape:
            raise ValueError(f"{ivar_hdu} shape {IV.shape} != {map_hdu} shape {M.shape}")

        # WCS lives on the spatial axes; attach from MAP header
        wcs2d = WCS(hdul[map_hdu].header).celestial

        neff = np.asarray(hdul[neff_hdu].data, float) if neff_hdu in hdul else np.ones(M.shape[1:], float)

        if lines_hdu not in hdul:
            raise ValueError(f"HDU '{lines_hdu}' not found; cannot look up line indices by name.")
        lines = hdul[lines_hdu].data
        names = [str(x).strip() for x in lines["LINE_NAME"]]
        waves = np.asarray(lines["WAVE"], float)

        def idx_of(name: str) -> int:
            if name in names:
                return names.index(name)
            low = [n.lower() for n in names]
            if name.lower() in low:
                return low.index(name.lower())
            raise ValueError(f"Line '{name}' not in LINES. Available: {names}")

        i_num = idx_of(numerator)
        i_den = idx_of(denominator)

        R, IVR = ratio_and_ivar(M[i_num], IV[i_num], M[i_den], IV[i_den], min_den=min_den)

        result = dict(
            ratio=R,                # (ny, nx)
            ivar=IVR,               # (ny, nx)
            neff=neff,              # (ny, nx)
            ratio_names=[f"{numerator}/{denominator}"],
            wave_num=np.array([waves[i_num]], dtype=float),
            wave_den=np.array([waves[i_den]], dtype=float),
            wcs2d=wcs2d,
        )

    save_ratio_map_to_fits(result=result, path=str(out_fits))

