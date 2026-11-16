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

The fiducial fibermap YAML is NO LONGER REQUIRED. targettype and fibstatus
are taken from the first source that has them -- the science table's own
columns, then an sframe SLITMAP, then the YAML -- because the per-exposure
flags describe the exposure being read, while the fiducial file is a nominal
snapshot whose vintage is not recorded anywhere in the output. The resolution
is printed and written to the output header as MASKSRC. See
resolve_science_mask() and list_mask_columns().

Three flags, found independently
--------------------------------
    targettype  design: science / sky / standard. No data-quality test can
                recover it -- a sky fiber is healthy and carries real flux.
    fibstatus   health: 0 = good. Per-exposure.
    quality     health, alternative spelling: a plain per-fiber 0/1 column.
                LVM DAP's PT extension has one called "mask".

Each is taken from the first source that has it, so a product carrying one
and not the others still works: PT supplies ra/dec/mask/fiberid and no
targettype, which resolves as "health from PT, design from the sframe" -- or,
if the product is already restricted to science fibers, with
assume_all_science=True. The polarity of a quality column is NOT standardised
and is never guessed silently; see MASK_GOOD_VALUE.

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

Per-fiber catalog (FIBERCAT)
----------------------------
read_dap_multi also returns meta["fibercat"]: a dict of 1D arrays, one entry
per fiber, in exactly the same order as the FiberSet rows (and therefore as
the columns of the projection matrix A). It always holds fiberid, ra, dec,
x_arcsec, y_arcsec and the flux/ivar of each requested line; `extra_cols`
adds any other DAP columns (exact names or wildcards such as "vel_*").
save_map_to_fits writes it as a FIBERCAT binary table. Use
list_dap_columns() to see what a DAP file contains.

Kinematics follow the lines you ask for
-----------------------------------------
With kinematics=True (the default), each requested line also contributes its
velocity and velocity dispersion, with errors. The column names are not
listed anywhere: they are derived from the flux column, which shares a stem
with its companions --

    flux_Halpha_6562.85   ->  stem "Halpha_6562.85"
      vel_Halpha_6562.85,  e_vel_Halpha_6562.85
     disp_Halpha_6562.85, e_disp_Halpha_6562.85

so adding a line to LINE_DEFS brings its kinematics along with no further
bookkeeping. The prefixes live in KINEMATIC_PREFIXES; if your DAP release
spells them differently (sigma_ rather than disp_, say), change that one
tuple. A derived column that is not in the file is reported and skipped
rather than raising, so a release that fits flux but not kinematics for some
line still works.

These are catalog columns only -- they are deliberately NOT projected into
map planes. A velocity is an intensity-like quantity, so the flux-conserving
projection in kernels.py is the wrong operator for it; summing velocity over
pixels is meaningless. To make a velocity map, divide the projected result by
the coverage map (see kernels.coverage_from_A) to get a weighted average
instead. Keeping kinematics per-fiber sidesteps the question entirely.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import fnmatch
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


# Per-line quantities carried into FIBERCAT alongside flux, as prefixes on the
# line's stem. Checked against the file and skipped (with a note) if absent,
# so an unexpected spelling degrades to "no kinematics" rather than a crash.
# Verify against your own release with:
#     list_dap_columns(path, "*Halpha*")
KINEMATIC_PREFIXES = ("vel", "e_vel", "disp", "e_disp")

_FLUX_PREFIX = "flux_"


def _line_stem(flux_col: str) -> str:
    """
    The part of a DAP flux column that its companions share.

        "flux_Halpha_6562.85"  ->  "Halpha_6562.85"
        "flux_[NII]_6583.45"   ->  "[NII]_6583.45"

    Raises if the column does not start with "flux_", since every other
    derived name would then be wrong too.
    """
    if not flux_col.startswith(_FLUX_PREFIX):
        raise ValueError(
            f"cannot derive companion columns from {flux_col!r}: expected it "
            f"to start with {_FLUX_PREFIX!r}"
        )
    return flux_col[len(_FLUX_PREFIX):]


def _kinematics_columns(
    dap_tbl,
    lines: Sequence[str],
    rows: np.ndarray,
    prefixes: Sequence[str] = KINEMATIC_PREFIXES,
) -> dict:
    """
    Velocity and dispersion columns for the requested lines, pulled through
    `rows` so they stay aligned with the FiberSet.

    For each line, every prefix in `prefixes` is tried against the line's
    stem. Absent columns are collected and reported once, because a release
    that fits flux but not kinematics for one line is a normal thing rather
    than an error.

    Returned keys use the *logical* line name, not the DAP column name, so
    the catalog reads consistently: flux_Halpha, ivar_Halpha, vel_Halpha,
    e_vel_Halpha, disp_Halpha, e_disp_Halpha. The DAP name is always
    recoverable from LINE_DEFS[line][0].
    """
    out = {}
    missing = []
    names = set(dap_tbl.colnames)
    for ln in lines:
        stem = _line_stem(LINE_DEFS[ln][0])
        for pre in prefixes:
            col = f"{pre}_{stem}"
            if col not in names:
                missing.append(col)
                continue
            c = dap_tbl[col]
            if hasattr(c, "filled"):
                kind = c.dtype.kind
                c = c.filled(np.nan if kind == "f" else (0 if kind in "iub" else ""))
            arr = np.asarray(c)
            if arr.ndim != 1:
                print(f"[note] FIBERCAT: skipping vector-valued column {col!r} "
                      f"(shape {arr.shape})")
                continue
            out[f"{pre}_{ln}"] = arr[rows]
    if missing:
        print(f"[note] FIBERCAT: {len(missing)} kinematic column(s) absent from "
              f"this DAP file and skipped: {missing[:6]}"
              f"{' ...' if len(missing) > 6 else ''}")
        print("       (check the real spelling with "
              "io_lvm.list_dap_columns(path, '*Halpha*') and adjust "
              "io_lvm.KINEMATIC_PREFIXES)")
    return out


# ---------------------------------------------------------------------------
# Science-fiber mask: where the flags come from
# ---------------------------------------------------------------------------
# Two independent facts decide whether a fiber belongs in a science map, and
# only one of them is visible in the data:
#
#   targettype  -- DESIGN. science / sky / standard. Never changes. A sky
#                  fiber is perfectly healthy, so no data-quality test will
#                  ever catch it; projected into a science map it contributes
#                  real flux at its own RA/Dec and nothing complains.
#   fibstatus   -- CALIBRATION. 0 = good, nonzero = dead/blocked/low
#                  throughput. Changes with time. A low-throughput fiber
#                  returns a finite, plausible, WRONG flux -- isfinite() and
#                  ivar > 0 both pass. Absence of NaN is not evidence of
#                  health.
#
# So the mask cannot be reconstructed from the fluxes, and it is not optional.
# What it can be is LOCAL: the per-exposure SLITMAP (and often the DAP table
# built from it) usually carries both columns. That source is preferred over
# the fiducial YAML for two reasons:
#
#   1. It is more correct. fibstatus is a function of time; the fiducial
#      fibermap is a nominal snapshot. A fiber that died in March is flagged
#      in March's SLITMAP but may be clean in a YAML downloaded in January --
#      or flagged in one downloaded today, retroactively nuking good December
#      data.
#   2. It is reproducible. A mask read from the file travels with the data.
#      A mask read from a path on your laptop does not, and the output
#      records nothing about which vintage was used.
#
# Column names differ between releases, so each flag is looked up through a
# list of aliases, and whichever source answers first wins. The resolution is
# always reported and written to the output header (MASKSRC/NMASKED), never
# guessed silently. If NO source supplies targettype, this raises: assuming
# every fiber is science is the one failure mode that produces a corrupt map
# which looks fine.
MASK_COLUMN_ALIASES = {
    "targettype": ("targettype", "target_type", "targtype", "fibertype",
                   "fib_type", "type"),
    "fibstatus": ("fibstatus", "fiberstatus", "fib_status", "fibermask",
                  "fibmask", "status"),
}

# A third flag some products carry instead of fibstatus: a plain per-fiber
# quality column, usually 0/1. LVM DAP's PT extension has one called "mask".
# It is accepted as a stand-in for fibstatus -- but its POLARITY is not
# standardised and getting it backwards is a silent catastrophe (you keep only
# the bad fibers), so it is never assumed. See MASK_GOOD_VALUE and
# _infer_good_value below.
QUALITY_COLUMN_ALIASES = ("mask", "quality", "qual", "qflag", "goodfiber")

# The LVM fiber complement, used to decide whether a product that carries no
# targettype has already been restricted to science fibers. One IFU slit holds
# roughly 1801 science fibers plus ~59 sky and ~24 standard/spec fibers, so
# about 1944 in total. A table with substantially more rows than
# N_SCIENCE_FIBERS must still contain sky and standard fibers; one with that
# many or fewer cannot (there are not enough science fibers for it to be a
# full-slit table, so the non-science rows are already gone).
#
# These are used only as an upper bound for that one decision, never to
# validate a row count, so being a few fibers off does not matter. Adjust if
# your instrument configuration differs.
N_SCIENCE_FIBERS = 1801
N_ALL_FIBERS = 1944


def science_only_by_count(n_rows: int) -> tuple[bool, str]:
    """
    Can a table of `n_rows` fibers still contain sky or standard fibers?

    The argument is one-directional and that is what makes it safe. If the
    table holds more rows than there are science fibers, non-science rows must
    be present -- so "assume all science" is provably wrong and is refused. If
    it holds that many or fewer, a full-slit table is ruled out, so the
    non-science fibers have already been dropped upstream.

    This cannot be fooled in the dangerous direction. The failure it rules out
    is keeping sky fibers; a table that has been cut to, say, 1754 rows has
    had something removed, and there is no way to remove rows from a full slit
    and arrive under the science count while keeping the sky fibers.

    Returns (is_science_only, reason).
    """
    n_rows = int(n_rows)
    if n_rows > N_SCIENCE_FIBERS:
        return False, (
            f"{n_rows} rows exceeds the {N_SCIENCE_FIBERS} science fibers in a "
            f"slit (total complement {N_ALL_FIBERS}), so sky and/or standard "
            f"fibers must still be present"
        )
    return True, (
        f"{n_rows} rows is at or below the {N_SCIENCE_FIBERS} science fibers "
        f"in a slit, so this table cannot be a full-slit list "
        f"({N_ALL_FIBERS}); non-science fibers were dropped upstream"
    )


def verify_science_only(fiberid, slitmap_path, verbose: bool = True) -> dict:
    """
    Settle the question against a file that actually has targettype.

    The row-count argument above is sound but indirect. This is the direct
    check: join these fiberids against a slitmap's targettype and count what
    is not science. Run it ONCE per release, then trust the count argument.

    Returns a dict with n_checked, n_not_science, and the offending types.
    """
    got = _align_flags(_flags_from_fits_slitmap(slitmap_path),
                       np.asarray(fiberid, int))
    if "targettype" not in got:
        raise KeyError(f"{Path(slitmap_path).name} has no targettype column; "
                       f"nothing to verify against")
    tt = np.char.lower(np.char.strip(np.asarray(got["targettype"], str)))
    bad = ~np.isin(tt, SCIENCE_TARGETTYPES)
    types, counts = np.unique(tt[bad], return_counts=True)
    out = dict(n_checked=len(tt), n_not_science=int(bad.sum()),
               n_unmatched=int(got.get("n_unmatched", 0)),
               types=dict(zip(types.tolist(), counts.tolist())))
    if verbose:
        if out["n_not_science"] == 0:
            print(f"[verify] all {out['n_checked']} fiberids are science "
                  f"fibers in {Path(slitmap_path).name} -- "
                  f"assume_all_science is safe for this product")
        else:
            print(f"[verify] {out['n_not_science']} of {out['n_checked']} are "
                  f"NOT science: {out['types']}. Do NOT use "
                  f"assume_all_science on this product; pass "
                  f"slitmap_path={Path(slitmap_path).name!r} instead.")
        if out["n_unmatched"]:
            print(f"[verify] {out['n_unmatched']} fiberids absent from the "
                  f"slitmap (counted as non-science above)")
    return out


# How to read a quality column: "auto" infers the polarity from the data on
# the assumption that most fibers in an exposure are fine, prints what it
# concluded, and refuses if the split is close enough to 50/50 that the
# assumption cannot carry the decision. Set to 0 or 1 to state the convention
# explicitly, which is what you should do once you have confirmed it.
MASK_GOOD_VALUE = "auto"       # "auto" | 0 | 1

# Values of targettype that count as a science fiber (lowercased, stripped).
SCIENCE_TARGETTYPES = ("science", "sci", "srd")

# fibstatus is treated as "good iff == 0". If your release uses it as a
# bitmask and you only want to reject specific bits, set this to the OR of the
# bits that disqualify a fiber; None means "any nonzero value disqualifies".
FIBSTATUS_BAD_BITS: Optional[int] = None


def _infer_good_value(arr: np.ndarray, colname: str,
                      good_value="auto", verbose: bool = True) -> np.ndarray:
    """
    Boolean "this fiber is usable" from a quality column of unknown polarity.

    A 0/1 column is read by majority vote: whichever value most fibers carry
    is taken to mean "good", because a healthy exposure has far more good
    fibers than bad. That inference is printed, and it is REFUSED when the
    split is between 35% and 65% -- at that point the majority argument does
    not hold and a guess would be a coin flip on throwing away most of the
    data.

    More than two distinct values is read as a bitmask (nonzero = bad), the
    standard FITS convention, with a warning: a bitmask may set a bit for
    something harmless at one wavelength, so the bits should be inspected
    before the column is trusted wholesale.
    """
    arr = np.asarray(arr)
    uniq = np.unique(arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr)

    if uniq.size > 2:
        warnings.warn(
            f"quality column {colname!r} has {uniq.size} distinct values "
            f"{uniq[:8]}...; reading it as a bitmask (nonzero = bad). If some "
            f"bits are harmless, select them yourself and pass a boolean.",
            RuntimeWarning,
        )
        return np.asarray(arr) == 0

    if uniq.size < 2:
        if verbose:
            print(f"[mask] quality column {colname!r} is constant "
                  f"({uniq.tolist()}) -- it rejects nothing")
        # A constant column carries no information. Treat every fiber as good
        # rather than guessing that they are all bad.
        return np.ones(arr.size, bool)

    if good_value in (0, 1):
        good = np.asarray(arr) == good_value
        if verbose:
            print(f"[mask] quality column {colname!r}: good == {good_value} "
                  f"(set explicitly), {good.sum()}/{arr.size} good")
        return good

    if good_value != "auto":
        raise ValueError(f"MASK_GOOD_VALUE must be 0, 1 or 'auto', got "
                         f"{good_value!r}")

    hi = uniq.max()
    frac_hi = float(np.mean(np.asarray(arr) == hi))
    if 0.35 <= frac_hi <= 0.65:
        raise ValueError(
            f"cannot infer the polarity of quality column {colname!r}: "
            f"{frac_hi:.0%} of fibers have the value {hi} and "
            f"{1 - frac_hi:.0%} have {uniq.min()}, which is too close to even "
            f"for the 'most fibers are good' argument.\n"
            f"Determine it yourself and then set it, e.g.\n"
            f"    io_lvm.MASK_GOOD_VALUE = 1\n"
            f"Getting this backwards keeps only the BAD fibers, so it is not "
            f"guessed."
        )
    good_val = hi if frac_hi > 0.65 else uniq.min()
    good = np.asarray(arr) == good_val
    if verbose:
        print(f"[mask] quality column {colname!r}: inferred good == "
              f"{good_val} ({good.sum()}/{arr.size} = {good.mean():.1%} good; "
              f"majority vote, set io_lvm.MASK_GOOD_VALUE to override)")
    return good


def _find_column(available: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    """
    First alias present in `available`, case-insensitively. Returns the name
    as it actually appears in the file, so the caller can index with it.
    """
    lower = {str(c).strip().lower(): str(c) for c in available}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _flags_from_table(colnames: Sequence[str], getcol) -> dict:
    """
    Pull targettype / fibstatus out of anything column-shaped.

    `getcol` is a callable name -> 1D array, so this works unchanged on an
    astropy Table, a FITS_rec, a polars DataFrame or a plain dict. Returns a
    dict with whichever of the two flags were found, plus the real column
    names used, so the provenance report can name them.
    """
    out = {}
    lookups = dict(MASK_COLUMN_ALIASES)
    lookups["quality"] = QUALITY_COLUMN_ALIASES
    for key, aliases in lookups.items():
        col = _find_column(colnames, aliases)
        if col is None:
            continue
        arr = np.asarray(getcol(col))
        if arr.ndim != 1:
            continue
        out[key] = arr
        out[f"{key}_col"] = col
    return out


# Extension names that have held the per-fiber table in one LVM product or
# another. SLITMAP is the sframe's; PT ("pointing table") is the DAP's own
# copy, which makes a DAP file self-describing -- pass slitmap_path=<the DAP
# file itself> and the flags come from the same file as the fluxes, with no
# external input at all. Names are tried in order, and if none of them is
# present any binary table carrying both a fiberid-like and a targettype-like
# column is accepted, so an unfamiliar product still resolves.
SLITMAP_EXTNAMES = ("SLITMAP", "PT", "FIBERMAP", "FIBERS", "SLITMAP1")

_FIBERID_ALIASES = ("fiberid", "fib_id", "fiberID", "fiber_id", "id")


def _flags_from_fits_slitmap(path: str | Path) -> dict:
    """
    targettype/fibstatus and fiberid from a file's per-fiber table.

    Looks for the extensions in SLITMAP_EXTNAMES, then, failing that, for any
    binary table that has both a fiber-id column and a targettype column. The
    extension actually used comes back as "extname" so the provenance report
    can name it.
    """
    path = Path(path)
    with fits.open(path) as hdul:
        present = [h.name for h in hdul]
        candidates = [e for e in SLITMAP_EXTNAMES if e in hdul]

        # fallback: sniff every table for the two columns we need
        if not candidates:
            for h in hdul:
                if getattr(h, "columns", None) is None:
                    continue
                nm = [str(c) for c in h.columns.names]
                if (_find_column(nm, _FIBERID_ALIASES)
                        and _find_column(nm, MASK_COLUMN_ALIASES["targettype"])):
                    candidates = [h.name]
                    break

        if not candidates:
            raise KeyError(
                f"{path.name} has no per-fiber table: looked for "
                f"{SLITMAP_EXTNAMES} and found extensions {present}. "
                f"Add the right name to io_lvm.SLITMAP_EXTNAMES."
            )

        extname = candidates[0]
        rec = hdul[extname].data
        names = [str(c) for c in rec.columns.names]
        got = _flags_from_table(names, lambda c: np.array(rec[c]))
        fid_col = _find_column(names, _FIBERID_ALIASES)
        if fid_col is None:
            raise KeyError(
                f"{path.name}[{extname}] has no fiberid column; "
                f"columns are {names}")
        got["fiberid"] = np.asarray(np.array(rec[fid_col]), int)
        got["extname"] = extname
    return got


def _flags_from_yaml(fibermap_yaml: str | Path) -> dict:
    """targettype/fibstatus and fiberid from the fiducial fibermap YAML."""
    with open(fibermap_yaml) as f:
        yml = yaml.safe_load(f)
    columns = [c["name"] for c in yml["schema"]]
    fibs = pl.DataFrame(yml["fibers"], schema=columns, orient="row")
    got = _flags_from_table(columns, lambda c: fibs[c].to_numpy())
    got["fiberid"] = np.asarray(fibs["fiberid"].to_numpy(), int)
    return got


def _align_flags(flags: dict, fiberid: np.ndarray) -> dict:
    """
    Reorder a flag set onto `fiberid`.

    A source that carries its own fiberid column (SLITMAP, YAML) is joined;
    one that came from the science table itself is already in row order and
    needs nothing. Unmatched fiberids get targettype "" and fibstatus -1,
    which both fail the good test, so an incomplete source drops fibers
    rather than passing them.
    """
    if "fiberid" not in flags:
        return flags                               # same table, same rows
    src_fid = np.asarray(flags["fiberid"], int)
    lut = {int(f): i for i, f in enumerate(src_fid)}
    idx = np.array([lut.get(int(f), -1) for f in fiberid], int)
    hit = idx >= 0
    out = {k: v for k, v in flags.items() if k.endswith("_col")}
    if "extname" in flags:
        out["extname"] = flags["extname"]
    out["n_unmatched"] = int((~hit).sum())
    if "targettype" in flags:
        tt = np.asarray(flags["targettype"]).astype(str)
        got = np.full(fiberid.size, "", dtype=object)
        got[hit] = tt[idx[hit]]
        out["targettype"] = got
    if "fibstatus" in flags:
        fs = np.asarray(flags["fibstatus"])
        got = np.full(fiberid.size, -1, dtype=np.int64)
        got[hit] = np.asarray(fs[idx[hit]], np.int64)
        out["fibstatus"] = got
    if "quality" in flags:
        q = np.asarray(flags["quality"])
        # Unmatched rows get a value that is NOT in the column, so they fall
        # out whichever polarity is inferred.
        fill = -1 if q.dtype.kind in "iub" else np.nan
        got = np.full(fiberid.size, fill, dtype=np.result_type(q.dtype, type(fill)))
        got[hit] = q[idx[hit]]
        out["quality"] = got
    return out


def resolve_science_mask(
    fiberid: np.ndarray,
    table_colnames: Optional[Sequence[str]] = None,
    table_getcol=None,
    slitmap_path: Optional[str | Path] = None,
    fibermap_yaml: Optional[str | Path] = None,
    mask_source: str = "auto",
    assume_all_science="auto",
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Boolean keep-mask for `fiberid`, plus a record of where it came from.

    Tries, in order: the science table's own columns, the SLITMAP of
    `slitmap_path`, then `fibermap_yaml`. The first source that supplies
    targettype wins; if that source has no fibstatus, the remaining sources
    are consulted for it alone, so a SLITMAP with targettype but an empty
    fibstatus still gets health flags from the YAML.

    Parameters
    ----------
    fiberid : (K,) int
        Fiber IDs in the row order of the data being masked.
    table_colnames, table_getcol
        Column names of the science table and a name -> array accessor. Pass
        both to let the table mask itself; this is the preferred path.
    slitmap_path : path, optional
        A FITS file with a SLITMAP extension (an sframe, usually). Use this
        when reading a DAP file whose own table lacks the flags.
    fibermap_yaml : path, optional
        The fiducial fibermap. Last resort, see the notes above.
    mask_source : {"auto", "table", "slitmap", "yaml"}
        "auto" is the priority chain. Naming one source restricts resolution
        to it and raises if it cannot answer -- use that when you need a run
        to be reproducible against a known source rather than whatever
        happens to be available.
    verbose : bool
        Print the resolution and the rejection tally.

    Returns
    -------
    ok : (K,) bool
        True for science fibers with a good fibstatus.
    info : dict
        source, targettype_col, fibstatus_col, n_total, n_keep,
        n_not_science, n_bad_status, n_unmatched, targettype, fibstatus.
        The last two are full-length (K,) arrays -- the per-fiber record of
        why each fiber was kept or dropped.
    """
    fiberid = np.asarray(fiberid, int)
    K = fiberid.size

    candidates = []
    if table_colnames is not None and table_getcol is not None:
        candidates.append(("table", lambda: _flags_from_table(table_colnames,
                                                              table_getcol)))
    if slitmap_path is not None:
        candidates.append(("slitmap", lambda: _flags_from_fits_slitmap(slitmap_path)))
    if fibermap_yaml is not None:
        candidates.append(("yaml", lambda: _flags_from_yaml(fibermap_yaml)))

    if mask_source != "auto":
        candidates = [c for c in candidates if c[0] == mask_source]
        if not candidates:
            raise ValueError(
                f"mask_source={mask_source!r} but no such source was given")

    # Walk every source and take each flag from the first source that has it.
    # The three are independent -- a product may carry the design flag without
    # the health flag or the other way round (LVM DAP's PT has mask but no
    # targettype), so stopping at the first source that answers anything would
    # throw away flags a later one could supply.
    flags, origin, tried = {}, {}, []
    for name, loader in candidates:
        try:
            got = _align_flags(loader(), fiberid)
        except (KeyError, OSError, ValueError) as exc:
            tried.append(f"{name} ({type(exc).__name__}: {exc})")
            continue
        found_here = []
        for key in ("targettype", "fibstatus", "quality"):
            if key in got and key not in flags:
                flags[key] = got[key]
                flags[f"{key}_col"] = got.get(f"{key}_col")
                origin[key] = name
                found_here.append(key)
        if found_here:
            if not flags.get("extname"):
                flags["extname"] = got.get("extname")
            if not flags.get("n_unmatched"):
                flags["n_unmatched"] = got.get("n_unmatched", 0)
        else:
            tried.append(f"{name} (none of targettype/fibstatus/quality)")
        if {"targettype"} <= set(flags) and ({"fibstatus"} <= set(flags)
                                             or {"quality"} <= set(flags)):
            break                                   # design + health, done

    if not flags:
        raise ValueError(
            "no fiber flags found in any source. Tried: "
            + ("; ".join(tried) if tried else "nothing -- pass a table, "
               "slitmap_path= or fibermap_yaml=")
            + ".\nList what a file actually has with:\n"
              "    io_lvm.list_mask_columns(path)\n"
              "and add the real spelling to io_lvm.MASK_COLUMN_ALIASES or "
              "io_lvm.QUALITY_COLUMN_ALIASES."
        )

    # ---- design flag: is this fiber pointed at the science field?
    if "targettype" in flags:
        tt = np.asarray(flags["targettype"]).astype(str)
        is_sci = np.isin(np.char.lower(np.char.strip(tt)), SCIENCE_TARGETTYPES)
    elif assume_all_science is not False and (
            assume_all_science is True
            or science_only_by_count(K)[0]):
        # An explicit decision by the caller, recorded as such. Legitimate
        # when the product has already been filtered to science fibers -- the
        # LVM DAP line-fitting output, for instance, which carries no
        # targettype because every row in it is already a science fiber.
        # VERIFY IT: compare the row count against the full fiber complement.
        # If the file still holds sky and standard fibers, this assumption
        # projects their real flux into your map at their own RA/Dec, and
        # nothing downstream will complain.
        ok_count, why = science_only_by_count(K)
        tt = np.full(K, "assumed-science", dtype=object)
        is_sci = np.ones(K, bool)
        origin["targettype"] = ("assumed-forced" if assume_all_science is True
                                and not ok_count else "assumed-by-count")
        flags["targettype_col"] = "(assumed)"
        if verbose:
            print(f"[mask] no targettype column; treating all {K} rows as "
                  f"science: {why}")
            if assume_all_science is True and not ok_count:
                print("[mask] WARNING: you forced assume_all_science=True "
                      "against that count. Sky fibers carry real flux and "
                      "will contaminate the map.")
            print("       Confirm once with "
                  "io_lvm.verify_science_only(fiberid, SFRAME).")
    else:
        _, why = science_only_by_count(K)
        raise ValueError(
            "could not determine which fibers are science targets: no "
            "targettype column in any source. Tried: "
            + "; ".join(tried)
            + f".\nThe row count does not settle it either: {why}.\n"
              "Two ways forward:\n"
              "  1. point at a file that has targettype, e.g. the sframe:\n"
              "         read_dap_multi(..., slitmap_path=SFRAME)\n"
              "  2. override the count argument, if you know better:\n"
              "         read_dap_multi(..., assume_all_science=True)\n"
              "Sky and standard fibers carry real flux, so getting this wrong "
              "gives a corrupt map that looks entirely plausible. Settle it "
              "with io_lvm.verify_science_only(fiberid, SFRAME)."
        )

    # ---- health flag: fibstatus if present, else the quality column
    fs = np.full(K, -99, np.int64)                  # -99 = "not available"
    good_status = np.ones(K, bool)
    health_col, health_src = None, None

    if "fibstatus" in flags:
        fs = np.asarray(flags["fibstatus"], np.int64)
        good_status = (fs == 0) if FIBSTATUS_BAD_BITS is None \
            else (fs & int(FIBSTATUS_BAD_BITS)) == 0
        health_col, health_src = flags.get("fibstatus_col"), origin["fibstatus"]
    elif "quality" in flags:
        q = np.asarray(flags["quality"])
        good_status = _infer_good_value(q, flags.get("quality_col", "quality"),
                                        good_value=MASK_GOOD_VALUE,
                                        verbose=verbose)
        # Store it in the fibstatus slot of the record, inverted to the
        # fibstatus convention (0 = good) so one column in FIBERMASK means the
        # same thing whichever source filled it.
        fs = np.where(good_status, 0, 1).astype(np.int64)
        health_col, health_src = flags.get("quality_col"), origin["quality"]
    else:
        warnings.warn(
            "no fibstatus or quality column found in any source; dead and "
            "low-throughput fibers will NOT be rejected. Check with "
            "io_lvm.list_mask_columns().",
            RuntimeWarning,
        )

    ok = is_sci & good_status
    info = dict(
        source=origin.get("targettype", "?"),
        extname=flags.get("extname"),
        targettype_col=flags.get("targettype_col"),
        fibstatus_col=health_col,
        fibstatus_source=health_src or "none",
        health_kind=("fibstatus" if "fibstatus" in flags
                     else "quality" if "quality" in flags else "none"),
        n_total=K,
        n_keep=int(ok.sum()),
        n_not_science=int((~is_sci).sum()),
        n_bad_status=int((is_sci & ~good_status).sum()),
        n_unmatched=int(flags.get("n_unmatched", 0) or 0),
        targettype=np.asarray(tt, str),
        fibstatus=fs,
        keep=ok,
    )

    if verbose:
        ext = f"[{info['extname']}]" if info["extname"] else ""
        print(f"[mask] targettype from {info['source']!r}{ext} "
              f"({info['targettype_col']!r}); "
              f"health from {info['fibstatus_source']!r} "
              f"({info['health_kind']}: {info['fibstatus_col']!r})")
        print(f"[mask] keeping {info['n_keep']} of {K} fibers: "
              f"{info['n_not_science']} not science, "
              f"{info['n_bad_status']} flagged bad"
              + (f", {info['n_unmatched']} unmatched" if info["n_unmatched"] else ""))
    return ok, info


def list_mask_columns(path: str | Path, pattern: str = "*") -> list:
    """
    What flag columns does this file actually have?

    Checks a DAP file's table and, if present, a SLITMAP extension, and says
    which alias each candidate matched. Run this once per release before
    trusting the auto-detection.
    """
    path = Path(path)
    found = {}
    try:
        with fits.open(path) as hdul:
            for h in hdul:
                if getattr(h, "columns", None) is None:
                    continue
                names = [str(c) for c in h.columns.names]
                hits = {k: _find_column(names, a)
                        for k, a in MASK_COLUMN_ALIASES.items()}
                hits["quality"] = _find_column(names, QUALITY_COLUMN_ALIASES)
                hits["fiberid"] = _find_column(names, _FIBERID_ALIASES)
                found[h.name or "?"] = dict(
                    n_cols=len(names),
                    matched={k: v for k, v in hits.items() if v},
                    candidates=[n for n in names if fnmatch.fnmatch(n.lower(),
                                                                    pattern.lower())],
                )
    except OSError as exc:
        print(f"[mask] cannot open {path} as FITS: {exc}")
    for hdu, d in found.items():
        print(f"  HDU {hdu:<10s} {d['n_cols']:4d} cols   matched: {d['matched']}")
    return list(found.items())


def _nan_rejected(flux: np.ndarray, ivar: np.ndarray, ok: np.ndarray,
                  label: str = "fibers") -> tuple:
    """
    Stamp rejected fibers with NaN flux and zero ivar, in place on a copy.

    Why NaN and not simply dropping the rows here: the two are complementary.
    Dropping is what keeps a bad fiber out of the projection matrix, and is
    done next. The NaN stamp is what makes a bad fiber UNMISTAKABLE if it
    ever survives -- if some later path forgets to drop, a NaN is loud (it
    propagates, and build_map counts and reports it) whereas a sky fiber's
    real flux is silent. Zeroing instead would be a positive claim that this
    fiber measured no light, which is a different and false statement.
    """
    flux = np.array(flux, float, copy=True)
    ivar = np.array(ivar, float, copy=True)
    bad = ~np.asarray(ok, bool)
    if bad.any():
        flux[bad] = np.nan
        ivar[bad] = 0.0
    return flux, ivar

def _good_science_mask(fiberid: np.ndarray, fibermap_yaml: str | Path) -> np.ndarray:
    """
    DEPRECATED: YAML-only mask, kept so old call sites keep working.

    Use resolve_science_mask(), which prefers the per-exposure flags in the
    file itself and reports where the mask came from. This wrapper is the
    yaml branch of that function with the provenance thrown away.
    """
    ok, _ = resolve_science_mask(fiberid, fibermap_yaml=fibermap_yaml,
                                 mask_source="yaml", verbose=False)
    return ok


def _good_science_mask_yaml_legacy(fiberid, fibermap_yaml) -> np.ndarray:
    """The original polars implementation, unused; kept for reference."""
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
    keep = _finite_rows(flux, ivar, ra, dec)
    n_bad = int((~keep).sum())
    if n_bad:
        print(f"[info] dropped {n_bad} of {keep.size} {label} with non-finite "
              f"flux/ivar/position")
    return flux[keep], ivar[keep], ra[keep], dec[keep], fiberid[keep]


def _finite_rows(flux, ivar, ra, dec) -> np.ndarray:
    """Rows kept by _drop_nonfinite (split out so the catalog can follow it)."""
    finite_any = np.isfinite(flux).any(axis=1) & np.isfinite(ivar).any(axis=1)
    return finite_any & np.isfinite(ra) & np.isfinite(dec)


def list_dap_columns(dap_path: str | Path, pattern: str = "*") -> list[str]:
    """
    Column names in a DAP file, optionally filtered by a shell-style pattern.

        list_dap_columns(p)              # everything
        list_dap_columns(p, "*Halpha*")  # just the Halpha quantities
    """
    cols = read_DAP_file(Path(dap_path)).colnames
    return [c for c in cols if fnmatch.fnmatch(c, pattern)]


_CORE_CATALOG_KEYS = ("fiberid", "ra", "dec", "x_arcsec", "y_arcsec")


def _catalog_columns(dap_tbl, patterns: Sequence[str], rows: np.ndarray) -> dict:
    """
    Pull every DAP column matching any of `patterns` (exact names or
    shell-style wildcards) for the table rows in `rows` -- the original row
    index of each fiber kept in the FiberSet, so catalog rows line up with
    the FiberSet exactly. Masked numeric entries become NaN (floats) or 0
    (integers); vector-valued columns are skipped with a note.
    """
    out = {}
    names = dap_tbl.colnames
    for pat in patterns:
        hits = [c for c in names if fnmatch.fnmatch(c, pat)]
        if not hits:
            raise KeyError(
                f"no DAP column matches {pat!r}; use list_dap_columns() to see "
                "what this file contains"
            )
        for c in hits:
            if c in out or c in _CORE_CATALOG_KEYS:
                continue
            col = dap_tbl[c]
            if hasattr(col, "filled"):
                kind = col.dtype.kind
                col = col.filled(np.nan if kind == "f" else (0 if kind in "iub" else ""))
            arr = np.asarray(col)
            if arr.ndim != 1:
                print(f"[note] FIBERCAT: skipping vector-valued column {c!r} "
                      f"(shape {arr.shape})")
                continue
            out[c] = arr[rows]
    return out


def read_lvm_sframe(
    sframe_path: str | Path,
    fibermap_yaml: Optional[str | Path] = None,
    drop_nonfinite: bool = True,
    mask_source: str = "auto",
    assume_all_science="auto",
    verbose: bool = True,
) -> FiberSet:
    """
    Read an LVM SFrame file and return a FiberSet.

    Parameters
    ----------
    sframe_path : path to SFrame FITS
    fibermap_yaml : path, optional
        The fiducial fibermap. No longer required: the SLITMAP in the sframe
        normally carries targettype/fibstatus itself, and those are preferred
        because they describe THIS exposure. Pass it only as a fallback, or
        when the SLITMAP's fibstatus is empty.
    drop_nonfinite : bool
        Remove fibers with non-finite flux/ivar/position before returning.
    mask_source : {"auto", "table", "yaml"}
        "table" means the SLITMAP. Name one to pin reproducibility.
    verbose : bool
        Print where the mask came from and how many fibers it rejected.

    Returns
    -------
    FiberSet
        meta["fibermask"] holds the full-length per-fiber record: fiberid,
        targettype, fibstatus and keep, for every fiber in the file including
        the rejected ones. meta["mask_info"] records the provenance.
    """

    sframe_path = Path(sframe_path)
    fibermap_yaml = Path(fibermap_yaml) if fibermap_yaml is not None else None

    with fits.open(sframe_path) as hdul:
        flux = np.asarray(hdul["FLUX"].data)
        ivar = np.asarray(hdul["IVAR"].data)
        wave = np.asarray(hdul["WAVE"].data).astype("float32")
        slitmap = hdul["SLITMAP"].data
        slit_cols = [str(c) for c in slitmap.columns.names]
        slit_get = {c: np.array(slitmap[c]) for c in slit_cols}

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

    # The SLITMAP is this exposure's own fibermap. Prefer it; fall back to
    # the YAML only if it cannot answer.
    ok, mask_info = resolve_science_mask(
        fiberid,
        table_colnames=slit_cols,
        table_getcol=slit_get.__getitem__,
        fibermap_yaml=fibermap_yaml,
        mask_source=mask_source,
        assume_all_science=assume_all_science,
        verbose=verbose,
    )

    ra_all = np.asarray(slitmap["ra"], float)
    dec_all = np.asarray(slitmap["dec"], float)

    # Full-length record of every fiber and why it was kept or dropped. This
    # is the audit trail: it survives into the FITS output as FIBERMASK.
    fibermask = dict(
        fiberid=fiberid,
        ra=ra_all,
        dec=dec_all,
        targettype=np.asarray(mask_info["targettype"], str),
        fibstatus=np.asarray(mask_info["fibstatus"], np.int64),
        keep=np.asarray(ok, bool),
    )

    # NaN the rejected fibers before dropping them, so a bad fiber is loud if
    # it ever survives a later code path. See _nan_rejected.
    flux, ivar = _nan_rejected(flux, ivar, ok)

    flux = flux[ok]
    ivar = ivar[ok]
    ra = ra_all[ok]
    dec = dec_all[ok]
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
        meta=dict(ra_deg=ra, dec_deg=dec, fiberid=fid,
                  fibermask=fibermask, mask_info=mask_info)
    )


def read_dap_multi(
    dap_path: str | Path,
    fibermap_yaml: Optional[str | Path] = None,
    lines: list[str] | None = None,
    drop_nonfinite: bool = True,
    extra_cols: Sequence[str] | None = None,
    kinematics: bool = True,
    slitmap_path: Optional[str | Path] = None,
    mask_source: str = "auto",
    assume_all_science="auto",
    verbose: bool = True,
) -> FiberSet:
    """
    Read an LVM dap file + fibermap YAML and return FiberSet
    with *one or more* emission lines.

    Parameters
    ----------
    dap_path : path to dap FITS

    fibermap_yaml : path, optional
        The fiducial fibermap. No longer required. The mask is taken from
        the first source that has targettype: the DAP table's own columns,
        then slitmap_path's SLITMAP, then this. Pass it only as a fallback.

    slitmap_path : path, optional
        A FITS file with a SLITMAP extension -- normally the sframe this DAP
        file was built from. Use it when the DAP table itself does not carry
        the flags; it is still per-exposure, so still better than the YAML.

    mask_source : {"auto", "table", "slitmap", "yaml"}
        "auto" walks the chain above. Name one source to pin a run to it.

    assume_all_science : {"auto", True, False}
        What to do when NO source supplies targettype -- the LVM DAP's PT
        extension, for one, which has only id/ra/dec/mask/fiberid/exposure.

        "auto" (default) decides from the row count, which is sound in the
        direction that matters: a table with more rows than there are science
        fibers in a slit must still hold sky and standard fibers, so the
        assumption is refused; one with fewer cannot be a full-slit table, so
        the non-science rows are already gone and the assumption holds. The
        reasoning is printed either way. True forces it regardless of the
        count (with a warning if the count disagrees); False always raises.

        Confirm it directly, once per release, with
        verify_science_only(fiberid, SFRAME) -- that joins against a file
        which does have targettype and counts what is not science.

    verbose : bool
        Print where the mask came from and how many fibers it rejected.

    lines : list of str, optional
        Logical line names to extract (keys of LINE_DEFS). Defaults to
        ["Halpha"].

    drop_nonfinite : bool
        Remove fibers with non-finite flux/ivar/position. DAP line fits fail
        on individual fibers routinely, and each bad fiber would otherwise
        NaN every pixel in its kernel support.

    extra_cols : sequence of str, optional
        Extra per-fiber DAP columns to carry into the catalog, by exact name
        or shell-style wildcard, e.g. ["vel_*", "disp_Halpha*"]. Check the
        real names with list_dap_columns() first. Use this for quantities
        that are not per-line; the kinematics of the requested lines come in
        automatically via `kinematics`.

    kinematics : bool
        Add each requested line's velocity and velocity dispersion, with
        errors, to the catalog -- derived from the line's flux column stem,
        see the module docstring. Columns absent from the file are reported
        and skipped. Default True.

    Returns
    -------
    FiberSet
        flux : (K, N_lines)
        ivar : (K, N_lines)
        wave : (N_lines,)   rest wavelengths, used as plane labels
        x_arcsec, y_arcsec : (K,)
        meta : dict with ra_deg, dec_deg, fiberid, line_names, fibercat
               (fibercat: dict of (K,) arrays, see the module docstring)
    """

    dap_path = Path(dap_path)
    fibermap_yaml = Path(fibermap_yaml) if fibermap_yaml is not None else None

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

    # The DAP table often carries targettype/fibstatus itself. Prefer that,
    # then the sframe's SLITMAP, then the fiducial YAML.
    ok, mask_info = resolve_science_mask(
        fiberid,
        table_colnames=list(dap_tbl.colnames),
        table_getcol=lambda c: np.asarray(dap_tbl[c]),
        slitmap_path=slitmap_path,
        fibermap_yaml=fibermap_yaml,
        mask_source=mask_source,
        assume_all_science=assume_all_science,
        verbose=verbose,
    )

    # Full-length record of every fiber in the file and why it was kept or
    # dropped -- the audit trail, written out as the FIBERMASK table.
    fibermask = dict(
        fiberid=fiberid,
        ra=np.array(ra, float),
        dec=np.array(dec, float),
        targettype=np.asarray(mask_info["targettype"], str),
        fibstatus=np.asarray(mask_info["fibstatus"], np.int64),
        keep=np.asarray(ok, bool),
    )

    # NaN the rejected fibers before dropping, so a bad fiber that survives
    # some later path is loud rather than silently plausible.
    flux, ivar = _nan_rejected(flux, ivar, ok)

    # original DAP-table row of every fiber we keep; the catalog is pulled
    # through this one index so it cannot drift out of step with the FiberSet
    rows = np.nonzero(ok)[0]

    flux = flux[ok]
    ivar = ivar[ok]
    ra = ra[ok]
    dec = dec[ok]
    fid = fiberid[ok]

    if drop_nonfinite:
        rows = rows[_finite_rows(flux, ivar, ra, dec)]
        flux, ivar, ra, dec, fid = _drop_nonfinite(
            flux, ivar, ra, dec, fid, label="DAP fibers")

    x_arcsec, y_arcsec = radec_to_tangent_arcsec(ra, dec)

    # per-fiber catalog, same row order as flux/ivar
    fibercat = dict(fiberid=fid, ra=ra, dec=dec, x_arcsec=x_arcsec, y_arcsec=y_arcsec)
    for i, ln in enumerate(line_names):
        fibercat[f"flux_{ln}"] = flux[:, i]
        fibercat[f"ivar_{ln}"] = ivar[:, i]
    if kinematics:
        fibercat.update(_kinematics_columns(dap_tbl, line_names, rows))
    if extra_cols:
        fibercat.update(_catalog_columns(dap_tbl, extra_cols, rows))

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
            line_names=line_names,
            fibercat=fibercat,
            fibermask=fibermask,
            mask_info=mask_info,),
    )


def read_dap(
    dap_path: str | Path,
    fibermap_yaml: Optional[str | Path] = None,
    drop_nonfinite: bool = True,
    extra_cols: Sequence[str] | None = None,
    kinematics: bool = True,
    slitmap_path: Optional[str | Path] = None,
    mask_source: str = "auto",
    assume_all_science="auto",
    verbose: bool = True,
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
        extra_cols=extra_cols,
        kinematics=kinematics,
        slitmap_path=slitmap_path,
        mask_source=mask_source,
        assume_all_science=assume_all_science,
        verbose=verbose,
    )
