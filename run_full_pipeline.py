#!/usr/bin/env python
"""
run_full_pipeline.py
---------------------
End-to-end run: cross-match LVM pointings against the WISE HII Region
catalog, download the matching DAP files, build Halpha maps, HEALPix-bin
each one, and merge into a single sparse HEALPix table.

Fill in the CONFIG section below with your actual file paths and column
names before running. Everything else reflects the settings/fixes worked
out over the course of getting this pipeline running -- see the comments
next to each one for why it's set the way it is.

Run as a script (`python run_full_pipeline.py`) or paste section by
section into IPython -- either works. If pasting into IPython, do the
TREE_DIR block FIRST, before importing sdss_access/Access anywhere in
that session (it only takes effect if set before the first Access/
HttpAccess object is created).
"""

from __future__ import annotations
import os
import re

# ============================================================
# TREE_DIR -- must happen before any sdss_access import below.
# Skip this block entirely if you've already set TREE_DIR via your conda
# activate hook / shell profile (recommended for a persistent fix).
# ============================================================
os.environ.setdefault("TREE_DIR", os.path.expanduser("~/tree_local"))


# ============================================================
# CONFIG -- edit these for your setup
# ============================================================

# --- Input catalogs (metadata tables, not spectra -- small, fast to load) ---
LVM_POINTINGS_PATH = "drpall-1.2.1.fits"     # your drpall-style pointing table
WISE_CATALOG_PATH = "wise_HII.fits"      # downloaded from VizieR / astro.phys.wvu.edu/wise

# --- Column names -- CHECK THESE against your actual files; they vary by
# export/reader (see conversation history: units, naming, etc. all bit us
# at some point) ---
POINTING_RA_COL, POINTING_DEC_COL = "sci_ra", "sci_dec"
CATALOG_RA_COL, CATALOG_DEC_COL = "_RA_icrs", "_DE_icrs"
CATALOG_NAME_COL = "Name"
CATALOG_RADIUS_COL, CATALOG_RADIUS_UNIT = "Rad", "arcsec"   # confirmed arcsec, not arcmin

# --- sdss_access / DAP file identification ---
RELEASE = "sdsswork"
DOWNLOAD_METHOD = "http"          # "rsync" (fast, needs dtn.sdss.org reachable) or "http" (fallback)
DAPVER = "1.2.1"
RSPID = "rsp108"
SNLEVEL = "sn20"
DAPTYPE = "dap"

# --- Map-building parameters (kernels.py / geometry.py) ---
# Two independent constraints, see the kernels.py module docstring:
#   1. flux conservation needs r_cap = R_SIGMA_MULT*FWHM/2.3548 to exceed the
#      pixel size (here 3*35/2.3548 = 44.6" > 15", comfortably safe). If it
#      does not, fibers whose disc holds no pixel centre are dropped and their
#      flux is LOST -- build_projection_A now raises rather than doing that
#      silently.
#   2. spatial sampling needs FWHM comparable to the ~37" fiber spacing, or
#      each fiber paints an isolated dot with empty gaps between them.
# Map values are flux PER PIXEL and so scale with pixel area; the HEALPix
# binning below divides by the pixel solid angle (to_surface_brightness=True,
# the default) so the merged table does not depend on this choice.
PIXEL_SIZE_ARCSEC = 15.0
FWHM_ARCSEC = 35
FIBER_RADIUS_ARCSEC = 18.5
R_SIGMA_MULT = 3.0
KERNEL_MODE = "gauss"
FIBERMAP_YAML = "lvm_fiducial_fibermap.yaml"
LINES = ["Halpha"]                 # single line -> HEALPix column will be "VALUE", not "VALUE0"
EXTRA_COLS = None                  # extra DAP columns for each map's FIBERCAT table, e.g. ["vel_*"];
                                   # check real names with ifuCube.io_lvm.list_dap_columns(path)

# --- HEALPix ---
#NSIDE = 131072                     # n = 131072 matches PIXEL_SIZE_ARCSEC=2.0 (see suggest_nside)
NSIDE = 4096                        # n = 4096 combine with psa=15" and fwhm of 35 (about size of fiber)

# --- Batch size / safety ---
N_POINTINGS_TO_USE = 1000            # slice of matched pointings to actually process
MAPCAT_DIR = "/mnt/mako-big/rwhite/mapcats_nside4096"              # all map_*.fits / hp_*.fits land here, not the cwd
HP_DIR = "healpix"                 # directory for HEALPix files
DRY_RUN = False                     # keep True until the printed download plan looks right


# ============================================================
# Imports (after TREE_DIR is set)
# ============================================================
from astropy.table import Table, unique

from ifuCube.healpix_build import (
    find_matching_pointings,
    download_lvm_files,
    wcs_fits_to_healpix_table,
    merge_healpix_tables,
    crossmatch_catalog_to_healpix,
    plot_matched_regions,
)
from ifuCube.io_lvm import read_dap_multi
from ifuCube.mapcat_build import build_map, save_map_to_fits


# ============================================================
# 1. Load metadata tables and cross-match (no spectra downloaded yet)
# ============================================================
print("=== 1. Loading tables and cross-matching ===")
lvm_pointings = Table.read(LVM_POINTINGS_PATH)
wise_table = Table.read(WISE_CATALOG_PATH)

matched, match_info = find_matching_pointings(
    lvm_pointings, wise_table,
    pointing_ra_col=POINTING_RA_COL, pointing_dec_col=POINTING_DEC_COL,
    catalog_ra_col=CATALOG_RA_COL, catalog_dec_col=CATALOG_DEC_COL,
    catalog_radius_col=CATALOG_RADIUS_COL, catalog_radius_unit=CATALOG_RADIUS_UNIT,
)

# de-duplicate repeat exposures of the same tile -- see conversation: a
# pointing can have several rows (different mjd/expnum) sharing one tileid
matched_unique = unique(matched, keys="tileid")
print(f"{len(matched)} matched rows -> {len(matched_unique)} unique tileids")

batch = matched_unique[:N_POINTINGS_TO_USE]


# ============================================================
# 2. Download DAP files for the batch
# ============================================================
print("\n=== 2. Downloading DAP files ===")
dap_dests = download_lvm_files(
    batch,
    column_map={"mjd": "mjd", "drpver": "drpver", "tileid": "tileid", "expnum": "expnum"},
    fixed_kwargs={"dapver": DAPVER, "rspid": RSPID, "snlevel": SNLEVEL, "daptype": DAPTYPE},
    filetype="lvm_dap",
    release=RELEASE,
    method=DOWNLOAD_METHOD,
    skip_missing=True,   # not every exposure has DAP output at this snlevel -- drop cleanly, don't crash
    dry_run=DRY_RUN,
)

if DRY_RUN:
    print("\nDRY_RUN=True -- stopping here. Check the printed plan above, "
          "then set DRY_RUN=False at the top of this script and re-run.")
    raise SystemExit(0)


# ============================================================
# 3. Build Halpha maps + HEALPix-bin each one (skip/report failures,
#    don't let one bad file kill the whole batch)
# ============================================================
print("\n=== 3. Building maps and HEALPix tables ===")
os.makedirs(MAPCAT_DIR, exist_ok=True)
os.makedirs(HP_DIR, exist_ok=True)

def _expnum_from_dap_path(path: str) -> str:
    """Pull the exposure number back out of a DAP filename, e.g.
    '.../dap-rsp108-sn20-00003469.dap.fits.gz' -> '00003469'. We do this
    instead of trusting the loop index because skip_existing_local /
    skip_missing can drop rows from dap_dests, which would silently break
    a simple index-into-batch lookup once any row upstream gets dropped."""
    m = re.search(r"-(\d{8})\.", os.path.basename(path))
    if not m:
        raise ValueError(f"couldn't find an 8-digit exposure number in {path}")
    return m.group(1)

hp_files = []
failed = []

for dap_path in dap_dests:
    if not os.path.exists(dap_path):
        print(f"[skip] missing on disk: {dap_path}")
        failed.append(dap_path)
        continue
    try:
        expnum_str = _expnum_from_dap_path(dap_path)

        fibers = read_dap_multi(dap_path, fibermap_yaml=FIBERMAP_YAML, lines=LINES,
                                extra_cols=EXTRA_COLS)
        result = build_map(
            fibers, pixel_size_arcsec=PIXEL_SIZE_ARCSEC, fwhm_arcsec=FWHM_ARCSEC,
            fiber_radius_arcsec=FIBER_RADIUS_ARCSEC, r_sigma_mult=R_SIGMA_MULT,
            kernel_mode=KERNEL_MODE, wcs_strategy="fit",
        )
        map_path = os.path.join(MAPCAT_DIR, f"map_exp{expnum_str}.fits")
        save_map_to_fits(result, map_path)

        hp_path = os.path.join(HP_DIR, f"hp_exp{expnum_str}.fits")
        # weights_hdu is not optional in practice: without it, pixels that
        # received no flux hold a finite 0.0 and get averaged in as real
        # zeros, dragging the HEALPix values down. With it they carry zero
        # weight and are correctly dropped.
        # to_surface_brightness=True (default) divides by the map pixel area
        # so VALUE is flux/arcsec**2, independent of PIXEL_SIZE_ARCSEC.
        wcs_fits_to_healpix_table(map_path, hp_path,
                                  nside=NSIDE, hdu="FLUX", weights_hdu="IVAR",
                                  coord_system_out="galactic",
                                  to_surface_brightness=True)
        hp_files.append(hp_path)
    except Exception as e:
        # keep the traceback: an unanticipated failure is much easier to fix
        # with a line number than with a bare message
        import traceback
        print(f"[skip] {dap_path}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed.append(dap_path)

print(f"\n{len(hp_files)} succeeded, {len(failed)} failed/skipped")


# ============================================================
# 4. Merge all HEALPix tables into one
# ============================================================
print("\n=== 4. Merging HEALPix tables ===")
if hp_files:
    merge_healpix_tables(hp_files, "merged_halpha.fits")
else:
    print("Nothing to merge -- all files failed/skipped, check the errors above.")
    raise SystemExit(1)


# ============================================================
# 5. Cross-match the merged footprint against the WISE catalog,
#    and plot the covered regions
# ============================================================
print("\n=== 5. Cross-matching and plotting ===")
crossmatches = crossmatch_catalog_to_healpix(
    "merged_halpha.fits", wise_table,
    ra_col=CATALOG_RA_COL, dec_col=CATALOG_DEC_COL, name_col=CATALOG_NAME_COL,
    radius_col=CATALOG_RADIUS_COL, radius_unit=CATALOG_RADIUS_UNIT,
    value_col="VALUE", weight_col="WEIGHT",
    min_matched_pixels=10,   # filter out near-degenerate edge-clip matches
)
crossmatches.write("hii_crossmatches.fits", overwrite=True)
print(f"Wrote {len(crossmatches)} cross-matched source(s) -> hii_crossmatches.fits")

import matplotlib.pyplot as plt
fig = plot_matched_regions(
    "merged_halpha.fits", wise_table,
    ra_col=CATALOG_RA_COL, dec_col=CATALOG_DEC_COL, name_col=CATALOG_NAME_COL,
    radius_col=CATALOG_RADIUS_COL, radius_unit=CATALOG_RADIUS_UNIT,
    value_col="VALUE", min_matched_pixels=10,
    ncols=5, max_plots=20,
)
if fig is not None:
    fig.savefig("matched_regions.png", dpi=150, bbox_inches="tight")
    print("Saved matched_regions.png")

print("\n=== Done ===")