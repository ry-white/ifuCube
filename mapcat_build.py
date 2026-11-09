#!/usr/bin/env python

"""
mapcat_build.py
---------------
Builds high quality DAP emission line maps
And catalogs wanted DAP quantities

Wires together:
    - io_reader.FiberSet
    - geometry.make_grid_from_points
    - geometry.fit_celestial_wcs
    - kernels.build_projection_A / splat_cube / ivar_through_A / neff_from_A

Units of the output -- read this before comparing map values
-------------------------------------------------------------
The FLUX image is stored as *flux per pixel* (BUNIT = 'flux/pixel').

Why: the projection is flux-conserving. Each fiber's measured flux is shared
out over nearby pixels with weights that sum to 1, so summing the pixels of
a region gives the total flux in that region, exactly -- the natural unit
for photometry. The price is that a pixel value depends on how big the
pixel is: build the same data with 5" instead of 15" pixels and each value
drops 9x (the light is split over 9x as many pixels), while totals stay the
same.

To compare maps built with different pixel sizes, or to average pixels
(which the HEALPix step does), divide by the pixel solid angle first. That
gives a surface brightness, flux per arcsec**2, which does not depend on the
pixel size:

    surface_brightness = FLUX / PIXAR_A2          [flux / arcsec**2]
    ivar_sb            = IVAR * PIXAR_A2**2       (variance scales with the square)

PIXAR_A2 is written in the FLUX header. read_map() below does this for you,
and wcs_fits_to_healpix_table does it by default.

A second, constant factor: fiber fill factor
--------------------------------------------
DAP fluxes are totals per fiber aperture (35.3" diameter), but LVM fibers
sit on a 37" hexagonal grid, so apertures cover only

    APERFILL = pi * (35.3/2)**2 / ((sqrt(3)/2) * 37**2) = 0.8255

of the sky; light falling between fibers is never measured. The map spreads
each fiber's flux over its whole grid cell, so on smooth emission the
surface brightness above reads ~0.83x the true sky value. That is the same
for every pixel size and kernel, so it does not affect comparisons between
LVM maps, but divide by APERFILL (read_map(..., correct_fill_factor=True))
when comparing against other surveys or a model in absolute units.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Sequence

import numpy as np
from pathlib import Path as _Path
from astropy.io import fits
from astropy.wcs import WCS

from .io_readers import FiberSet
from .geometry import pixel_area_arcsec2
from .kernels import splat_cube, ivar_through_A, neff_from_A, coverage_from_A
from .projection import prepare_grid, coverage_plateau

# The LVM instrument facts now live in constants.py so that cube_build and
# mapcat_build cannot disagree about them. Re-exported here because callers
# and notebooks import LVM_FILL_FACTOR from this module.
from .constants import (LVM_FIBER_DIAMETER_ARCSEC,      # noqa: F401
                        LVM_FIBER_PITCH_ARCSEC,         # noqa: F401
                        LVM_FILL_FACTOR)                # noqa: F401

def build_map(
    fibers: FiberSet,
    pixel_size_arcsec: float,
    fwhm_arcsec: float,
    fiber_radius_arcsec: float,
    r_sigma_mult: float = 3.0,
    kernel_mode: str = "gauss",
    wcs_strategy: str = "fit", # "fit" or "none"
    max_dropped_frac: float = 0.02,
) -> Dict[str, Any]:
    """
    Build flux-conserving emission-line maps from a FiberSet

    Steps
    -----
    1. Build a grid + inside_IFU mask
    2. Build the projection matrix A (pixels x fibers)
    3. Forward project fiber line fluxes -> maps
    4. Propagate IVAR through A
    5. Compute Neff and coverage maps.
    6. Optionally fit a celestial WCS.

    Parameters
    ----------
    fibers : FiberSet
        Input fiber data
    pixel_size_arcsec : float
        spaxel size
    fwhm_arcsec : float
        Kernel FWHM used for splatting. Set this from the fiber spacing
        (~37 arcsec for LVM), not from the pixel size -- see kernels.py.
    fiber_radius_arcsec : float
        Fiber radius; sets the grid padding and hull expansion.
    r_sigma_mult : float
        Sigma multiplier for truncation radius
    kernel_mode : {"gauss", "tophat"}
        Kernel type
    wcs_strategy : {"fit", "none"}
        If "fit", use fibers.meta["ra_deg"], ["dec_deg"] to fit a celestial WCS
    max_dropped_frac : float
        Forwarded to build_projection_A: refuse to build if more than this
        fraction of fibers would have their flux discarded.

    Returns
    -------
    results : dict with keys
        "map" : (N_lines, ny, nx) emission maps, or (ny, nx) for a single line
        "ivar" : same shape, inverse variance map
        "neff" : (ny, nx) effective number of fibers per pixel
        "coverage" : (ny, nx) summed kernel weight per pixel (row sums of A);
                     divide "map" by this to get a weighted average instead
                     of a flux-conserving sum. NOT a fraction: it sums to K
                     over the whole grid, so it sits near K/N_inside (~0.19
                     for 15" LVM pixels), and its maximum moves with the
                     kernel width. Do not threshold on it directly.
        "coverage_norm" : (ny, nx) coverage / cov_plateau -- 1.0 on the
                     well-sampled interior, 0 outside, for any kernel width
                     and pixel size. THIS is the one to threshold on.
        "cov_plateau" : float, K / N_inside; the exact mean of coverage over
                     the footprint, used as the normalisation above
        "A"    : (P, K) projection matrix
        "xg", "yg", "inside" : grid and IFU mask
        "wcs2d" : WCS or None
        "wave" : wavelength array
        "line_names" : list[str] if present in fibers.meta
        "pixel_size_arcsec" : echoed back, so save_map_to_fits can record it
        "fibercat" : per-fiber catalog from the reader (dict of (K,) arrays)
                     or None; written by save_map_to_fits as FIBERCAT
    """

    # Shared setup: scrub, grid, A, WCS2D -- identical to what build_cube
    # does, which is why it lives in projection.py rather than being written
    # out twice. float64 here because L is a handful of emission lines, not
    # ~12,000 spectral planes, so there is nothing to save by going narrower.
    g = prepare_grid(
        fibers,
        pixel_size_arcsec=pixel_size_arcsec,
        fwhm_arcsec=fwhm_arcsec,
        fiber_radius_arcsec=fiber_radius_arcsec,
        r_sigma_mult=r_sigma_mult,
        kernel_mode=kernel_mode,
        wcs_strategy=wcs_strategy,
        max_dropped_frac=max_dropped_frac,
        dtype=float,
    )
    flux, ivar, wave = g["flux"], g["ivar"], g["wave"]
    xg, yg, inside = g["xg"], g["yg"], g["inside"]
    ny, nx, A = g["ny"], g["nx"], g["A"]

    # Forward project, get a cube shape (L, ny, nx)
    map_cube = splat_cube(flux, A, ny=ny, nx=nx)

    # propagate uncertainties through A
    ivar_cube = ivar_through_A(ivar, A, ny=ny, nx=nx)

    # effective number of fibers per pixel, and raw coverage
    #
    # NOTE ON THE SCALE OF `coverage`. It is the row sum of A, and the columns
    # of A sum to 1, so
    #
    #     sum_p coverage(p) = sum_p sum_k A[p,k] = sum_k 1 = K
    #
    # exactly, whatever the kernel width or pixel size. That makes the mean of
    # coverage over the covered footprint exactly K / N_inside -- typically
    # ~0.19 for 15" pixels on the LVM grid, NOT ~1. A threshold like
    # `coverage > 0.3` therefore masks everything, and the number at which it
    # would start to bite moves with both FWHM and pixel size. It is not a
    # fraction of anything and has no fixed ceiling: a narrow kernel piles
    # weight into few pixels and pushes the maximum well above the mean (0.51
    # at FWHM 20", 0.22 at FWHM 60", measured).
    #
    # `coverage_norm` below divides by that exact mean, so it is 1.0 on the
    # well-sampled interior and falls to 0 at the edge, for ANY kernel width
    # and pixel size. Threshold on that, or on `neff`, never on raw coverage.
    neff = neff_from_A(A, ny=ny, nx=nx)
    coverage = coverage_from_A(A, ny=ny, nx=nx)

    # Exact plateau value: sum_p coverage(p) == K, so the mean over the
    # footprint is K / N_inside. Derived from the data, not from the fibre
    # aperture or the grid spacing, so it stays correct for any bundle.
    # Derivation in projection.coverage_plateau.
    cov_plateau = coverage_plateau(flux.shape[0], inside)
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage_norm = (coverage / cov_plateau if np.isfinite(cov_plateau)
                         else np.zeros_like(coverage))

    # L=1 for a single emission line
    if map_cube.shape[0] == 1:
        emap = map_cube[0]          # (ny, nx)
        eivar = ivar_cube[0]    # (ny, nx)
    else:
        # if multiple lines are given
        emap = map_cube
        eivar = ivar_cube

    # 2D WCS, already fitted by prepare_grid (which passes the grid origin
    # x0/y0 -- omitting those offsets the WCS from the array by about a
    # third of a degree; see geometry.py).
    wcs2d: Optional[WCS] = g["wcs2d"]

    line_names = fibers.meta.get("line_names", None)

    return dict(
        map=emap,
        ivar=eivar,
        neff=neff,
        coverage=coverage,
        coverage_norm=coverage_norm,
        cov_plateau=cov_plateau,
        A=A,
        xg=xg,
        yg=yg,
        inside=inside,
        wave=wave,
        wcs2d=wcs2d,
        line_names=line_names,
        pixel_size_arcsec=float(pixel_size_arcsec),
        flux_in=flux,
        fibercat=fibers.meta.get("fibercat"),
        fibermask=fibers.meta.get("fibermask"),
        mask_info=fibers.meta.get("mask_info"),
        fill_info=fibers.meta.get("fill_info"),
    )

_KIN_ALIASES = {
    "vel":  ("vel", "velocity", "v", "vgas", "vel_gas"),
    "disp": ("disp", "sigma", "sig", "vdisp", "disp_gas"),
}


def _find_kin_column(cat, quantity: str, line: str, err_prefix: str = "e_"):
    """
    Locate the value and error columns for one quantity of one line.

    Why this is a search and not an f-string: DAP column spellings move
    between releases and between the maps/cube products ('vel_Halpha',
    'Halpha_vel', 'VEL_HA'), and an f-string that guesses wrong raises a
    KeyError three functions deep. This tries every ordering of every alias
    case-insensitively, then reports what it DID find so the message is
    actionable.

    Returns (value_column_name, error_column_name_or_None).
    """
    names = list(cat.keys())
    low = {n.lower(): n for n in names}
    stems = _KIN_ALIASES.get(quantity, (quantity,))
    ln = str(line)

    val = None
    for st in stems:
        for cand in (f"{st}_{ln}", f"{ln}_{st}", f"{st}{ln}", f"{ln}{st}"):
            if cand.lower() in low:
                val = low[cand.lower()]
                break
        if val:
            break
    if val is None:
        near = [n for n in names if ln.lower() in n.lower()]
        raise KeyError(
            f"no {quantity!r} column for line {line!r}. Tried stems "
            f"{stems} in both orders. Columns mentioning {line!r}: "
            f"{near or 'none'}")

    # The error column: same spelling, with the error marker bolted on in
    # whichever of the two usual places.
    err = None
    for cand in (f"{err_prefix}{val}", f"{val}_err", f"{val}_error",
                 f"{val}_e", f"e{val}"):
        if cand.lower() in low:
            err = low[cand.lower()]
            break
    return val, err


def build_kinematic_map(
    result: Dict[str, Any],
    quantity: str = "vel",
    lines: Optional[Sequence[str]] = None,
    weighting: str = "ivar",
    min_weight_frac: float = 0.05,
    err_prefix: str = "e_",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build a kinematic map by reusing an existing build_map() result.

    A DIFFERENT OPERATOR FROM build_map
    ===================================
    build_map projects FLUX, which is an extensive quantity: it splits each
    fiber's flux among pixels with column-normalised weights, so summing
    pixels recovers the total. Velocity is intensive -- it does not add up.
    Splatting velocity the same way would give a pixel the SUM of its
    neighbours' velocities, so a region at a uniform -27 km/s with three
    contributing fibers would read -81 km/s.

    The right operator is a weighted AVERAGE: with w_k the per-fiber weight
    and A the same projection matrix,

        W_p      = sum_k A[p,k] * w_k                       (the normaliser)
        v_p      = (1/W_p) * sum_k A[p,k] * w_k * v_k
        Var(v_p) = (1/W_p**2) * sum_k (A[p,k] * w_k)**2 * Var(v_k)

    Note the weights enter the variance SQUARED. This reuses result["A"], so
    the kinematic map lands on exactly the same grid and WCS as the flux map
    and the two can be divided, masked or stacked without reprojection.

    MASKED FIBERS ARE ZEROED IN THE WEIGHTS, NOT IN THE VALUES
    ==========================================================
    build_map NaN-scrubs bad fluxes to 0.0, which is harmless there because a
    zero contributes zero flux. Here it would be a disaster: 0 km/s is a
    perfectly valid velocity, so a dead fiber scrubbed to zero drags its
    whole kernel toward zero. Instead a fiber with a non-finite value, a
    non-finite error, or a zero/negative error gets w_k = 0, which removes it
    from both the numerator and the normaliser -- the pixel is then the
    average of its surviving neighbours, which is the correct answer.

    Parameters
    ----------
    result : dict
        Output of build_map(). Must still carry "A", "fibercat", "inside".
    quantity : {"vel", "disp"} or any column stem present in the catalog
        Aliases are resolved case-insensitively and in either word order.
    lines : sequence of str, optional
        Which lines to build. Default: every line in result["line_names"]
        that has a column for this quantity.
    weighting : {"ivar", "equal"}
        "ivar": w_k = 1/sigma_k**2 from the error column -- the minimum
        variance choice, and the one that automatically down-weights the
        faint-line fits that dominate DAP kinematic scatter. "equal": w_k = 1
        for every usable fiber, which is what you want if you suspect the
        reported errors are wrong.
    min_weight_frac : float
        Pixels whose normaliser W_p falls below this fraction of the
        footprint median are set NaN. These are the edge pixels fed by a
        single far-off fiber, where the "average" is one noisy number
        stretched over many pixels. 0.05 is loose; raise it to 0.2 for a
        figure.
    err_prefix : str
        Prefix marking the error column ("e_vel_Halpha").

    Returns
    -------
    dict with
        "map"        : (n_lines, ny, nx) weighted-average maps, NaN where
                       unsampled. Units are the catalog's units -- km/s for
                       vel; for DAP `disp_*` this is ANGSTROM, not km/s.
        "ivar"       : matching inverse variance, 0 where map is NaN
        "err"        : 1/sqrt(ivar), for convenience
        "weight_sum" : (n_lines, ny, nx) the normaliser W_p
        "neff_kin"   : (n_lines, ny, nx) Kish effective fiber count,
                       (sum w)**2 / sum w**2, using the SAME weights as the
                       average -- so this is the honest independent-sample
                       count for these maps, not result["neff"]
        "lines"      : list of line names actually built
        "quantity", "weighting" : echoed
        plus "xg", "yg", "inside", "wcs2d", "pixel_size_arcsec" copied from
        `result`, so this dict can go straight to save_map_to_fits().
    """
    A = result.get("A")
    if A is None:
        raise ValueError("result has no 'A'; pass the dict build_map returned")

    # A is sparse (build_projection_A returns csr_matrix) and must STAY
    # sparse: at 2" pixels a dense (P, K) would be hundreds of GB, and
    # np.asarray() on it silently produces a 0-d object array that only
    # fails later with an unreadable message. Everything below is written
    # against a tiny shim so the same arithmetic runs either way.
    import scipy.sparse as _sp
    _issparse = _sp.issparse(A)
    if not _issparse:
        A = np.asarray(A, float)

    cat = result.get("fibercat")
    if not cat:
        raise ValueError(
            "result has no 'fibercat'. Kinematics live in the per-fiber "
            "catalog, so read the DAP with a reader that builds one "
            "(read_dap_multi(..., ) populates meta['fibercat']).")

    inside = np.asarray(result["inside"], bool)
    ny, nx = inside.shape
    P, K = A.shape
    if P != ny * nx:
        raise ValueError(f"A has {P} rows but the grid has {ny*nx} pixels; "
                         "this A was built for a different grid")

    # Which lines to do. Default to every line that actually HAS a column,
    # silently skipping the rest -- a DAP file commonly fits kinematics for
    # the bright lines only, and failing the whole call because [OI]6300 has
    # no velocity would be unhelpful.
    if lines is None:
        avail = result.get("line_names") or []
        lines = []
        for ln in avail:
            try:
                _find_kin_column(cat, quantity, ln, err_prefix)
                lines.append(ln)
            except KeyError:
                pass
        if not lines:
            raise KeyError(
                f"no line in {list(avail)} has a {quantity!r} column. "
                f"Catalog columns: {sorted(cat.keys())[:30]}")
    lines = [str(l) for l in lines]

    out = np.full((len(lines), ny, nx), np.nan)
    oiv = np.zeros((len(lines), ny, nx))
    wsum = np.zeros((len(lines), ny, nx))
    nkin = np.zeros((len(lines), ny, nx))

    flat_inside = inside.ravel()

    for i, ln in enumerate(lines):
        vcol, ecol = _find_kin_column(cat, quantity, ln, err_prefix)
        v = np.asarray(cat[vcol], float).ravel()
        if v.size != K:
            raise ValueError(
                f"column {vcol!r} has {v.size} rows but A has {K} fibers. "
                "The catalog must be the fibers that built this map, in the "
                "same order -- did the map come from a different read?")

        if ecol is not None:
            e = np.asarray(cat[ecol], float).ravel()
        else:
            e = np.full(K, np.nan)

        # Usable fibers. Note the error test is part of the usability test:
        # a velocity with no error is not a measurement we can weight.
        good = np.isfinite(v)
        if weighting == "ivar":
            good &= np.isfinite(e) & (e > 0)
            w = np.where(good, 1.0 / np.maximum(e, 1e-30) ** 2, 0.0)
        elif weighting == "equal":
            w = good.astype(float)
        else:
            raise ValueError(f"weighting must be 'ivar' or 'equal', got {weighting!r}")

        # Zero the WEIGHT, and zero the value only where its weight is
        # already zero -- so a NaN can never enter the matrix product.
        vz = np.where(good, v, 0.0)
        var = np.where(good & np.isfinite(e) & (e > 0), e ** 2, 0.0)

        # Aw[p,k] = A[p,k] * w_k, and Aw2 is that squared ELEMENTWISE (not a
        # matrix square). For sparse A, .multiply() is elementwise and keeps
        # the sparsity pattern; `A * w` would be a matrix product.
        if _issparse:
            Aw = A.multiply(w[None, :]).tocsr()
            Aw2 = Aw.multiply(Aw)
            W = np.asarray(Aw.sum(axis=1)).ravel()
            s2 = np.asarray(Aw2.sum(axis=1)).ravel()
        else:
            Aw = A * w[None, :]
            Aw2 = Aw ** 2
            W = Aw.sum(axis=1)
            s2 = Aw2.sum(axis=1)

        num = Aw @ vz                       # (P,)

        with np.errstate(divide="ignore", invalid="ignore"):
            vmap = num / W
            # Var = (1/W^2) * sum_k (A[p,k] w_k)^2 Var(v_k)
            vvar = (Aw2 @ var) / W ** 2
            # Kish N_eff on the SAME weights that formed the average
            neff_k = np.where(s2 > 0, W ** 2 / s2, 0.0)

        # Drop pixels whose normaliser is a small fraction of the typical
        # interior value: one distant fiber smeared over many pixels.
        Win = W[flat_inside]
        med = np.nanmedian(Win[Win > 0]) if np.any(Win > 0) else 0.0
        bad = (~np.isfinite(vmap)) | (W <= min_weight_frac * med) | (~flat_inside)

        vmap = np.where(bad, np.nan, vmap)
        ivm = np.where(bad | ~np.isfinite(vvar) | (vvar <= 0), 0.0, 1.0 / vvar)

        out[i] = vmap.reshape(ny, nx)
        oiv[i] = ivm.reshape(ny, nx)
        wsum[i] = W.reshape(ny, nx)
        nkin[i] = np.where(bad, 0.0, neff_k).reshape(ny, nx)

        if verbose:
            fin = np.isfinite(out[i]) & inside
            print(f"[kin] {ln:10s} {quantity}: col={vcol!r} err={ecol!r} "
                  f"{int(good.sum())}/{K} fibers usable, "
                  f"{int(fin.sum())}/{int(inside.sum())} pixels filled, "
                  f"median={np.nanmedian(out[i][fin]):+.2f} "
                  f"median err={np.nanmedian(1/np.sqrt(oiv[i][fin])):.2f} "
                  f"median Neff={np.nanmedian(nkin[i][fin]):.2f}")

    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(oiv > 0, 1.0 / np.sqrt(oiv), np.nan)

    return dict(
        map=out, ivar=oiv, err=err, weight_sum=wsum, neff_kin=nkin,
        lines=lines, quantity=quantity, weighting=weighting,
        # Copied through so this dict is accepted by save_map_to_fits().
        neff=result.get("neff"), coverage=result.get("coverage"),
        coverage_norm=result.get("coverage_norm"),
        cov_plateau=result.get("cov_plateau"),
        xg=result.get("xg"), yg=result.get("yg"), inside=inside,
        wcs2d=result.get("wcs2d"), line_names=lines,
        wave=np.asarray(result.get("wave", np.arange(len(lines))), float)[:len(lines)]
             if np.size(result.get("wave", [])) >= len(lines)
             else np.arange(len(lines), dtype=float),
        pixel_size_arcsec=result.get("pixel_size_arcsec"),
        fibercat=None, fibermask=None, mask_info=result.get("mask_info"),
        fill_info=result.get("fill_info"),
    )


def save_map_to_fits(
    result: Dict[str, Any],
    path: str,
    extra_header: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> None:
    """
    Save an emission-line map result (from build_map) into a FITS file.

    Layout
    ======
    HDU 0 : PRIMARY
    HDU 1 : FLUX   (Nlines, ny, nx) or (ny, nx)
    HDU 2 : IVAR  (same shape)
    HDU 3 : NEFF  (ny, nx)
    HDU 4 : LINES (BinTable: LINE_NAME, WAVE)
    HDU 5 : COVERAGE (ny, nx)   -- summed kernel weight, if present
    HDU 6 : FIBERMASK (BinTable) -- EVERY fiber in the input, with
            targettype, fibstatus and keep; the audit trail for the mask
    HDU 7 : FIBERCAT (BinTable) -- one row per fiber USED, if present

    WCS (if present) is written to the FLUX, IVAR, NEFF and COVERAGE headers
    (they all share the same grid) and applies to the last two axes; the first
    axis of FLUX/IVAR is "line index".

    BUNIT / PIXAR_A2 are written so that downstream code knows these values
    are flux *per pixel* and can convert to surface brightness; the FLUX
    header also carries COMMENT cards spelling out the conversion. See the
    module docstring, and read_map() for a reader that applies it.
    """

    from pathlib import Path as _P
    path = _P(path)

    M = np.asarray(result["map"], float)
    IV = np.asarray(result["ivar"], float)
    neff = np.asarray(result["neff"], float)
    wave = np.asarray(result["wave"], float)
    wcs2d: Optional[WCS] = result.get("wcs2d", None)
    line_names: Optional[Sequence[str]] = result.get("line_names", None)
    coverage = result.get("coverage", None)

    # ensure MAP and IVAR have identical shape
    if M.shape != IV.shape:
        raise ValueError("map and ivar must have the same shape")

    if M.ndim == 2:
        # promote to (1, ny, nx) for single line case
        M = M[None, ...]
        IV = IV[None, ...]

    n_lines, ny, nx = M.shape

    # Primary HDU
    primary_hdu = fits.PrimaryHDU()
    if extra_header:
        for k, v in extra_header.items():
            primary_hdu.header[k] = v

    # attach spatial WCS to the image headers
    if wcs2d is not None:
        map_hdr = wcs2d.to_header(relax=True)
        pix_area = pixel_area_arcsec2(wcs2d)
    else:
        map_hdr = fits.Header()
        ps = result.get("pixel_size_arcsec", None)
        pix_area = float(ps) ** 2 if ps else None

    # Record the units convention. These two keywords are what let the
    # HEALPix binning (or any other averaging step) convert to surface
    # brightness instead of silently inheriting this pixel size.
    map_hdr["BUNIT"] = ("flux/pixel", "flux-conserving: amount per output pixel")
    if pix_area is not None:
        map_hdr["PIXAR_A2"] = (pix_area, "pixel solid angle [arcsec**2]")
    map_hdr["FLUXCONS"] = (True, "columns of projection matrix A sum to 1")
    map_hdr["APERFILL"] = (round(float(LVM_FILL_FACTOR), 4),
                           "LVM fiber aperture fill factor, see COMMENT")
    for line in (
        "FLUX is flux PER PIXEL (flux-conserving: summing pixels gives the",
        "total flux). Values scale with pixel area, so to compare maps with",
        "different pixel sizes, or to average pixels, convert first:",
        "  surface brightness [flux/arcsec**2] = FLUX / PIXAR_A2",
        "  its inverse variance                = IVAR * PIXAR_A2**2",
        "Fibers cover APERFILL of the sky (35.3in apertures, 37in hex grid);",
        "divide surface brightness by APERFILL for true sky brightness.",
        "ifuCube.mapcat_build.read_map() applies both conversions.",
    ):
        map_hdr.add_comment(line)

    ivar_hdr = map_hdr.copy()
    ivar_hdr["BUNIT"] = ("1/(flux/pixel)**2", "inverse variance of FLUX")
    aux_hdr = wcs2d.to_header(relax=True) if wcs2d is not None else fits.Header()

    map_hdu = fits.ImageHDU(M.astype("f4"), header=map_hdr, name="FLUX")
    ivar_hdu = fits.ImageHDU(IV.astype("f4"), header=ivar_hdr, name="IVAR")
    neff_hdu = fits.ImageHDU(neff.astype("f4"), header=aux_hdr, name="NEFF")

    # Line metadata table
    if line_names is None:
        line_names = [f"LINE{i}" for i in range(n_lines)]

    line_names = np.asarray(line_names, dtype="U32")
    if wave.size != n_lines:
        # sanity
        raise ValueError(f"wave.size ({wave.size}) != n_lines ({n_lines})")

    col_name = fits.Column(name="LINE_NAME", format="32A", array=line_names)
    col_wave = fits.Column(name="WAVE", format="E", unit="Angstrom", array=wave.astype("f4"))
    lines_hdu = fits.BinTableHDU.from_columns([col_name, col_wave], name="LINES")

    hdus = [primary_hdu, map_hdu, ivar_hdu, neff_hdu, lines_hdu]
    if coverage is not None:
        hdus.append(fits.ImageHDU(np.asarray(coverage, float).astype("f4"),
                                  header=aux_hdr, name="COVERAGE"))

    # Per-fiber catalog: one row per fiber that went into this map, in the
    # same order as the columns of A. Missing numeric values are stored as NaN.
    # Mask provenance: which source said which fibers were usable. Recorded
    # in the header so a map can be traced back to its mask without the
    # external file that may have supplied it.
    mi = result.get("mask_info") or {}
    if mi:
        mh = map_hdu.header          # the live header; map_hdr was copied
        mh["MASKSRC"] = (str(mi.get("source", "unknown"))[:60],
                         "source of targettype/fibstatus")
        mh["MASKFSRC"] = (str(mi.get("fibstatus_source", "unknown"))[:60],
                          "source of fibstatus specifically")
        mh["MASKEXT"] = (str(mi.get("extname") or "")[:60],
                         "FITS extension the mask came from")
        mh["MASKTCOL"] = (str(mi.get("targettype_col", ""))[:60],
                          "targettype column used")
        mh["MASKSCOL"] = (str(mi.get("fibstatus_col") or "")[:60],
                          "health flag column used")
        mh["MASKKIND"] = (str(mi.get("health_kind", ""))[:60],
                          "fibstatus | quality | none")
        mh["NFIBTOT"] = (int(mi.get("n_total", 0)), "fibers in input file")
        mh["NFIBUSED"] = (int(mi.get("n_keep", 0)), "fibers used in FLUX")
        mh["NNOTSCI"] = (int(mi.get("n_not_science", 0)), "rejected: not science")
        mh["NBADSTAT"] = (int(mi.get("n_bad_status", 0)), "rejected: fibstatus")

    cov_plateau = result.get("cov_plateau")
    if cov_plateau is not None and np.isfinite(cov_plateau):
        map_hdu.header["COVPLATE"] = (float(cov_plateau),
                                      "mean coverage over footprint = K/N_inside")
        map_hdu.header["COVNOTE"] = ("COVERAGE/COVPLATE ~ 1 on the interior",
                                     "threshold on that, not on COVERAGE")

    # Fill provenance, written whether or not any fiber was imputed --
    # FILLED=F is information,
    fi = result.get("fill_info") or {}
    mh = map_hdu.header
    mh["FILLED"] = (bool(fi.get("n_filled", 0)), "T if dead fibers were imputed")
    if fi:
        mh["FILLMETH"] = (str(fi.get("method", ""))[:20], "imputation method")
        mh["FILLK"]    = (int(fi.get("k", 0)), "neighbors per estimate")
        mh["FILLN"]    = (int(fi.get("n_filled", 0)), "fibers imputed")
        mh["FILLSKIP"] = (int(fi.get("n_too_far", 0)), "refused: gap too large")
        mh["FILLGAP"]  = (float(fi.get("max_gap_spacings", 0.0)),
                          "worst mean-gap seen [fiber spacings]")
        mh["FILLSPAC"] = (float(fi.get("spacing_arcsec", 0.0)),
                          "median fiber spacing [arcsec]")
        mh["COMMENT"]  = ("Some fibers were imputed; The FIBERCAT column "
                          " 'imputed' flags which fibers were.")
        
    fibermask = result.get("fibermask")
    if fibermask:
        from astropy.table import Table as _T
        mask_hdu = fits.table_to_hdu(_T(fibermask))
        mask_hdu.name = "FIBERMASK"
        mask_hdu.header["COMMENT"] = ("Every fiber in the input file, kept or "
                                      "not. keep=F fibers are absent from "
                                      "FLUX and FIBERCAT.")
        hdus.append(mask_hdu)

    fibercat = result.get("fibercat")
    if fibercat:
        from astropy.table import Table
        cat_hdu = fits.table_to_hdu(Table(fibercat))
        cat_hdu.name = "FIBERCAT"
        cat_hdu.header["COMMENT"] = ("One row per fiber used to build FLUX; "
                                     "flux_* are per-fiber aperture totals.")
        hdus.append(cat_hdu)

    # Create the parent directory rather than failing after all the work is
    # done. ratio.py has always done this; the inconsistency meant a typo in a
    # notebook config cell surfaced as a FileNotFoundError from inside astropy,
    # several hundred lines from the line that caused it.
    _parent = _Path(path).parent
    if str(_parent) not in ("", "."):
        _parent.mkdir(parents=True, exist_ok=True)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)


def _plane_index(name: str, names) -> int:
    """
    Plane number of a line, by name, case-insensitively.

    Exists so nobody has to write names.index(...) and get a bare ValueError
    when the spelling is off; this says what IS in the file.
    """
    want = str(name).strip()
    if want in names:
        return names.index(want)
    low = [n.lower() for n in names]
    if want.lower() in low:
        return low.index(want.lower())
    raise KeyError(f"line {name!r} is not in this map. Available: {names}")


def read_map(
    path: str,
    units: str = "surface_brightness",
    correct_fill_factor: bool = False,
    hdu: str = "FLUX",
) -> Dict[str, Any]:
    """
    Read a map written by save_map_to_fits, converting units on the way in.

    Parameters
    ----------
    path : str
        Map FITS file.
    units : {"surface_brightness", "flux_per_pixel"}
        "surface_brightness" (default): FLUX / PIXAR_A2, in flux/arcsec**2.
        Independent of the pixel size the map was built with, so maps built
        with different settings can be compared pixel by pixel.
        "flux_per_pixel": the stored values; sum pixels for total flux.
    correct_fill_factor : bool
        Also divide by APERFILL (~0.83), so smooth emission matches the true
        sky surface brightness. Only valid with units="surface_brightness".
    hdu : str
        Image HDU to read (default "FLUX").

    Returns
    -------
    dict with data, ivar, wcs, bunit, header, fibercat, line_names, wave and
    index. `index` is a callable -- m["index"]("Halpha") gives the plane
    number -- so nothing downstream has to index FLUX by position.

    Returns
    -------
    dict with
        "data"     : (N_lines, ny, nx) or (ny, nx) array in the chosen units
        "ivar"     : matching inverse variance
        "wcs"      : celestial WCS
        "bunit"    : units string of "data"
        "header"   : header of the image HDU
        "fibercat" : astropy Table of the FIBERCAT extension, or None

    Example
    -------
        m = read_map("map_exp00006116.fits")
        plt.subplot(projection=m["wcs"]); plt.imshow(m["data"][0], origin="lower")
    """
    from astropy.table import Table

    with fits.open(path) as hdul:
        hdr = hdul[hdu].header.copy()
        data = np.asarray(hdul[hdu].data, float)
        ivar = np.asarray(hdul["IVAR"].data, float) if "IVAR" in hdul else None
        cat = Table.read(hdul["FIBERCAT"]) if "FIBERCAT" in hdul else None
    wcs2d = WCS(hdr).celestial
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
        ivar = ivar[0] if ivar is not None else None

    bunit = str(hdr.get("BUNIT", "flux/pixel"))
    if units == "surface_brightness":
        if bunit != "flux/pixel":
            raise ValueError(f"{hdu} has BUNIT={bunit!r}, not 'flux/pixel'; "
                             "use units='flux_per_pixel' to read it as stored")
        area = float(hdr.get("PIXAR_A2", pixel_area_arcsec2(wcs2d)))
        data = data / area
        ivar = ivar * area ** 2 if ivar is not None else None
        bunit = "flux/arcsec**2"
        if correct_fill_factor:
            fill = float(hdr.get("APERFILL", LVM_FILL_FACTOR))
            data = data / fill
            ivar = ivar * fill ** 2 if ivar is not None else None
            bunit += " (aperture-fill corrected)"
    elif units == "flux_per_pixel":
        if correct_fill_factor:
            raise ValueError("correct_fill_factor needs units='surface_brightness'")
    else:
        raise ValueError(f"units must be 'surface_brightness' or 'flux_per_pixel', got {units!r}")

    # Which plane is which. Without this the caller holds an (N, ny, nx) cube
    # and no way to index it except by position -- and reordering --lines once
    # then hands them an Halpha map labelled Hbeta, which looks entirely
    # plausible. Empty list if the file has no LINES table.
    line_names, waves = [], []
    try:
        with fits.open(path, memmap=False) as _h:
            if "LINES" in _h:
                _l = _h["LINES"].data
                line_names = [str(x).strip() for x in _l["LINE_NAME"]]
                waves = np.asarray(_l["WAVE"], float)
    except (OSError, KeyError):
        pass

    return dict(data=data, ivar=ivar, wcs=wcs2d, bunit=bunit, header=hdr,
                fibercat=cat, line_names=line_names,
                wave=np.asarray(waves, float),
                index=lambda name, _n=line_names: _plane_index(name, _n))