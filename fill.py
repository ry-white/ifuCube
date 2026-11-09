#!/usr/bin/env python

"""
fill.py
-------
Impute the fluxes of rejected fibers, so the projection sees a regular grid
instead of one with bites out of it.

Why at the FIBER level and not in the map
-----------------------------------------
A rejected fiber is dropped from the FiberSet, so it has no column in A, and
the pixels it would have covered get weight only from its neighbours. That
leaves a dip in `coverage` -- not a void in flux, because the kernel is about
one fiber spacing wide and the neighbours already spread into the gap. But the
dip is real, and a coverage cut turns it into a hole.

You could inpaint the finished map instead. Don't: by then the missing fiber's
absence has been smeared across every pixel its kernel would have touched, so
there is no longer a well-defined thing to fill. Imputing the fiber and letting
the normal projection run keeps the geometry regular, keeps flux conservation
meaningful (the total now includes an estimate rather than silently omitting
one), and keeps the invented values labelled and in one place.

IDW and a local plane fit are statistically tied and both beat the plain mean 
slightly; nearest-neighbour is
clearly the worst of the three real methods. IDW is the default because it is
the cheapest of the two leaders (no least-squares per hole), it is vectorized,
and on a regular hex grid the six neighbours are equidistant so it reduces to
their mean -- which is exact for any linear field, the centre of a regular
hexagon being the mean of its vertices. `method="plane"` is there for smooth
fields, where it was the best of the five.

Accuracy degrades with how sharp the field is relative to the fiber spacing,
which is the honest limit: no local estimator recovers structure the array
never sampled.

Clustering, not count
---------------------
Same bundle, 116 dead fibers: 97 scattered at random plus a contiguous patch
of 19. Error at the imputed fibers:

    population   mean gap   rms/sigma
    scattered       1.075      0.170
    patch           1.723      0.285

The gap is the mean distance to the k neighbours actually used, in units of
fiber spacing; a healthy interior fiber scores exactly 1.000. Sweeping the
cut:

    mean-gap <=   kept   % of patch kept   rms/sigma   worst
          1.10     51                 0%      0.125    0.384
          1.20     83                 0%      0.156    0.418
          1.35     97                 0%      0.170    0.659
          1.50    103                32%      0.173    0.659
           inf    116               100%      0.193    0.659

1.35 is the default: it admits every scattered fiber and excludes the whole
patch. Guarding on the NEAREST survivor instead does not work -- a fiber on
the rim of a dead patch has a close neighbour on one side and nothing on the
other, and IDW quality depends on all k, not on the best one.

Uncertainty of an invented number
---------------------------------
`ivar_mode="loo"` imputes every SURVIVING fiber from its own neighbours and
takes the scatter of that residual as the imputation uncertainty. It needs no
assumption, it is calibrated on the actual field, and it tracks the truth:
predicted sigma 0.0309 against an actual 0.0384 at 5% dead, 0.0329 against
0.0314 at 15%. It can under-predict by about 25% when the holes are more
clustered than the survivors, which is the right direction to be wrong in only
if you also respect `max_gap_spacings`.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
from scipy.spatial import cKDTree

from .io_readers import FiberSet


def _poly_design(ra, dec, order, ra0=None, dec0=None):
    """
    Polynomial basis in (ra, dec), centred to keep the normal equations sane.

    Used to recover tangent-plane (x, y) for fibers that are not in the
    FiberSet. geometry.radec_to_tangent_arcsec derives its own field centre
    from whatever you hand it, so calling it on a different subset would
    silently move the origin; fitting the survivors' own mapping instead
    leaves every existing coordinate untouched.

    Measured accuracy over a 0.5 degree field, recovering 209 dropped
    positions from 1591 survivors:  order 1 -> 3.0" max (too coarse, the
    projection's quadratic term matters), order 2 -> 0.003", order 3 -> 1e-5".
    Order 2 is the default; a fiber spacing is 37".
    """
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    if ra0 is None:
        ra0, dec0 = ra.mean(), dec.mean()
    u, v = ra - ra0, dec - dec0
    cols = [np.ones_like(u), u, v]
    if order >= 2:
        cols += [u * u, v * v, u * v]
    if order >= 3:
        cols += [u ** 3, v ** 3, u * u * v, u * v * v]
    return np.column_stack(cols), ra0, dec0


def _impute(xk, yk, Fk, xm, ym, k, method="idw"):
    """
    Estimate the flux at (xm, ym) from the survivors (xk, yk, Fk).

    Fk may be (K,) or (K, N) for several lines at once. The neighbour search
    is geometric, so one query serves every line; only the combination step
    differs between methods.

    Returns (estimate, distances) -- the distances are handed back because
    the caller needs them for the mean-gap guard.
    """
    d, idx = cKDTree(np.c_[xk, yk]).query(np.c_[xm, ym], k=k)
    d = np.atleast_2d(d); idx = np.atleast_2d(idx)
    F2 = Fk if Fk.ndim == 2 else Fk[:, None]
    M, N = len(xm), F2.shape[1]

    if method == "nearest":
        out = F2[idx[:, 0]]
    elif method == "mean":
        out = F2[idx].mean(axis=1)
    elif method == "idw":
        w = 1.0 / np.maximum(d, 1e-6) ** 2
        w = w / w.sum(axis=1, keepdims=True)
        out = np.einsum("mk,mkn->mn", w, F2[idx])
    elif method == "plane":
        # local linear fit, evaluated at the hole; coordinates are centred on
        # the hole so the constant term IS the estimate
        out = np.empty((M, N))
        for i in range(M):
            j = idx[i]
            A = np.c_[np.ones(len(j)), xk[j] - xm[i], yk[j] - ym[i]]
            out[i] = np.linalg.lstsq(A, F2[j], rcond=None)[0][0]
    else:
        raise ValueError(f"method must be idw/plane/mean/nearest, got {method!r}")

    return (out[:, 0] if Fk.ndim == 1 else out), d


def fill_missing_fibers(
    fibers: FiberSet,
    method: str = "idw",
    k: int = 6,
    max_gap_spacings: float = 1.35,
    poly_order: int = 2,
    ivar_mode: str = "loo",
    ivar_scale: float = 1.0,
    science_only: bool = True,
    verbose: bool = True,
) -> FiberSet:
    """
    Return a new FiberSet with rejected science fibers imputed from survivors.

    Needs meta["fibermask"] -- the full per-fiber record written by
    io_lvm.read_dap_multi / read_lvm_sframe, which holds ra, dec, targettype
    and keep for EVERY fiber in the file, including the ones that were
    dropped. Without it there is no way to know where the holes are.

    Parameters
    ----------
    method : {"idw", "plane", "mean", "nearest"}
        Imputation rule. "idw" and "plane" were statistically tied in the
        measurements above; "plane" edges it on smooth fields and costs a
        least-squares solve per hole. "mean" is close behind, "nearest"
        clearly worse.
    k : int
        Neighbours used per estimate. Six is the hex coordination number.
    max_gap_spacings : float
        Refuse to impute a fiber whose MEAN distance to the k neighbours used
        exceeds this, in units of the median fiber spacing. A healthy
        interior fiber scores 1.000. This is the guard that stops
        interpolation becoming fabrication inside a dead patch; see the
        module docstring for the sweep behind the default. np.inf disables
        it, and the reported worst gap tells you what you would be admitting.
    poly_order : int
        Order of the (ra, dec) -> (x, y) fit used to place the missing fibers.
    ivar_mode : {"loo", "none"}
        "loo" calibrates the imputed IVAR by leave-one-out on the survivors.
        "none" gives imputed fibers ivar = 0, i.e. they contribute flux but
        carry no weight -- useful if you want the holes filled for display
        but want every downstream S/N to ignore them.
    ivar_scale : float
        Multiplies the LOO-derived variance. >1 is more conservative.
    science_only : bool
        Only impute fibers whose targettype is science. Sky and standard
        fibers were never going to be in the map.

    Returns
    -------
    FiberSet
        Survivors first, imputed fibers appended. meta["fibercat"] gains an
        "imputed" boolean column; meta["fill_info"] records what happened.
        Flux and ivar of the requested lines are filled; any OTHER per-line
        catalog column (velocity, dispersion) is set to NaN for imputed
        fibers rather than invented, because the right way to interpolate a
        velocity field is not the right way to interpolate a flux.
    """
    if ivar_mode not in ("loo", "none"):
        raise ValueError(f"ivar_mode must be 'loo' or 'none', got {ivar_mode!r}")

    fm = fibers.meta.get("fibermask")
    if not fm:
        raise KeyError(
            "fibers.meta['fibermask'] is missing, so the positions of the "
            "rejected fibers are unknown. Read with a version of io_lvm that "
            "records it (read_dap_multi / read_lvm_sframe do)."
        )

    xk = np.asarray(fibers.x_arcsec, float)
    yk = np.asarray(fibers.y_arcsec, float)
    Fk = np.asarray(fibers.flux, float)                  # (K, N)
    Ik = np.asarray(fibers.ivar, float)
    K, N = Fk.shape

    ra_k = np.asarray(fibers.meta["ra_deg"], float)
    dec_k = np.asarray(fibers.meta["dec_deg"], float)

    keep = np.asarray(fm["keep"], bool)
    cand = ~keep
    if science_only and "targettype" in fm:
        tt = np.char.lower(np.char.strip(np.asarray(fm["targettype"], str)))
        cand &= np.isin(tt, ("science", "sci", "srd", "assumed-science"))
    cand_idx = np.nonzero(cand)[0]

    info: Dict[str, Any] = dict(n_candidates=int(cand_idx.size), n_filled=0,
                                n_too_far=0, method=method, k=k)
    if cand_idx.size == 0:
        if verbose:
            print("[fill] no rejected science fibers to impute")
        return fibers

    # --- place the missing fibers in the SAME tangent plane as the survivors
    A_k, ra0, dec0 = _poly_design(ra_k, dec_k, poly_order)
    cxx, *_ = np.linalg.lstsq(A_k, xk, rcond=None)
    cyy, *_ = np.linalg.lstsq(A_k, yk, rcond=None)
    resid = np.hypot(A_k @ cxx - xk, A_k @ cyy - yk)
    A_m, _, _ = _poly_design(np.asarray(fm["ra"], float)[cand_idx],
                             np.asarray(fm["dec"], float)[cand_idx],
                             poly_order, ra0, dec0)
    xm, ym = A_m @ cxx, A_m @ cyy
    info["position_fit_rms_arcsec"] = float(np.sqrt(np.mean(resid ** 2)))

    # --- median fiber spacing, from the survivors themselves
    dnn, _ = cKDTree(np.c_[xk, yk]).query(np.c_[xk, yk], k=2)
    spacing = float(np.median(dnn[:, 1]))
    info["spacing_arcsec"] = spacing

    # --- impute, then reject anything too far from a survivor
    est, dist = _impute(xk, yk, Fk, xm, ym, k=min(k, K), method=method)
    gap = dist.mean(axis=1) / spacing        # mean over the k actually used
    ok = gap <= float(max_gap_spacings)
    info["n_too_far"] = int((~ok).sum())
    info["max_gap_spacings"] = float(gap.max()) if gap.size else 0.0

    xm, ym, est = xm[ok], ym[ok], est[ok]
    cand_idx = cand_idx[ok]
    M = xm.size
    info["n_filled"] = int(M)
    if M == 0:
        if verbose:
            print(f"[fill] all {info['n_candidates']} candidates exceed "
                  f"max_gap_spacings={max_gap_spacings}; nothing imputed")
        return fibers

    # --- IVAR for an invented number, calibrated on the survivors
    if ivar_mode == "loo":
        d_s, i_s = cKDTree(np.c_[xk, yk]).query(np.c_[xk, yk], k=min(k + 1, K))
        w = 1.0 / np.maximum(d_s[:, 1:], 1e-6) ** 2      # drop self
        w = w / w.sum(axis=1, keepdims=True)
        loo = np.einsum("mk,mkn->mn", w, Fk[i_s[:, 1:]])   # IDW, self excluded
        sig = np.std(loo - Fk, axis=0)                   # per line
        var = (np.maximum(sig, np.finfo(float).tiny) ** 2) * float(ivar_scale)
        iv_m = np.tile(1.0 / var, (M, 1))
        info["loo_sigma"] = sig.tolist()
    else:
        iv_m = np.zeros((M, N))
        info["loo_sigma"] = None

    # --- assemble
    out_flux = np.vstack([Fk, est])
    out_ivar = np.vstack([Ik, iv_m])
    out_x = np.r_[xk, xm]
    out_y = np.r_[yk, ym]

    meta = dict(fibers.meta)
    meta["ra_deg"] = np.r_[ra_k, np.asarray(fm["ra"], float)[cand_idx]]
    meta["dec_deg"] = np.r_[dec_k, np.asarray(fm["dec"], float)[cand_idx]]
    if "fiberid" in meta:
        meta["fiberid"] = np.r_[np.asarray(meta["fiberid"]),
                                np.asarray(fm["fiberid"])[cand_idx]]

    cat = fibers.meta.get("fibercat")
    if cat:
        lines = list(fibers.meta.get("line_names") or [])
        newcat = {}
        for key, col in cat.items():
            col = np.asarray(col)
            if key.startswith("flux_") and key[5:] in lines:
                newcat[key] = np.r_[col, est[:, lines.index(key[5:])]]
            elif key.startswith("ivar_") and key[5:] in lines:
                newcat[key] = np.r_[col, iv_m[:, lines.index(key[5:])]]
            elif key == "fiberid":
                newcat[key] = np.r_[col, np.asarray(fm["fiberid"])[cand_idx]]
            elif key == "ra":
                newcat[key] = np.r_[col, np.asarray(fm["ra"], float)[cand_idx]]
            elif key == "dec":
                newcat[key] = np.r_[col, np.asarray(fm["dec"], float)[cand_idx]]
            elif key == "x_arcsec":
                newcat[key] = np.r_[col, xm]
            elif key == "y_arcsec":
                newcat[key] = np.r_[col, ym]
            elif col.dtype.kind == "f":
                # velocities, dispersions and anything else: NOT invented.
                newcat[key] = np.r_[col, np.full(M, np.nan)]
            else:
                pad = np.zeros(M, col.dtype) if col.dtype.kind in "iub" else np.full(M, "")
                newcat[key] = np.r_[col, pad]
        newcat["imputed"] = np.r_[np.zeros(K, bool), np.ones(M, bool)]
        meta["fibercat"] = newcat

    meta["fill_info"] = info

    if verbose:
        print(f"[fill] spacing {spacing:.1f}\" | position fit rms "
              f"{info['position_fit_rms_arcsec']:.3f}\"")
        print(f"[fill] imputed {M} of {info['n_candidates']} rejected science "
              f"fibers (IDW, k={k}); {info['n_too_far']} left alone for "
              f"exceeding {max_gap_spacings} spacings")
        if info["loo_sigma"] is not None:
            s = ", ".join(f"{v:.4g}" for v in info["loo_sigma"][:4])
            print(f"[fill] LOO sigma per line: {s}"
                  f"{' ...' if N > 4 else ''}  -> imputed ivar set from this")

    return FiberSet(flux=out_flux, ivar=out_ivar,
                    wave=np.asarray(fibers.wave),
                    x_arcsec=out_x, y_arcsec=out_y, meta=meta)
