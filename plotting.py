#!/usr/bin/env python

"""
plotting.py
-----------
Quick-look figures for maps, cubes and fibre catalogs.

WHY THIS IS ITS OWN MODULE. matplotlib is an optional dependency of this
package -- healpix_build imports it inside the one function that needs it,
precisely so that building a map on a headless machine does not require a
plotting stack. Putting `import matplotlib.pyplot` at the top of
mapcat_build.py or cube_build.py would make matplotlib mandatory for
`python -m ifuCube.mapcat_build`, which is the wrong trade for a server job
that writes FITS and never draws anything. Keeping the figures here means
only code that wants a picture pays for one.

The same argument answers "one copy or two": maps and cubes want the same
2D panel, so it lives here once rather than in both builders.

ORIENTATION. Every image here is drawn with origin="lower", without
exception. Array row 0 is the SOUTH edge of the field, and matplotlib's
default origin="upper" puts row 0 at the top -- i.e. upside down, with Dec
increasing downward. astropy's WCSAxes quietly forces origin="lower" for
you, so a function that omits it is correct when handed a WCS and flipped
when not, which is worse than being consistently wrong: the same call gives
two different pictures depending on an argument that has nothing to do with
orientation.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


__all__ = ["show_map", "show_planes", "show_fibers", "show_wave",
           "good_pixels"]


# ---------------------------------------------------------------------------
# masking helper
# ---------------------------------------------------------------------------

def good_pixels(result: Dict[str, Any], min_coverage: float = 0.5,
                min_neff: Optional[float] = None) -> np.ndarray:
    """
    The mask you actually want: inside the footprint AND well sampled.

    Pass `min_coverage` as a fraction of the coverage PLATEAU, not as a raw
    coverage value. Raw `coverage` is the row sum of the projection matrix;
    because the columns of A sum to 1, sum_p coverage(p) == K exactly, so its
    mean over the footprint is K/N_inside -- about 0.19 for 15" pixels, not
    ~1. A cut like `coverage > 0.3` therefore masks the entire map, and the
    value at which it starts to bite moves with both the kernel width and the
    pixel size.

    This function divides by the plateau for you, so min_coverage=0.5 means
    "half the typical interior sampling" at any pixel size and any FWHM.

    Parameters
    ----------
    result : dict
        From build_map / build_cube (uses "coverage_norm" or "coverage" +
        "cov_plateau", and "inside"), or from read_map / read_cube (falls
        back to COVERAGE / COVPLATE in the header).
    min_coverage : float
        Threshold on coverage_norm. 0.5 is a reasonable default; 0.8 for a
        publication figure; 0.0 keeps the whole footprint.
    min_neff : float, optional
        Additionally require this many effective fibres per pixel. Useful
        when you care about independence rather than sampling -- a pixel with
        neff ~ 1 carries one fibre's information however well covered it is.

    Returns
    -------
    (ny, nx) bool array, True where the pixel is usable.
    """
    cn = result.get("coverage_norm")
    if cn is None:
        cov = result.get("coverage")
        if cov is None:
            raise KeyError(
                "result has neither 'coverage_norm' nor 'coverage'. For a "
                "file read back from disk, pass the header's COVERAGE HDU "
                "and COVPLATE, or re-read with read_map/read_cube.")
        plateau = result.get("cov_plateau")
        if plateau is None:
            hdr = result.get("header") or {}
            plateau = hdr.get("COVPLATE")
        if plateau is None or not np.isfinite(plateau) or plateau == 0:
            raise KeyError(
                "no cov_plateau / COVPLATE available, so raw COVERAGE cannot "
                "be normalised. Thresholding raw COVERAGE is what masks the "
                "whole map -- see the docstring.")
        with np.errstate(divide="ignore", invalid="ignore"):
            cn = np.asarray(cov, float) / float(plateau)

    m = np.isfinite(cn) & (cn > min_coverage)
    inside = result.get("inside")
    if inside is not None:
        m &= np.asarray(inside, bool)
    if min_neff is not None:
        neff = result.get("neff")
        if neff is None:
            raise KeyError("min_neff given but result has no 'neff'")
        m &= np.asarray(neff, float) >= min_neff
    return m


# ---------------------------------------------------------------------------
# one 2D plane
# ---------------------------------------------------------------------------

def show_map(data, wcs=None, ax=None, label="", title=None, pct=(1, 99.5),
             log=False, mask=None, cmap="inferno", grid=True, colorbar=True,
             east_left=True, **kw):
    """
    One 2D plane with an optional WCS grid, percentile stretch and colorbar.

    Parameters
    ----------
    data : (ny, nx) array
        The plane. A 3D array raises rather than silently plotting a slice.
    wcs : astropy WCS, optional
        Celestial WCS. When given, the axes get RA/Dec ticks.
    ax : matplotlib Axes, optional
        Draw here instead of making a figure. Must already have the right
        projection if you want WCS ticks.
    pct : (lo, hi)
        Percentile clip for the colour limits, computed over finite pixels
        only (and over positive pixels only when log=True).
    log : bool
        LogNorm, using positive values only.
    mask : (ny, nx) bool array, optional
        True where the pixel should be SHOWN. Build it with good_pixels(),
        not by thresholding raw coverage -- `coverage > 0.3` masks
        everything, because the mean of coverage is K/N_inside (~0.19 at
        15"), not ~1.
    colorbar : bool
        Attach one. Set False when you are managing a shared colorbar.
    east_left : bool
        Put East on the left, the usual orientation for a sky image.

        This is NOT cosmetic here. fit_celestial_wcs produces CD1_1 > 0 (a
        positive determinant), i.e. RA increasing with column index, which
        is the mirror of the convention nearly every sky image follows. The
        array and its WCS agree with each other, so coordinates read out
        correctly either way -- but drawn as stored the field is flipped
        left-right compared with how you would see it in DS9 or in a paper.

        show_fibers has always inverted its x axis, so with east_left=False
        the same field plotted as a map and as fibres comes out mirrored
        between the two. Both functions take this flag and default it to
        True so they agree. Set False on both to see the array as stored.

    Returns
    -------
    The Axes.
    """
    d = np.asarray(data, float)
    if d.ndim != 2:
        raise ValueError(
            f"show_map wants a 2D plane, got shape {d.shape}. For a cube or a "
            "multi-line map, pick a plane (res['map'][res['index']('Halpha')]) "
            "or use show_planes / show_wave.")

    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != d.shape:
            raise ValueError(f"mask shape {mask.shape} != data shape {d.shape}")
        # Copy only now that we must: np.asarray above avoids a copy of a
        # large plane in the common unmasked case.
        d = np.where(mask, d, np.nan)

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        raise ValueError(
            "nothing finite to plot. If you passed a mask, this is usually a "
            "raw-coverage threshold: use good_pixels(result, 0.5) instead of "
            "result['coverage'] > 0.3.")

    if log:
        pos = finite[finite > 0]
        if pos.size == 0:
            raise ValueError("log=True but no positive values")
        vmin, vmax = np.percentile(pos, pct)
        kw["norm"] = LogNorm(vmin=max(vmin, pos.min()), vmax=vmax)
    else:
        kw.setdefault("vmin", np.percentile(finite, pct[0]))
        kw.setdefault("vmax", np.percentile(finite, pct[1]))

    if ax is None:
        fig = plt.figure(figsize=(5.4, 4.6))
        ax = (fig.add_subplot(projection=wcs) if wcs is not None
              else fig.add_subplot())

    # origin="lower" ALWAYS -- see the module docstring. Without it the
    # no-WCS path is flipped vertically while the WCS path is not.
    im = ax.imshow(d, cmap=cmap, origin="lower", **kw)

    if colorbar:
        # ax.figure.colorbar, not plt.colorbar: the latter goes through the
        # pyplot "current figure", which is the wrong one as soon as the
        # caller is building several figures or working in a notebook cell
        # that already made one.
        ax.figure.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title, fontsize=10)
    if wcs is not None:
        ax.set_xlabel("RA"); ax.set_ylabel("Dec")
        if grid:
            ax.grid(color="w", alpha=0.25, ls=":")
    else:
        ax.set_xlabel("x [pixels]"); ax.set_ylabel("y [pixels]")

    # Flip to East-left unless asked not to. Checked against the axis's
    # current direction rather than applied blindly, so passing a shared
    # `ax` through several calls cannot flip it twice.
    if east_left:
        lo, hi = ax.get_xlim()
        if hi > lo:
            ax.invert_xaxis()
    return ax


def show_planes(cube, names=None, wcs=None, ncols=3, label="",
                figsize_per=(5.0, 4.3), **kw):
    """
    One panel per plane, each with its own stretch and colorbar.

    `cube` may be (N, ny, nx) or a bare (ny, nx) -- read_map squeezes a
    single-line map to 2D, so accepting both is the difference between this
    working and a TypeError about "Invalid shape".

    names : sequence of str, optional
        Panel titles. Defaults to plane indices. If shorter than the cube,
        only that many planes are drawn, which is the usual way to plot the
        first few lines.
    """
    arr = np.asarray(cube, float)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"expected (N, ny, nx) or (ny, nx), got {arr.shape}")

    if names is None:
        names = [f"plane {i}" for i in range(arr.shape[0])]
    names = list(names)
    n = min(len(names), arr.shape[0])
    if n == 0:
        raise ValueError("no planes to plot")

    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    for i in range(n):
        ax = (fig.add_subplot(nrows, ncols, i + 1, projection=wcs)
              if wcs is not None else fig.add_subplot(nrows, ncols, i + 1))
        show_map(arr[i], wcs=wcs, ax=ax, title=names[i], label=label, **kw)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# wavelength slices out of a cube
# ---------------------------------------------------------------------------

def show_wave(src, waves, width: float = 0.0, combine: str = "mean",
              ncols: int = 3, label: Optional[str] = None, hdu: str = "FLUX",
              units: str = "surface_brightness", mask=None, **kw):
    """
    Plot one or more wavelengths out of a cube (or named lines out of a map).

    Parameters
    ----------
    src : str, Path, or dict
        A cube/map FITS path, or a result dict from build_cube / build_map /
        read_cube / read_map. Passing a PATH is much cheaper: only the
        wavelength windows you asked for are read off disk, so a 3.3 GB cube
        costs a few MB. Passing a dict uses what is already in memory.
    waves : float or sequence of float
        Wavelengths in Angstrom. The nearest plane is used, and the panel
        title reports the wavelength actually found -- so a 0.4 A miss is
        visible rather than silent.
    width : float
        Window width in Angstrom. 0 (default) plots the single nearest
        plane. A real spectral cube plane is noisy, so a few Angstrom
        averaged across the line is usually a better picture; width=0 is
        right for a map, whose "planes" are already integrated line fluxes.
    combine : {"mean", "sum"}
        How to collapse the window. "mean" keeps the surface-brightness
        units of a single plane; "sum" integrates over the window, which
        changes the units to per-window and is what you want for a
        narrow-band flux image.
    mask : (ny, nx) bool array, optional
        Passed to show_map. Build it with good_pixels().

    Returns
    -------
    The Figure.
    """
    waves = np.atleast_1d(np.asarray(waves, float))
    if waves.size == 0:
        raise ValueError("no wavelengths given")
    if combine not in ("mean", "sum"):
        raise ValueError(f"combine must be 'mean' or 'sum', got {combine!r}")

    planes, titles, wcs2d, bunit = [], [], None, ""

    if isinstance(src, dict):
        data = src.get("data")
        if data is None:                      # a builder result, not a reader
            data = src.get("cube", src.get("map"))
        if data is None:
            raise KeyError("dict has none of 'data', 'cube', 'map'")
        data = np.asarray(data, float)
        if data.ndim == 2:
            data = data[None, ...]
        wave_all = np.asarray(src.get("wave"), float)
        wcs2d = src.get("wcs2d") or src.get("wcs")
        bunit = src.get("bunit", "")
        names = src.get("line_names") or []
        for w in waves:
            i, lo, hi = _window(wave_all, w, width)
            planes.append(_collapse(data[lo:hi + 1], combine))
            titles.append(_title(w, wave_all, i, lo, hi, names))
    else:
        from astropy.io import fits
        from .cube_build import read_cube
        with fits.open(src) as h:
            is_cube = "WAVE" in h and np.asarray(h[hdu].data).ndim == 3
        if is_cube:
            for w in waves:
                # Read only the window: this is why passing a path beats
                # loading the whole cube first.
                half = max(width / 2.0, 1e-6)
                r = read_cube(src, units=units, hdu=hdu,
                              wave_range=(w - half - 5.0, w + half + 5.0),
                              with_ivar=False)
                wsub = np.asarray(r["wave"], float)
                d = np.asarray(r["data"], float)
                if d.ndim == 2:
                    d = d[None, ...]
                i, lo, hi = _window(wsub, w, width)
                planes.append(_collapse(d[lo:hi + 1], combine))
                titles.append(_title(w, wsub, i, lo, hi, []))
                wcs2d, bunit = r.get("wcs2d"), r.get("bunit", "")
        else:
            from .mapcat_build import read_map
            m = read_map(src, units=units, hdu=hdu)
            d = np.asarray(m["data"], float)
            if d.ndim == 2:
                d = d[None, ...]
            wave_all = np.asarray(m["wave"], float)
            wcs2d, bunit = m["wcs"], m.get("bunit", "")
            names = m.get("line_names") or []
            for w in waves:
                i, lo, hi = _window(wave_all, w, 0.0)
                planes.append(d[i])
                titles.append(_title(w, wave_all, i, lo, hi, names))

    return show_planes(np.stack(planes), titles, wcs=wcs2d, ncols=ncols,
                       label=(bunit if label is None else label),
                       mask=mask, **kw)


def _window(wave, target, width):
    """Nearest-plane index, plus the inclusive window around it."""
    wave = np.asarray(wave, float)
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("no usable wavelength array")
    i = int(np.argmin(np.abs(wave - target)))
    if width <= 0:
        return i, i, i
    sel = np.nonzero(np.abs(wave - target) <= width / 2.0)[0]
    if sel.size == 0:
        return i, i, i
    return i, int(sel[0]), int(sel[-1])


def _collapse(block, combine):
    if block.shape[0] == 1:
        return block[0]
    return (np.nanmean(block, axis=0) if combine == "mean"
            else np.nansum(block, axis=0))


def _title(target, wave, i, lo, hi, names):
    got = float(wave[i])
    name = names[i] if (names and i < len(names)) else None
    base = f"{name} " if name else ""
    if hi > lo:
        return (f"{base}{target:.1f} A  "
                f"[{wave[lo]:.1f}-{wave[hi]:.1f}, {hi - lo + 1} planes]")
    miss = abs(got - target)
    return (f"{base}{got:.2f} A" if miss < 0.05
            else f"{base}{got:.2f} A (asked {target:.2f})")


# ---------------------------------------------------------------------------
# the fibres themselves
# ---------------------------------------------------------------------------

def show_fibers(cat, c, label="", s=16, cmap="RdBu_r", pct=(2, 98), ax=None,
                colorbar=True, east_left=True, **kw):
    """
    The fibres themselves, in tangent-plane arcsec -- no resampling.

    Use this rather than a map whenever you want the DATA instead of a
    picture of it: a BPT diagram, a histogram, a fit. At 9" pixels a 35"
    kernel puts each fibre on roughly 15 pixels, so a per-pixel scatter plot
    shows every measurement ~15 times over, with neighbouring points
    correlated. Density contours and mixture models fitted to that are
    overconfident by about sqrt(15). The fibre catalog has one independent
    row per measurement.

    Parameters
    ----------
    cat : mapping with "x_arcsec" and "y_arcsec"
        A FIBERCAT dict or astropy Table (result["fibercat"], or
        read_map(...)["fibercat"]).
    c : (K,) array
        The quantity to colour by. Non-finite entries are dropped.
    """
    for key in ("x_arcsec", "y_arcsec"):
        if key not in getattr(cat, "colnames", cat):
            raise KeyError(
                f"fibre catalog has no {key!r}. Available: "
                f"{list(getattr(cat, 'colnames', cat))[:12]}")

    v = np.asarray(c, float)
    xs = np.asarray(cat["x_arcsec"], float)
    ys = np.asarray(cat["y_arcsec"], float)
    if not (v.shape == xs.shape == ys.shape):
        raise ValueError(
            f"length mismatch: values {v.shape}, x {xs.shape}, y {ys.shape}. "
            "The catalog must be the fibres that produced these values.")

    good = np.isfinite(v)
    if not good.any():
        raise ValueError("no finite values to plot")

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    lo, hi = np.percentile(v[good], pct)
    sc = ax.scatter(xs[good], ys[good], c=v[good], s=s, cmap=cmap,
                    vmin=lo, vmax=hi, **kw)
    if colorbar:
        ax.figure.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_aspect("equal")
    # x_arcsec is the skyoffset longitude, i.e. +East, so inverting gives the
    # conventional North-up East-left orientation. Same default as show_map,
    # so a fibre plot and a map of the same field are never mirrored.
    if east_left:
        lo, hi = ax.get_xlim()
        if hi > lo:
            ax.invert_xaxis()
    return ax
