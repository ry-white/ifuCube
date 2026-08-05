#!/usr/bin/env python

"""
healpix_build.py
----------------
Bridge from this package's WCS-based maps/cubes (tangent-plane, built by
cube_build.py / mapcat_build.py / mosaic.py) onto a HEALPix grid.

Why this is a separate step
----------------------------
Everything upstream (geometry.py, kernels.py, mosaic.py) works in a flat
tangent-plane (TAN) projection, which is the right tool for a single IFU
footprint or a WCS-based mosaic of several pointings.

HEALPix is a different, whole-sky equal-area pixelization, indexed by
`nside` (must be a power of 2), with `npix = 12 * nside**2` pixels total.
It's the right tool once you want to combine/compare against all-sky or
large-area products (e.g. Planck, other HEALPix-native surveys), or you
want strictly equal-area pixels at every declination.

Why binning instead of reproject_to_healpix
--------------------------------------------
reproject's reproject_to_healpix() builds a *full-sky* output array and
asks, for every one of its 12*nside**2 pixel centers, "does this land
inside my input image?". For an IFU field that's a few arcmin across,
you need nside fine enough that several HEALPix pixel centers actually
fall inside that tiny patch -- which, worked out, means nside in the
hundreds of thousands, and a full-sky array at that nside needs hundreds
of GB of RAM even in float32. That's true regardless of framing; it's
not a bug, it's just the wrong algorithm for a small footprint.

Instead, this module maps each *input* pixel's sky coordinate directly
to a HEALPix index (astropy_healpix.lonlat_to_healpix) and bins/averages
values into whichever HEALPix pixels are actually hit. Memory scales
with your input data size, not with nside**2, so you can go to whatever
nside matches your native spaxel scale without any special-casing.

Requires: astropy-healpix (pip install astropy-healpix)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


def suggest_nside(pixel_size_arcsec: float, round: str = "nearest") -> int:
    """
    Suggest a HEALPix nside matching (or, with round='up', at least as
    fine as) a given angular pixel scale.

    Parameters
    ----------
    pixel_size_arcsec : float
        Desired angular resolution in arcsec.
    round : {"nearest", "up", "down"}
        Rounding behaviour, forwarded to astropy_healpix.

    Returns
    -------
    nside : int
        Valid HEALPix nside (always a power of 2). Matching a typical
        IFU spaxel scale (a few arcsec) can mean nside in the hundreds
        of thousands -- that's fine with the binning approach used here
        (see module docstring), just don't be alarmed by the number.
    """
    from astropy_healpix import pixel_resolution_to_nside

    nside = pixel_resolution_to_nside(pixel_size_arcsec * u.arcsec, round=round)
    return int(nside)


def wcs_map_to_healpix(
    data: np.ndarray,
    wcs2d: WCS,
    nside: int,
    nested: bool = True,
    coord_system_out: str = "icrs",
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin a single 2D WCS image (e.g. result["map"] from mapcat_build, or
    one plane of a cube) onto HEALPix pixels by mapping each input pixel's
    sky coordinate directly to a HEALPix index.

    Every finite-valued input pixel contributes to exactly one HEALPix
    pixel. If several input pixels land in the same HEALPix pixel (the
    usual case when nside is coarser than your native pixel scale), their
    values are combined as an (optionally weighted) mean.

    Parameters
    ----------
    data : (ny, nx) array
        Image data (e.g. a FLUX plane). NaNs are dropped.
    wcs2d : astropy.wcs.WCS
        Celestial WCS describing `data` (e.g. result["wcs2d"], or
        WCS(header).celestial read back from a saved FITS file).
    nside : int
        HEALPix resolution parameter (power of 2). See suggest_nside().
    nested : bool
        HEALPix pixel ordering scheme. Use True (NESTED) unless you have
        a specific reason to want RING (some external tools expect RING).
    coord_system_out : str
        Output sky frame, e.g. "icrs" or "galactic".
    weights : (ny, nx) array, optional
        E.g. an IVAR map. If given, values are combined as an
        inverse-variance-style weighted mean instead of a plain mean,
        and the returned `wsum_out` is the summed weight per HEALPix
        pixel (directly usable as an output IVAR).

    Returns
    -------
    pix_idx : (Ncovered,) int64
        HEALPix pixel indices that received at least one valid sample.
    value_out : (Ncovered,) float32
        (Weighted) mean value in each covered HEALPix pixel.
    wsum_out : (Ncovered,) float32
        Summed weight per covered pixel (all 1s if `weights` is None,
        i.e. just the sample count).
    counts_out : (Ncovered,) int64
        Number of input pixels that landed in each covered HEALPix pixel
        -- useful as a simple coverage/Neff-style diagnostic.
    """
    from astropy_healpix import HEALPix

    data = np.asarray(data, float)
    ny, nx = data.shape

    yy, xx = np.mgrid[0:ny, 0:nx]
    sky = wcs2d.pixel_to_world(xx.ravel(), yy.ravel())
    frame = coord_system_out.lower()
    if frame == "galactic":
        sky = sky.galactic
        lon, lat = sky.l, sky.b
    else:
        sky = sky.icrs
        lon, lat = sky.ra, sky.dec

    hp = HEALPix(nside=nside, order="nested" if nested else "ring", frame=coord_system_out)
    pix = hp.lonlat_to_healpix(lon, lat).astype(np.int64)

    values = data.ravel()
    if weights is not None:
        w = np.asarray(weights, float).ravel()
    else:
        w = np.ones_like(values)

    good = np.isfinite(values) & np.isfinite(w) & (w >= 0)
    pix = pix[good]
    values = values[good]
    w = w[good]

    if pix.size == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype="f4"),
                np.array([], dtype="f4"), np.array([], dtype=np.int64))

    order_idx = np.argsort(pix, kind="stable")
    pix_sorted = pix[order_idx]
    values_sorted = values[order_idx]
    w_sorted = w[order_idx]

    unique_pix, start_idx, counts = np.unique(pix_sorted, return_index=True, return_counts=True)

    wsum = np.add.reduceat(w_sorted, start_idx)
    wval_sum = np.add.reduceat(values_sorted * w_sorted, start_idx)

    with np.errstate(divide="ignore", invalid="ignore"):
        value_out = np.where(wsum > 0, wval_sum / wsum, np.nan)

    return (
        unique_pix.astype(np.int64),
        value_out.astype("f4"),
        wsum.astype("f4"),
        counts.astype(np.int64),
    )


def wcs_fits_to_healpix_table(
    in_fits: str | Path,
    out_fits: str | Path,
    nside: int,
    hdu: str | int = "FLUX",
    weights_hdu: Optional[str | int] = None,
    nested: bool = True,
    coord_system_out: str = "icrs",
    extra_header: Optional[dict] = None,
) -> None:
    """
    Read a WCS-tagged map/mosaic FITS file (as written by save_map_to_fits,
    save_cube_to_fits, or mosaic.py) and write out a partial-sky HEALPix
    FITS table (explicit pixel indexing -- only covered pixels are stored,
    since an IFU footprint covers a tiny fraction of the sky).

    If the input HDU is 3D (e.g. a multi-line map or a spectral cube),
    every plane along axis 0 is binned separately and stored as columns
    VALUE0, VALUE1, ... plus NSAMP0, NSAMP1, ... (per-plane input-pixel
    counts -- some HEALPix pixels may have partial coverage per plane).

    Parameters
    ----------
    weights_hdu : str or int, optional
        HDU name/index for a weights array (e.g. "IVAR"). If given, values
        are combined as a weighted mean per HEALPix pixel, and a WEIGHT
        (or WEIGHT0, WEIGHT1, ...) column with the summed weight is written
        -- directly usable downstream as an output IVAR.

    Output layout
    -------------
    HDU 0 : PRIMARY (NSIDE, ORDERING, COORDSYS keywords + extra_header)
    HDU 1 : BinTable with columns:
        PIXEL          : int64 HEALPix pixel index (explicit indexing)
        VALUE[N]       : float32 (weighted) mean value
        WEIGHT[N]      : float32 summed weight (only if weights_hdu given)
        NSAMP[N]       : int64 number of native pixels contributing
    """
    in_fits = Path(in_fits)
    out_fits = Path(out_fits)
    out_fits.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(in_fits, memmap=False) as hdul:
        data = np.asarray(hdul[hdu].data, float)
        wcs2d = WCS(hdul[hdu].header).celestial
        wdata = np.asarray(hdul[weights_hdu].data, float) if weights_hdu is not None else None

    if data.ndim == 2:
        planes = [data]
        wplanes = [wdata] if wdata is not None else [None]
    elif data.ndim == 3:
        planes = [data[k] for k in range(data.shape[0])]
        if wdata is not None:
            if wdata.shape != data.shape:
                raise ValueError(f"weights shape {wdata.shape} != data shape {data.shape}")
            wplanes = [wdata[k] for k in range(wdata.shape[0])]
        else:
            wplanes = [None] * len(planes)
    else:
        raise ValueError(f"Expected a 2D or 3D array in HDU '{hdu}', got shape {data.shape}")

    # Bin every plane; collect the union of covered pixels across planes.
    per_plane = []
    all_pix = set()
    for plane, wplane in zip(planes, wplanes):
        pix_idx, value_out, wsum_out, counts_out = wcs_map_to_healpix(
            plane, wcs2d, nside=nside, nested=nested,
            coord_system_out=coord_system_out, weights=wplane,
        )
        per_plane.append((pix_idx, value_out, wsum_out, counts_out))
        all_pix.update(pix_idx.tolist())

    if not all_pix:
        raise ValueError("No valid (covered) HEALPix pixels found -- check input WCS/coverage")

    union_pix = np.array(sorted(all_pix), dtype=np.int64)
    pix_to_row = {p: i for i, p in enumerate(union_pix)}
    n_row = union_pix.size

    primary = fits.PrimaryHDU()
    primary.header["NSIDE"] = (nside, "HEALPix resolution parameter")
    primary.header["ORDERING"] = ("NESTED" if nested else "RING", "HEALPix pixel ordering")
    primary.header["COORDSYS"] = (coord_system_out.upper(), "Output sky frame")
    primary.header["INDXSCHM"] = ("EXPLICIT", "Partial-sky: only covered pixels stored")
    if extra_header:
        for k, v in extra_header.items():
            primary.header[k] = v

    cols = [fits.Column(name="PIXEL", format="K", array=union_pix)]
    multi = len(per_plane) > 1
    for i, (pix_idx, value_out, wsum_out, counts_out) in enumerate(per_plane):
        val_full = np.full(n_row, np.nan, dtype="f4")
        w_full = np.full(n_row, 0.0, dtype="f4")
        n_full = np.zeros(n_row, dtype=np.int64)
        rows = np.array([pix_to_row[p] for p in pix_idx.tolist()], dtype=np.int64)
        val_full[rows] = value_out
        w_full[rows] = wsum_out
        n_full[rows] = counts_out

        suffix = str(i) if multi else ""
        cols.append(fits.Column(name=f"VALUE{suffix}", format="E", array=val_full))
        if weights_hdu is not None:
            cols.append(fits.Column(name=f"WEIGHT{suffix}", format="E", array=w_full))
        cols.append(fits.Column(name=f"NSAMP{suffix}", format="K", array=n_full))

    table_hdu = fits.BinTableHDU.from_columns(cols, name="HEALPIX")
    fits.HDUList([primary, table_hdu]).writeto(out_fits, overwrite=True)
    print(f"[ok] Wrote HEALPix table ({n_row} of {12 * nside**2} pixels covered) -> {out_fits}")


def plot_healpix_scatter(
    pix_idx: np.ndarray,
    values: np.ndarray,
    nside: int,
    nested: bool = True,
    coord_system: str = "icrs",
    ax=None,
    **scatter_kwargs,
):
    """
    Scatter-plot a sparse/local HEALPix table using plain matplotlib.

    This is the right tool for a single pointing (or a few pointings) at
    a fine, matched-to-native-resolution nside: healpy's mollview/gnomview
    expect a *full-sky* array of length 12*nside**2, which is not
    allocatable at the nside values this module typically uses (see
    module docstring). Since your data only ever covers a tiny patch of
    sky, converting the covered pixel indices back to RA/Dec and
    scattering them is both cheap and correct.

    If instead you've merged many pointings into one large-area or
    survey-scale table and picked a *coarser* nside for that (e.g. a few
    hundred to a few thousand, appropriate for a multi-degree survey
    footprint), a full-sky array becomes feasible again and healpy's
    `hp.mollview` (whole sky) or `hp.gnomview` (zoomed, centered on a
    point) are the standard tools -- see the note at the end of this
    docstring for a minimal example.

    Parameters
    ----------
    pix_idx : (N,) int array
        HEALPix pixel indices, e.g. the PIXEL column of a table written
        by wcs_fits_to_healpix_table.
    values : (N,) array
        Value to color by, e.g. a VALUE column.
    nside, nested, coord_system : as elsewhere in this module -- must
        match the NSIDE/ORDERING/COORDSYS the table was written with.
    ax : matplotlib Axes, optional
        Axes to plot into. A new figure/axes is created if not given.
    **scatter_kwargs :
        Passed through to ax.scatter (e.g. s=8, cmap='inferno').

    Returns
    -------
    ax : matplotlib Axes

    healpy alternative for large/merged sky areas
    ----------------------------------------------
        import healpy as hp
        import numpy as np
        m = np.full(12 * nside**2, hp.UNSEEN)
        m[pix_idx] = values
        hp.mollview(m, nest=nested, title="Merged survey")     # whole sky
        hp.gnomview(m, nest=nested, rot=(ra0, dec0), xsize=400)  # zoomed
    """
    import matplotlib.pyplot as plt
    from astropy_healpix import HEALPix

    hp_grid = HEALPix(nside=nside, order="nested" if nested else "ring", frame=coord_system)
    lon, lat = hp_grid.healpix_to_lonlat(np.asarray(pix_idx))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    kwargs = dict(s=8, cmap="inferno")
    kwargs.update(scatter_kwargs)
    sca = ax.scatter(lon.to(u.deg).value, lat.to(u.deg).value, c=values, **kwargs)
    ax.set_xlabel(f"{'RA' if coord_system=='icrs' else 'l'} [deg]")
    ax.set_ylabel(f"{'Dec' if coord_system=='icrs' else 'b'} [deg]")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.median(lat.to(u.deg).value))))
    ax.invert_xaxis()  # RA/longitude convention: increases to the left
    plt.colorbar(sca, ax=ax, label="value")
    return ax


def merge_healpix_tables(
    in_fits_list: list,
    out_fits: str | Path,
    value_col: str = "VALUE",
    weight_col: Optional[str] = "WEIGHT",
) -> None:
    """
    Merge/coadd several per-pointing HEALPix tables (as written by
    wcs_fits_to_healpix_table) into one combined sparse table.

    Overlapping pointings land in identical HEALPix pixel indices by
    construction (no reprojection involved), so combining them is an
    exact weighted average per pixel -- not an image interpolation.

    Parameters
    ----------
    in_fits_list : list of str or Path
        Per-pointing HEALPix FITS tables. Must all share the same
        NSIDE/ORDERING/COORDSYS (checked and enforced).
    out_fits : str or Path
        Output merged table path.
    value_col : str
        Name of the value column to combine (e.g. "VALUE", or "VALUE0"
        for a specific line/plane if the inputs are multi-plane).
    weight_col : str, optional
        Name of the weight column (e.g. "WEIGHT", or "WEIGHT0"). If a
        table doesn't have this column (weights_hdu wasn't used when it
        was built), its samples are combined with weight=1 instead. Set
        to None to force a plain (unweighted) mean for all inputs.

    Output layout
    -------------
    HDU 0 : PRIMARY (NSIDE, ORDERING, COORDSYS carried over)
    HDU 1 : BinTable with columns PIXEL, VALUE, WEIGHT, NSAMP
        (WEIGHT is the summed weight across all contributing pointings;
        NSAMP is the summed input-pixel count across all pointings.)
    """
    out_fits = Path(out_fits)
    out_fits.parent.mkdir(parents=True, exist_ok=True)

    nside = ordering = coordsys = None
    pix_val_sums: dict = {}
    pix_wsums: dict = {}
    pix_nsamp: dict = {}

    for path in in_fits_list:
        with fits.open(path, memmap=False) as hdul:
            hdr = hdul[0].header
            tbl = hdul[1].data

            if nside is None:
                nside, ordering, coordsys = hdr["NSIDE"], hdr["ORDERING"], hdr["COORDSYS"]
            else:
                if hdr["NSIDE"] != nside or hdr["ORDERING"] != ordering or hdr["COORDSYS"] != coordsys:
                    raise ValueError(
                        f"{path}: NSIDE/ORDERING/COORDSYS mismatch "
                        f"({hdr['NSIDE']}/{hdr['ORDERING']}/{hdr['COORDSYS']} "
                        f"vs {nside}/{ordering}/{coordsys}). "
                        "All input tables must be built with the same nside/ordering/frame."
                    )

            pix = tbl["PIXEL"].astype(np.int64)
            val = np.asarray(tbl[value_col], float)
            if weight_col is not None and weight_col in tbl.columns.names:
                w = np.asarray(tbl[weight_col], float)
            else:
                w = np.ones_like(val)

            nsamp_col = "NSAMP" if "NSAMP" in tbl.columns.names else None
            nsamp = np.asarray(tbl[nsamp_col], np.int64) if nsamp_col else np.ones_like(pix)

            good = np.isfinite(val) & np.isfinite(w) & (w >= 0)
            for p, v, wi, n in zip(pix[good], val[good], w[good], nsamp[good]):
                pix_val_sums[p] = pix_val_sums.get(p, 0.0) + v * wi
                pix_wsums[p] = pix_wsums.get(p, 0.0) + wi
                pix_nsamp[p] = pix_nsamp.get(p, 0) + n

    if not pix_wsums:
        raise ValueError("No valid pixels found across input tables")

    union_pix = np.array(sorted(pix_wsums.keys()), dtype=np.int64)
    wsum_out = np.array([pix_wsums[p] for p in union_pix], dtype="f4")
    val_out = np.array(
        [pix_val_sums[p] / pix_wsums[p] if pix_wsums[p] > 0 else np.nan for p in union_pix],
        dtype="f4",
    )
    nsamp_out = np.array([pix_nsamp[p] for p in union_pix], dtype=np.int64)

    primary = fits.PrimaryHDU()
    primary.header["NSIDE"] = (nside, "HEALPix resolution parameter")
    primary.header["ORDERING"] = (ordering, "HEALPix pixel ordering")
    primary.header["COORDSYS"] = (coordsys, "Output sky frame")
    primary.header["INDXSCHM"] = ("EXPLICIT", "Partial-sky: only covered pixels stored")
    primary.header["NMERGE"] = (len(in_fits_list), "Number of input pointings merged")

    cols = [
        fits.Column(name="PIXEL", format="K", array=union_pix),
        fits.Column(name="VALUE", format="E", array=val_out),
        fits.Column(name="WEIGHT", format="E", array=wsum_out),
        fits.Column(name="NSAMP", format="K", array=nsamp_out),
    ]
    table_hdu = fits.BinTableHDU.from_columns(cols, name="HEALPIX")
    fits.HDUList([primary, table_hdu]).writeto(out_fits, overwrite=True)
    print(f"[ok] Merged {len(in_fits_list)} pointings -> {union_pix.size} covered pixels -> {out_fits}")


def crossmatch_catalog_to_healpix(
    healpix_fits: str | Path,
    catalog,
    ra_col: str = "RAJ2000",
    dec_col: str = "DEJ2000",
    name_col: Optional[str] = None,
    radius_col: Optional[str] = None,
    radius_unit: str = "arcsec",
    default_radius_arcsec: float = 30.0,
    value_col: str = "VALUE",
    weight_col: Optional[str] = "WEIGHT",
    min_matched_pixels: int = 1,
):
    """
    Cross-match a sky catalog (e.g. the WISE Catalog of Galactic HII
    Regions) against a HEALPix table written by this module (a single
    pointing from wcs_fits_to_healpix_table, or a merged table from
    merge_healpix_tables).

    Design note: rather than enumerating every HEALPix pixel inside each
    catalog source's disk (cone_search_lonlat) -- which gets slow fast at
    the fine nside this module uses, since cone size scales with
    (radius/pixel_scale)**2 -- this walks it the other way: your own
    footprint is always small (bounded by your actual data), so for each
    candidate source it computes angular separation from the source
    center to your footprint's own covered pixels directly. A cheap
    vectorized bounding pre-filter (separation from the footprint's
    centroid) is applied first so the per-source check only runs for
    catalog entries that could plausibly overlap at all.

    Parameters
    ----------
    healpix_fits : str or Path
        HEALPix FITS table (PIXEL + value/weight columns + NSIDE/ORDERING/
        COORDSYS header keywords) as written elsewhere in this module.
    catalog : astropy.table.Table or dict-like of columns
        Catalog to cross-match, e.g. an astropy Table read from the WISE
        Catalog of Galactic HII Regions.
    ra_col, dec_col : str
        Column names for catalog RA/Dec, in degrees. COORDSYS in the
        HEALPix file must be "ICRS" for these to be treated as RA/Dec
        (use a "galactic" HEALPix table with l/b catalog columns instead
        if that's what you're matching).
    name_col : str, optional
        Column with a source identifier/name, carried into the output.
    radius_col : str, optional
        Column with each source's angular radius/size. If not given,
        every source uses `default_radius_arcsec`.
    radius_unit : str
        Unit of `radius_col` values: "arcsec", "arcmin", or "deg".
    default_radius_arcsec : float
        Radius used when `radius_col` is None.
    value_col, weight_col : str
        Columns in the HEALPix table to summarize per match (weight_col
        may be None or absent -- falls back to an unweighted mean).
    min_matched_pixels : int
        Only return sources with at least this many matched footprint
        pixels (i.e. drop non-detections from the output).

    Returns
    -------
    astropy.table.Table with columns:
        NAME, RA, DEC, RADIUS_ARCSEC,
        N_PIX_MATCHED     : footprint pixels found inside the source disk
        N_PIX_REGION_EST  : ~ pi*radius^2 / pixel_area (source disk area
                             in pixel units -- an *estimate* of how many
                             HEALPix pixels the full disk would contain,
                             not an exact enumeration; see design note)
        COVERAGE_FRAC     : N_PIX_MATCHED / N_PIX_REGION_EST, clipped to
                             [0, 1] (can slightly exceed 1 due to the disk
                             area being an approximation -- clipped)
        MEAN_VALUE        : (weighted) mean of value_col over matched pixels
        SUMMED_WEIGHT     : summed weight_col over matched pixels
    """
    from astropy.table import Table, unique 
    from astropy.coordinates import SkyCoord
    from astropy_healpix import HEALPix, nside_to_pixel_area

    with fits.open(healpix_fits, memmap=False) as hdul:
        hdr = hdul[0].header
        tbl = hdul[1].data

    nside = int(hdr["NSIDE"])
    nested = hdr["ORDERING"].strip().upper() == "NESTED"
    coordsys = hdr["COORDSYS"].strip().lower()

    hp_grid = HEALPix(nside=nside, order="nested" if nested else "ring", frame=coordsys)
    footprint_pix = tbl["PIXEL"].astype(np.int64)
    footprint_lon, footprint_lat = hp_grid.healpix_to_lonlat(footprint_pix)
    footprint_coord = SkyCoord(footprint_lon, footprint_lat, frame=coordsys)

    values = np.asarray(tbl[value_col], float)
    if weight_col is not None and weight_col in tbl.columns.names:
        weights = np.asarray(tbl[weight_col], float)
    else:
        weights = np.ones_like(values)

    pixel_area = nside_to_pixel_area(nside).to(u.arcsec ** 2).value

    # Footprint bounding circle, for a cheap pre-filter.
    centroid = SkyCoord(np.mean(footprint_lon), np.mean(footprint_lat), frame=coordsys)
    footprint_radius_arcsec = float(centroid.separation(footprint_coord).to(u.arcsec).value.max())

    cat_ra = np.asarray(catalog[ra_col], float)
    cat_dec = np.asarray(catalog[dec_col], float)
    n_cat = cat_ra.size

    if radius_col is not None:
        raw_radius = np.asarray(catalog[radius_col], float)
        radius_arcsec = (raw_radius * u.Unit(radius_unit)).to(u.arcsec).value
    else:
        radius_arcsec = np.full(n_cat, float(default_radius_arcsec))

    cat_coord = SkyCoord(cat_ra * u.deg, cat_dec * u.deg, frame="icrs")
    if coordsys != "icrs":
        cat_coord = cat_coord.transform_to(coordsys)

    # Cheap vectorized pre-filter: only keep sources whose center could
    # plausibly overlap the footprint's bounding circle at all.
    sep_to_centroid = centroid.separation(cat_coord).to(u.arcsec).value
    candidate_mask = sep_to_centroid <= (footprint_radius_arcsec + radius_arcsec)
    candidate_idx = np.nonzero(candidate_mask)[0]

    names, out_ra, out_dec, out_radius = [], [], [], []
    n_matched_list, n_region_est_list, cov_frac_list = [], [], []
    mean_val_list, sum_w_list = [], []

    for i in candidate_idx:
        sep = cat_coord[i].separation(footprint_coord).to(u.arcsec).value
        matched = sep <= radius_arcsec[i]
        n_matched = int(np.count_nonzero(matched))
        if n_matched < min_matched_pixels:
            continue

        n_region_est = max(1.0, np.pi * radius_arcsec[i] ** 2 / pixel_area)
        cov_frac = float(np.clip(n_matched / n_region_est, 0.0, 1.0))

        w = weights[matched]
        v = values[matched]
        good = np.isfinite(v) & np.isfinite(w) & (w >= 0)
        wsum = float(np.sum(w[good]))
        mean_val = float(np.sum(v[good] * w[good]) / wsum) if wsum > 0 else float("nan")

        names.append(str(catalog[name_col][i]) if name_col is not None else str(i))
        out_ra.append(float(cat_ra[i]))
        out_dec.append(float(cat_dec[i]))
        out_radius.append(float(radius_arcsec[i]))
        n_matched_list.append(n_matched)
        n_region_est_list.append(n_region_est)
        cov_frac_list.append(cov_frac)
        mean_val_list.append(mean_val)
        sum_w_list.append(wsum)

    return Table(
        {
            "NAME": names,
            "RA": out_ra,
            "DEC": out_dec,
            "RADIUS_ARCSEC": out_radius,
            "N_PIX_MATCHED": n_matched_list,
            "N_PIX_REGION_EST": n_region_est_list,
            "COVERAGE_FRAC": cov_frac_list,
            "MEAN_VALUE": mean_val_list,
            "SUMMED_WEIGHT": sum_w_list,
        }
    )


def find_matching_pointings(
    pointings,
    catalog,
    pointing_ra_col: str = "sci_ra",
    pointing_dec_col: str = "dec",
    catalog_ra_col: str = "_RA_icrs",
    catalog_dec_col: str = "_DE_icrs",
    catalog_radius_col: Optional[str] = None,
    catalog_radius_unit: str = "arcsec",
    default_catalog_radius_arcsec: float = 30.0,
    lvm_fov_deg: float = 0.5,
):
    """
    Cross-match an LVM pointing summary table (e.g. your drpall-style FITS
    table) against a sky catalog (e.g. the WISE HII Region catalog) *before*
    downloading anything -- both tables here are just metadata/positions,
    so this step is cheap regardless of how many pointings or catalog
    entries you have.

    A pointing is kept if its field center falls within
    (lvm_fov_deg/2 + source_radius) of a catalog source -- i.e. the
    source's disk could plausibly fall at least partly inside that
    pointing's footprint. This is deliberately a generous pre-filter (see
    crossmatch_catalog_to_healpix for the precise pixel-level check you'd
    run later, once you actually have data in hand).

    Parameters
    ----------
    pointings : astropy.table.Table or dict-like of columns
        Your LVM pointing/observation summary table (e.g. an lvm_drpall
        FITS table), one row per exposure/observation.
    catalog : astropy.table.Table or dict-like of columns
        Sky catalog to match against (e.g. the WISE HII Region catalog).
    pointing_ra_col, pointing_dec_col : str
        Column names for the pointing center RA/Dec (degrees) in
        `pointings`. Check your actual file -- common names are "ra"/"dec"
        or "tile_ra"/"tile_dec"; this varies by data release.
    catalog_ra_col, catalog_dec_col : str
        Column names for RA/Dec (degrees) in `catalog`.
    catalog_radius_col : str, optional
        Column with each catalog source's angular radius/size. If not
        given, every source uses `default_catalog_radius_arcsec`.
    catalog_radius_unit : str
        Unit of `catalog_radius_col` values.
    lvm_fov_deg : float
        LVM-I's field of view (degrees) -- used as the matching aperture
        around each pointing center. ~0.5 deg is LVM-I's approximate FOV;
        adjust if you have a more precise number for your setup.

    Returns
    -------
    matched_pointings : same type as `pointings` (Table), containing only
        the rows with at least one plausible catalog match, in the order
        they were matched (a pointing matching multiple catalog sources
        appears once per match).
    match_info : astropy.table.Table
        One row per (pointing, catalog source) match, with columns
        POINTING_IDX (row index into `pointings`), CATALOG_IDX (row index
        into `catalog`), and SEP_ARCSEC (center-to-center separation).
    """
    from astropy.table import Table
    from astropy.coordinates import SkyCoord

    pointing_ra = np.asarray(pointings[pointing_ra_col], float)
    pointing_dec = np.asarray(pointings[pointing_dec_col], float)
    pointing_coord = SkyCoord(pointing_ra * u.deg, pointing_dec * u.deg, frame="icrs")

    cat_ra = np.asarray(catalog[catalog_ra_col], float)
    cat_dec = np.asarray(catalog[catalog_dec_col], float)
    cat_coord = SkyCoord(cat_ra * u.deg, cat_dec * u.deg, frame="icrs")

    if catalog_radius_col is not None:
        raw_radius = np.asarray(catalog[catalog_radius_col], float)
        radius_arcsec = (raw_radius * u.Unit(catalog_radius_unit)).to(u.arcsec).value
    else:
        radius_arcsec = np.full(cat_ra.size, float(default_catalog_radius_arcsec))

    # search_around_sky needs one seplimit, so use the largest plausible
    # aperture (fov/2 + max source radius) and refine per-source below.
    max_seplimit = (lvm_fov_deg / 2.0) * u.deg + radius_arcsec.max() * u.arcsec
    # search_around_sky(self=pointing_coord, arg=cat_coord, seplimit) returns
    # (indices_into_cat_coord, indices_into_pointing_coord, sep2d, dist3d) --
    # i.e. the FIRST return value indexes the argument you passed in, not
    # self. Verified empirically (not just from the docstring's field
    # names, which read ambiguously) before relying on it here.
    idx_cat, idx_pointing, sep2d, _ = pointing_coord.search_around_sky(cat_coord, max_seplimit)

    sep_arcsec = sep2d.to(u.arcsec).value
    keep = sep_arcsec <= ((lvm_fov_deg / 2.0) * 3600.0 + radius_arcsec[idx_cat])

    idx_pointing = idx_pointing[keep]
    idx_cat = idx_cat[keep]
    sep_arcsec = sep_arcsec[keep]

    match_info = Table(
        {"POINTING_IDX": idx_pointing, "CATALOG_IDX": idx_cat, "SEP_ARCSEC": sep_arcsec}
    )

    # De-duplicate for the returned pointings table: a pointing matching
    # several catalog sources should still only be downloaded/processed
    # once. match_info above retains every individual (pointing, source)
    # pairing if you need that detail.
    unique_pointing_idx = np.unique(idx_pointing)

    if isinstance(pointings, Table):
        matched_pointings = pointings[unique_pointing_idx]
    else:
        matched_pointings = Table(pointings)[unique_pointing_idx]

    print(
        f"[ok] {unique_pointing_idx.size} of {len(pointing_ra)} pointings "
        f"matched against {len(np.unique(idx_cat))} of {len(cat_ra)} catalog sources "
        f"({len(match_info)} total pointing-source pairs)"
    )
    return matched_pointings, match_info


def download_lvm_files(
    matched_pointings,
    column_map: Dict[str, str],
    filetype: str = "lvm_sframe",
    release: str = "sdsswork",
    username: Optional[str] = None,
    password: Optional[str] = None,
    dry_run: bool = True,
):
    """
    Download LVM data files for a set of matched pointings using
    sdss_access (pip install sdss-access), which resolves SDSS's abstract
    file-path templates into real SAS locations and handles the transfer.

    This is a thin wrapper -- it does the bookkeeping of turning table rows
    into the right `Access.add()` calls, plus a dry-run mode so you can see
    exactly what would be fetched before committing to a real download.

    Parameters
    ----------
    matched_pointings : astropy.table.Table or dict-like of columns
        Typically the first return value of find_matching_pointings(), or
        any table with one row per observation you want to fetch.
    column_map : dict
        Maps each keyword required by `filetype`'s path template to the
        column name in `matched_pointings` that holds it. Required keys
        depend on filetype -- e.g. for "lvm_sframe": mjd, drpver, tileid,
        expnum. Use `sdss_access.Access(release=release).lookup_keys(filetype)`
        yourself first if you're not sure what's required for a different
        filetype (e.g. "lvm_dap" needs more keys: tileid, dapver, expnum,
        snlevel, daptype, rspid, drpver, mjd).
    filetype : str
        sdss_access path-template name. Common LVM ones: "lvm_sframe"
        (calibrated row-stacked spectra -- what build.py's `cube` command
        reads), "lvm_dap" (per-line fit outputs -- what `dap-maps` reads),
        "lvm_drpall" (the observation summary table itself).
    release : str
        sdss_access release tag. Use whatever matches your access level --
        "sdsswork" or an "ipl-N" string for internal/collaboration access,
        a "DRxx" string once data is public. Check
        `sdss_access.Access.get_available_releases()` for valid values.
    username, password : str, optional
        SDSS SAS credentials. If both are None, sdss_access falls back to
        prompting interactively or reading a configured ~/.netrc -- if
        you've already got LVM access working some other way, you likely
        already have this set up and can leave these as None.
    dry_run : bool
        If True (the default -- deliberately safe), only prints what would
        be downloaded and where it would land locally; makes no network
        calls. Set False to actually fetch the files.

    Returns
    -------
    destinations : list of str
        Local file paths (existing after a real run; the paths *would*
        exist after a dry run).
    """
    # Need to implement HTTP access, as rsync doesn't seem to work.
    from sdss_access import Access

    a = Access(release=release)
    required_keys = a.lookup_keys(filetype)
    missing = [k for k in required_keys if k not in column_map]
    if missing:
        raise ValueError(
            f"column_map is missing required key(s) for filetype={filetype!r}: {missing}. "
            f"Full requirement: {required_keys}"
        )

    n = len(matched_pointings)
    destinations = []
    row_kwargs_list = []
    for i in range(n):
        row_kwargs = {k: matched_pointings[column_map[k]][i] for k in required_keys}
        row_kwargs_list.append(row_kwargs)
        dest = a.full(filetype, **row_kwargs)
        destinations.append(dest)

    if dry_run:
        print(f"[dry run] {n} file(s) would be fetched (filetype={filetype}, release={release}):")
        for kwargs, dest in zip(row_kwargs_list, destinations):
            print(f"    {kwargs} -> {dest}")
        print("Re-run with dry_run=False to actually download.")
        return destinations

    a.remote(username=username, password=password, inquire=(username is None and password is None))
    for kwargs in row_kwargs_list:
        a.add(filetype, **kwargs)
    a.set_stream()
    a.commit()
    print(f"[ok] Downloaded {n} file(s) via sdss_access (filetype={filetype}, release={release})")
    return destinations


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Bin a WCS-tagged map/mosaic FITS file onto a partial-sky HEALPix table."
    )
    p.add_argument("--in", dest="in_fits", required=True, help="Input FITS (map, cube, or mosaic output)")
    p.add_argument("--out", required=True, help="Output HEALPix FITS table path")
    p.add_argument("--hdu", default="FLUX", help="Input data HDU name (default: FLUX)")
    p.add_argument("--weights-hdu", default=None, help="Optional weights HDU name (e.g. IVAR)")
    p.add_argument("--nside", type=int, default=None, help="HEALPix nside (power of 2)")
    p.add_argument("--pixel-size-arcsec", type=float, default=None,
                   help="Alternative to --nside: derive nside from a desired angular resolution")
    p.add_argument("--ring", action="store_true", help="Use RING ordering instead of NESTED")
    p.add_argument("--frame", default="icrs", help="Output sky frame, e.g. icrs or galactic")

    args = p.parse_args()

    if args.nside is None and args.pixel_size_arcsec is None:
        p.error("must supply either --nside or --pixel-size-arcsec")
    nside = args.nside if args.nside is not None else suggest_nside(args.pixel_size_arcsec, round="up")
    if args.nside is None:
        print(f"[info] --pixel-size-arcsec {args.pixel_size_arcsec} -> nside={nside} "
              f"(npix={12 * nside**2:,})")

    wcs_fits_to_healpix_table(
        in_fits=args.in_fits,
        out_fits=args.out,
        nside=nside,
        hdu=args.hdu,
        weights_hdu=args.weights_hdu,
        nested=not args.ring,
        coord_system_out=args.frame,
    )


if __name__ == "__main__":
    main()

