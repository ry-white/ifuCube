#!/usr/bin/env python

"""
cube_build.py
-------------
High level cube builder

Wires together:
    - io_reader.FiberSet
    - geometry.make_grid_from_points
    - geometry.fit_celestial_wcs / make_wcs3d
    - kernels.build_projection_A / splat_cube / ivar_through_A / neff_from_A

Units of the output
-------------------
build_cube is flux-conserving: a voxel value is an amount of flux per output
pixel, and is therefore proportional to the pixel solid angle. Halve the pixel
size and every value drops by four while the total stays fixed.

That is the right thing to STORE, because it is the quantity the projection
actually conserves, but it is the wrong thing to COMPARE: two cubes built at
different pixel sizes are not comparable voxel by voxel, and averaging voxels
without dividing by the pixel area makes the answer depend on a build
parameter. Surface brightness has neither problem:

    surface brightness [flux/arcsec**2] = FLUX / PIXAR_A2
    its inverse variance                = IVAR * PIXAR_A2**2

the weights entering squared because Var(F/omega) = Var(F)/omega**2.

Both ends are supported, and the header always says which you have:

    save_cube_to_fits(..., units="surface_brightness")   write SB directly
    read_cube(path)                                      convert on the way in
    read_cube(path, units="flux_per_pixel")              leave it as stored

A further factor applies to extended emission. LVM fibers are 35.3" across on
a 37" pitch, so they collect

    APERFILL = pi * (35.3/2)**2 / ((sqrt(3)/2) * 37**2) = 0.8255

of the sky inside the footprint; smooth emission therefore reads about 17%
low. read_cube(..., correct_fill_factor=True) divides it out. It cancels in
line ratios and in any LVM-to-LVM comparison, and matters when comparing
against another survey or a model. See the mapcat_build module docstring.

Memory
------
The output is L x ny x nx values, twice over (FLUX and IVAR), so a cube is
memory-bound long before it is compute-bound. Measured on a 1801-fiber LVM
bundle: at 2" the grid is 789 x 908 = 716,412 pixels, and the arithmetic takes
about a minute, but the arrays do not fit anywhere convenient.

Three things used to multiply, and two of them were avoidable:

  1. pixel count, (15/2)**2 = 56x going from 15" to 2" pixels -- unavoidable,
     and the honest part of the cost;
  2. everything was float64 in memory, to write f4 to disk -- 2x wasted;
  3. splat_cube ended with `C.T.reshape(L, ny, nx)`, and C.T is not
     contiguous, so the reshape silently copied the whole cube. Measured peak
     RSS / live arrays converged to 2.3x across pixel scales; that extra 1.3x
     was this copy.

This module now projects in float32, in chunks over wavelength, writing each
chunk straight into a preallocated output through a flat view, so no
full-size temporary is ever created. At L = 4000 and 2" pixels that takes
peak memory from about 105 GB to about 25 GB. `chunk_mb` bounds the transient.

A word on whether you want a 2" cube at all: the kernel FWHM is set by the
fiber spacing (~37" for LVM), so a 2" grid stores ~56x more pixels carrying no
more information. Fine sampling pays off when mosaicking many dithered
pointings; for a single exposure, 5-8" already oversamples the beam.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .io_readers import FiberSet
from .geometry import (make_grid_from_points, fit_celestial_wcs, make_wcs3d,
                       pixel_area_arcsec2)
from .kernels import (build_projection_A, splat_cube, ivar_through_A,
                      neff_from_A, coverage_from_A)

try:                                    # keep the two modules agreeing
    from .mapcat_build import LVM_FILL_FACTOR
except ImportError:                     # older mapcat_build without it
    LVM_FILL_FACTOR = (np.pi * (35.3 / 2) ** 2
                       / ((np.sqrt(3) / 2) * 37.0 ** 2))      # 0.825477

# splat_cube / ivar_through_A are imported for callers that still want the
# one-shot versions (and mapcat_build uses them, where L is the number of
# emission lines and chunking would be pointless). build_cube uses
# _project_chunked below instead; see the module docstring.


def _project_chunked(
    flux: np.ndarray,
    ivar: np.ndarray,
    A,
    ny: int,
    nx: int,
    dtype=np.float32,
    chunk_mb: float = 256.0,
    verbose: bool = True,
):
    """
    Forward-project flux and propagate IVAR, in wavelength chunks.

    Same arithmetic as kernels.splat_cube and kernels.ivar_through_A:

        C      = A  @ F          flux, per output pixel
        Var    = A^2 @ Var_in    weights enter SQUARED
        ivar   = 1 / Var

    What differs is only how the memory is handled, and that is the whole
    point of the function:

      * the outputs are allocated once, in `dtype` (float32 by default),
        and written into through a flat (L, P) view. Assigning a transposed
        (c, P) block into a contiguous destination is done element-wise by
        numpy, so no intermediate copy of the block is made -- which is the
        cost that `C.T.reshape(...)` was quietly paying on the whole cube.
      * only one chunk of (P, c) is live at a time, so peak transient memory
        is set by `chunk_mb` rather than by L.

    Parameters
    ----------
    flux, ivar : (K, L)
        Per-fiber spectra, already scrubbed of non-finite values.
    A : (P, K) sparse
        Projection matrix, columns summing to 1.
    ny, nx : int
        Spatial grid shape; ny * nx must equal A.shape[0].
    dtype : numpy dtype
        Working and output precision. float32 halves the memory and is what
        gets written to FITS anyway. A flux-conserving sum over ~30 kernel
        terms carries a relative error near 1e-7 in float32, far below any
        calibration uncertainty. Pass np.float64 to reproduce the old
        behaviour exactly.
    chunk_mb : float
        Target size of ONE temporary, in MB. Peak transient is roughly 3x
        this (the matmul result, the variance block, and its reciprocal).

    Returns
    -------
    cube, ivar_cube : (L, ny, nx) arrays of `dtype`
    """
    K, L = flux.shape
    P = ny * nx
    if A.shape[0] != P:
        raise ValueError(f"A.shape[0] ({A.shape[0]}) must equal ny*nx ({P})")
    if A.shape[1] != K:
        raise ValueError(f"A.shape[1] ({A.shape[1]}) must equal K ({K})")

    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize

    # Chunk width from the memory budget, clamped to something sane.
    chunk = int(max(1, min(L, (chunk_mb * 1e6) // (P * itemsize))))

    # A is kept in its original precision in the result dict so that the
    # diagnostics (column_sum_check uses atol=1e-12) still pass; only these
    # working copies are cast down.
    A_w = A.astype(dtype)
    A2_w = A_w.multiply(A_w).tocsr()

    cube = np.empty((L, ny, nx), dtype)
    ivc = np.empty((L, ny, nx), dtype)
    cube_flat = cube.reshape(L, P)          # free view: cube is contiguous
    ivc_flat = ivc.reshape(L, P)

    if verbose:
        n_chunks = int(np.ceil(L / chunk))
        print(f"[cube] {L} planes in {n_chunks} chunk(s) of {chunk}, "
              f"{dtype.name}: output {2 * cube.nbytes / 1e9:.2f} GB, "
              f"transient ~{3 * P * chunk * itemsize / 1e9:.2f} GB")

    big = np.finfo(dtype).max
    for i0 in range(0, L, chunk):
        sl = slice(i0, min(i0 + chunk, L))

        cube_flat[sl] = (A_w @ flux[:, sl]).T

        # var_in is (K, c) and tiny next to (P, c); inf marks "no information",
        # and clipping keeps a legitimately huge variance from overflowing to
        # inf in float32, which would silently become ivar = 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            iv = ivar[:, sl]
            var_in = np.where(iv > 0, 1.0 / np.where(iv > 0, iv, 1.0), np.inf)
        np.clip(var_in, None, big, out=var_in, where=np.isfinite(var_in))

        Var = A2_w @ var_in                 # (P, c)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.reciprocal(Var, out=Var)     # 1/inf -> 0 ; 1/0 -> inf
        Var[~np.isfinite(Var)] = 0.0        # the 1/0 case: no contributing fiber
        ivc_flat[sl] = Var.T

    return cube, ivc

def build_cube(
    fibers: FiberSet,
    pixel_size_arcsec: float,
    fwhm_arcsec: float,
    fiber_radius_arcsec: float,
    r_sigma_mult: float = 3.0,
    kernel_mode: str = "gauss",
    wcs_strategy: str = "fit", # "fit" or "none"
    max_dropped_frac: float = 0.02,
    dtype=np.float32,
    chunk_mb: float = 256.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build a 3D flux-conserving cube from FiberSet

    Steps
    -----
    1. Build a grid + inside_IFU mask
    2. Build the projection matrix A (pixels x fibers)
    3. Forward project fiber spectra -> cube
    4. Propagate IVAR through A
    5. Compute Neff and coverage maps.
    6. Optionally fit a celestial WCS and embed into 3D.

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
    dtype : numpy dtype
        Working and output precision for the cube and its IVAR. float32 is
        the default and is what gets written to FITS anyway; it halves peak
        memory. Pass np.float64 for the old behaviour.
    chunk_mb : float
        Wavelength chunking budget, in MB per temporary. Lower it on a small
        machine; it does not change the result, only the peak memory.
    verbose : bool
        Print the chunk plan and the resulting array sizes.

    Returns
    -------
    results : dict with keys
        "cube" : (L, ny, nx) flux cube
        "ivar" : same shape, inverse variance cube
        "neff" : (ny, nx) effective number of fibers per pixel
        "coverage" : (ny, nx) summed kernel weight per pixel (row sums of A)
        "A"    : (P, K) projection matrix
        "xg", "yg", "inside" : grid and IFU mask
        "wcs3d": WCS or None
        "wcs2d": the celestial part, or None
        "wave" : wavelength array
        "pixel_size_arcsec" : echoed back for save_cube_to_fits
    """
    wave = np.asarray(fibers.wave, float)
    x = np.asarray(fibers.x_arcsec, float)
    y = np.asarray(fibers.y_arcsec, float)

    # A single NaN fiber flux propagates to every pixel in that fiber's
    # kernel support, so scrub before projecting.
    dtype = np.dtype(dtype)
    flux = np.nan_to_num(np.asarray(fibers.flux, dtype),
                         nan=0.0, posinf=0.0, neginf=0.0)
    ivar = np.nan_to_num(np.asarray(fibers.ivar, dtype),
                         nan=0.0, posinf=0.0, neginf=0.0)

    # 1) Grid + mask.
    # fiber_radius_arcsec was previously not forwarded here, so the grid
    # padding and hull expansion silently used the 18.5" default regardless
    # of what the caller asked for.
    xg, yg, inside = make_grid_from_points(
        x_arcsec=x,
        y_arcsec=y,
        pixel_size_arcsec=pixel_size_arcsec,
        fiber_radius_arcsec=fiber_radius_arcsec,
    )
    ny, nx = inside.shape

    # 2) Projection matrix
    A = build_projection_A(
        x_arcsec=x,
        y_arcsec=y,
        xg=xg,
        yg=yg,
        inside_mask=inside,
        fwhm_arcsec=fwhm_arcsec,
        fiber_radius_arcsec=fiber_radius_arcsec,
        r_sigma_mult=r_sigma_mult,
        mode=kernel_mode,
        max_dropped_frac=max_dropped_frac,
    )

    # 3+4) Forward projection and IVAR propagation, chunked over wavelength.
    # Done together so each chunk of the spectra is touched once; see
    # _project_chunked for why this is not just splat_cube + ivar_through_A.
    cube, ivar_cube = _project_chunked(
        flux, ivar, A, ny=ny, nx=nx,
        dtype=dtype, chunk_mb=chunk_mb, verbose=verbose,
    )

    # 5) Neff + coverage
    neff = neff_from_A(A, ny=ny, nx=nx)
    coverage = coverage_from_A(A, ny=ny, nx=nx)

    # 6) WCS.
    # x0/y0 must be the grid origin -- see geometry.py module docstring.
    wcs2d: Optional[WCS] = None
    wcs3d: Optional[WCS] = None
    if wcs_strategy == "fit" and "ra_deg" in fibers.meta and "dec_deg" in fibers.meta:
        wcs2d = fit_celestial_wcs(
            x_arcsec=x,
            y_arcsec=y,
            ra_deg=np.asarray(fibers.meta["ra_deg"], float),
            dec_deg=np.asarray(fibers.meta["dec_deg"], float),
            pixel_size_arcsec=pixel_size_arcsec,
            x0_arcsec=xg[0],
            y0_arcsec=yg[0],
        )
        wcs3d = make_wcs3d(wcs2d, wave)

    return dict(
        cube=cube,
        ivar=ivar_cube,
        neff=neff,
        coverage=coverage,
        A=A,
        xg=xg,
        yg=yg,
        inside=inside,
        wcs3d=wcs3d,
        wcs2d=wcs2d,
        wave=wave,
        pixel_size_arcsec=float(pixel_size_arcsec),
        flux_in=flux,
    )

def save_cube_to_fits(
    result: Dict[str, Any],
    path: str,
    extra_header: Optional[Dict[str, Any]] = None,
    units: str = "flux_per_pixel",
    inplace: bool = False,
) -> None:
    """
    Write a cube result (from build_cube) to a FITS file

    Layout:
        HDU 0 : PRIMARY (optional metadata)
        HDU 1 : FLUX (L, ny, nx)
        HDU 2 : IVAR (L, ny, nx)
        HDU 3 : NEFF (ny, nx)
        HDU 4 : WAVE (BinTable, column 'WAVE')
        HDU 5 : COVERAGE (ny, nx), if present

    WCS is attached to FLUX (3D) and to the 2D auxiliary images.
    The WAVE table holds the exact wavelength grid; the CD3_3 in the header
    can only describe a linear axis, so trust the table if they disagree.

    Parameters
    ----------
    units : {"flux_per_pixel", "surface_brightness"}
        What to WRITE. "flux_per_pixel" (default) stores the flux-conserving
        quantity the projection produces, and read_cube converts on the way
        back in. "surface_brightness" divides by the pixel solid angle before
        writing, which is the better choice when the file is a catalog
        product that other people (or other pixel scales) will consume: the
        values no longer depend on the grid it happened to be built on. The
        header records the choice in BUNIT and SBCONV, and read_cube refuses
        to convert an already-converted file.
    inplace : bool
        Only consulted when converting. False (default) converts a copy, so
        `result` is left untouched; that copy is the same one the f4 cast
        would make anyway. True divides the caller's arrays in place, which
        MUTATES `result` -- use it when the cube is large enough that a
        second copy is the difference between writing and swapping.
    """
    if units not in ("flux_per_pixel", "surface_brightness"):
        raise ValueError("units must be 'flux_per_pixel' or "
                         f"'surface_brightness', got {units!r}")
    cube = result["cube"]
    ivar = result["ivar"]
    neff = result["neff"]
    wave = np.asarray(result["wave"], float)
    wcs3d: Optional[WCS] = result.get("wcs3d", None)
    wcs2d: Optional[WCS] = result.get("wcs2d", None)
    coverage = result.get("coverage", None)

    if cube.shape != ivar.shape:
        raise ValueError("cube and ivar must have same shape")

    primary_hdu = fits.PrimaryHDU()
    if extra_header:
        for k, v in extra_header.items():
            primary_hdu.header[k] = v

    if wcs3d is not None:
        flux_header = wcs3d.to_header(relax=True)
    else:
        flux_header = fits.Header()

    if wcs2d is not None:
        pix_area = pixel_area_arcsec2(wcs2d)
    elif wcs3d is not None:
        pix_area = pixel_area_arcsec2(wcs3d)
    else:
        ps = result.get("pixel_size_arcsec", None)
        pix_area = float(ps) ** 2 if ps else None

    flux_header["BUNIT"] = ("flux/pixel", "flux-conserving: amount per output pixel")
    if pix_area is not None:
        flux_header["PIXAR_A2"] = (pix_area, "pixel solid angle [arcsec**2]")
    flux_header["FLUXCONS"] = (True, "columns of projection matrix A sum to 1")

    ivar_header = flux_header.copy()
    ivar_header["BUNIT"] = ("1/(flux/pixel)**2", "inverse variance of FLUX")
    aux_hdr = wcs2d.to_header(relax=True) if wcs2d is not None else fits.Header()

    # astype(copy=False) is a no-op when the cube is already float32, which it
    # is by default now. Without copy=False this line duplicated the entire
    # cube a second time, at the worst possible moment.
    # Convert before casting, so the division happens once. astype(copy=False)
    # is a no-op when the cube is already float32 (the default), which is why
    # the copy has to be forced here when we are about to divide in place and
    # the caller did not consent to their array being modified.
    want_sb = units == "surface_brightness"
    if want_sb and pix_area is None:
        raise ValueError(
            "cannot convert to surface brightness: no pixel area is "
            "available. Build with wcs_strategy='fit', or pass "
            "pixel_size_arcsec in the result."
        )

    cube_out = np.asarray(cube).astype("f4", copy=(want_sb and not inplace))
    ivar_out = np.asarray(ivar).astype("f4", copy=(want_sb and not inplace))
    if want_sb:
        cube_out /= np.float32(pix_area)            # flux/pixel -> flux/arcsec**2
        ivar_out *= np.float32(pix_area) ** 2       # Var scales as area**2
        flux_header["BUNIT"] = ("flux/arcsec**2",
                                "surface brightness: FLUX / PIXAR_A2")
        ivar_header["BUNIT"] = ("1/(flux/arcsec**2)**2", "inverse variance of FLUX")
    flux_header["SBCONV"] = (bool(want_sb), "T if already divided by PIXAR_A2")
    ivar_header["SBCONV"] = (bool(want_sb), "T if already multiplied by PIXAR_A2**2")
    flux_header["APERFILL"] = (round(float(LVM_FILL_FACTOR), 4),
                               "fiber fill factor; see read_cube")
    for line in (
        "FLUX is flux per pixel unless SBCONV is T, in which case it has",
        "already been divided by PIXAR_A2 and is a surface brightness.",
        "  surface brightness [flux/arcsec**2] = FLUX / PIXAR_A2",
        "  its inverse variance                = IVAR * PIXAR_A2**2",
        "Fibers cover APERFILL of the sky (35.3in apertures, 37in hex grid);",
        "divide surface brightness by APERFILL for true sky brightness.",
        "ifuCube.cube_build.read_cube() applies both conversions.",
    ):
        flux_header["COMMENT"] = line

    cube_hdu = fits.ImageHDU(cube_out, header=flux_header, name="FLUX")
    ivar_hdu = fits.ImageHDU(ivar_out, header=ivar_header, name="IVAR")
    neff_hdu = fits.ImageHDU(np.asarray(neff, float).astype("f4"),
                             header=aux_hdr, name="NEFF")

    col = fits.Column(name="WAVE", format="E", unit="Angstrom", array=wave.astype("f4"))
    wave_hdu = fits.BinTableHDU.from_columns([col], name="WAVE")

    hdus = [primary_hdu, cube_hdu, ivar_hdu, neff_hdu, wave_hdu]
    if coverage is not None:
        hdus.append(fits.ImageHDU(np.asarray(coverage, float).astype("f4"),
                                  header=aux_hdr, name="COVERAGE"))

    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_cube(
    path: str,
    units: str = "surface_brightness",
    correct_fill_factor: bool = False,
    hdu: str = "FLUX",
    wave_range: Optional[tuple] = None,
    planes: Optional[slice] = None,
    with_ivar: bool = True,
) -> Dict[str, Any]:
    """
    Read a cube written by save_cube_to_fits, converting units on the way in.

    The mirror of mapcat_build.read_map, and it exists for the same reason:
    writing BUNIT tells a reader what the numbers mean, this does the
    conversion for them, from the file's own header rather than from
    remembered constants.

    Parameters
    ----------
    path : str
        Cube FITS file.
    units : {"surface_brightness", "flux_per_pixel"}
        What you want back. "surface_brightness" (the default) is independent
        of the pixel size the cube was built with, so cubes built at
        different scales can be compared voxel by voxel. "flux_per_pixel"
        returns the flux-conserving values; sum voxels for a total flux.

        The conversion is keyed off the file's SBCONV keyword rather than
        off what the caller remembers, so it is idempotent in both
        directions: asking for surface brightness from a file that already
        holds surface brightness hands it back unchanged instead of
        dividing by the pixel area a second time. That double division is
        the mistake you would otherwise make exactly once, months later, on
        a file you had forgotten about.
    correct_fill_factor : bool
        Also divide by APERFILL (~0.8255), so smooth extended emission
        matches the true sky surface brightness. Only valid together with
        units="surface_brightness"; it cancels in line ratios.
    wave_range : (lo, hi), optional
        Wavelengths in Angstrom. ONLY the matching planes are read off disk,
        via a memory-mapped slice, so a 70 GB cube costs whatever the window
        costs. This is usually what you want -- a full LVM cube is ~12,400
        planes and a line window is a few hundred.
    planes : slice, optional
        Plane indices, as an alternative to wave_range. Ignored if
        wave_range is given.
    with_ivar : bool
        Read and convert IVAR too. False halves the memory and the I/O when
        you only want to look at the flux.

    Returns
    -------
    dict with data, ivar, wave, wcs3d, wcs2d, bunit, header, pixel_area,
    plane_slice (what was actually read).
    """
    if units not in ("surface_brightness", "flux_per_pixel"):
        raise ValueError("units must be 'surface_brightness' or "
                         f"'flux_per_pixel', got {units!r}")
    if correct_fill_factor and units != "surface_brightness":
        raise ValueError("correct_fill_factor needs units='surface_brightness'")

    # memmap so that slicing reads only the bytes we ask for
    with fits.open(path, memmap=True) as hdul:
        if hdu not in hdul:
            raise ValueError(f"HDU {hdu!r} not found. Available: "
                             f"{[h.name for h in hdul]}")
        hdr = hdul[hdu].header

        if "WAVE" in hdul:
            wave_all = np.asarray(hdul["WAVE"].data["WAVE"], float)
        else:
            wave_all = np.arange(hdul[hdu].shape[0], dtype=float)

        if wave_range is not None:
            lo, hi = float(wave_range[0]), float(wave_range[1])
            idx = np.nonzero((wave_all >= lo) & (wave_all <= hi))[0]
            if idx.size == 0:
                raise ValueError(
                    f"no planes between {lo} and {hi} Angstrom; the cube "
                    f"spans {wave_all.min():.1f} to {wave_all.max():.1f}")
            sl = slice(int(idx[0]), int(idx[-1]) + 1)
        elif planes is not None:
            sl = planes
        else:
            sl = slice(None)

        data = np.array(hdul[hdu].data[sl], dtype=np.float32)
        wave = wave_all[sl]
        ivar = (np.array(hdul["IVAR"].data[sl], dtype=np.float32)
                if (with_ivar and "IVAR" in hdul) else None)

        wcs3d = WCS(hdr)
        wcs2d = wcs3d.celestial if wcs3d.naxis >= 2 else None

    bunit = str(hdr.get("BUNIT", "flux/pixel")).strip()
    already_sb = bool(hdr.get("SBCONV", False))

    if wcs2d is not None and wcs2d.naxis == 2:
        area = float(hdr.get("PIXAR_A2", pixel_area_arcsec2(wcs2d)))
    else:
        area = float(hdr["PIXAR_A2"]) if "PIXAR_A2" in hdr else None

    if units == "surface_brightness" and not already_sb:
        if area is None:
            raise ValueError("no PIXAR_A2 and no usable WCS, so the pixel "
                             "area is unknown and SB cannot be computed")
        data /= np.float32(area)                   # in place: no second cube
        if ivar is not None:
            ivar *= np.float32(area) ** 2
        bunit = "flux/arcsec**2"
    elif units == "flux_per_pixel" and already_sb:
        if area is None:
            raise ValueError("file is surface brightness but has no "
                             "PIXAR_A2, so it cannot be converted back")
        data *= np.float32(area)
        if ivar is not None:
            ivar /= np.float32(area) ** 2
        bunit = "flux/pixel"
    elif units == "surface_brightness" and already_sb:
        bunit = bunit or "flux/arcsec**2"

    if correct_fill_factor:
        fill = float(hdr.get("APERFILL", LVM_FILL_FACTOR))
        data /= np.float32(fill)
        if ivar is not None:
            ivar *= np.float32(fill) ** 2
        bunit = f"{bunit} (aperture-fill corrected)"

    return dict(data=data, ivar=ivar, wave=wave, wcs3d=wcs3d, wcs2d=wcs2d,
                bunit=bunit, header=hdr, pixel_area=area, plane_slice=sl)