#!/usr/bin/env python

"""
mosaic.py

Mosaics 3D cubes or Emission line maps, 
Can use ratio.py to create ratio mosaics 

using reproject.mosaicking.find_optimal_celestial_wcs to build an output WCS
which encompasses all of the passed cubes/maps
- accomplished with --auto-wcs in CLI 

use --flux-per-pixel for these maps as they are integrated fluxes 

example run

# Mosaic Halpha maps (FLUX HDU) using IVAR weights, conserve integrated flux:
python -m ifuCube.mosaic \
  --inputs lvmDAPMaps-00006112.fits lvmDAPMaps-00006113.fits lvmDAPMaps-00006114.fits lvmDAPMaps-00006115.fits lvmDAPMaps-00006116.fits lvmDAPMaps-00006117.fits \
  --hdu FLUX --weights-hdu IVAR \
  --output mosaic_rosette.fits \
  --mode cube \
  --auto-wcs \
  --combine mean \
  --reproject exact \
  --flux-per-pixel

# Mosaic cubes (FLUX HDU is 3D), weighted by IVAR, conserve integrated flux:
python -m ifuCube.mosaic \
  --inputs lvmCubeClean-00006657.fits lvmCubeClean-00020501.fits lvmCubeClean-00006588.fits lvmCubeClean-00020509.fits \
  --hdu FLUX --weights-hdu IVAR \
  --output mosaic_cube.fits \
  --mode cube \
  --auto-wcs
  --combine mean \
  --reproject exact \
  --flux-per-pixel

When mosaicing multi-line maps, use mode cube
When mosaicing ratio maps do NOT use flux-per-pixel (dimensionless)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

from typing import Optional, Dict, Any, List, Tuple, Callable

def _read_hdu(path: Path, hdu:str | int):
    hdul = fits.open(path, memmap=False)
    try:
        data = hdul[hdu].data
        header = hdul[hdu].header
    finally:
        hdul.close()
    return np.asarray(data), header 

def _get_wcs_and_shape(template_path: Path, template_hdu: str | int):
    data, hdr = _read_hdu(template_path, template_hdu)
    wcs = WCS(hdr)
    shape = data.shape
    return wcs, shape, hdr

def _pixel_area_sr(wcs: WCS) -> float:
    """
    Solid angle per pixel in steradians (sr) for a celestial WCS.
    """

    # proj_plane_pixel_area returns deg^2 per pixel for celestial WCS
    area_deg2 = proj_plane_pixel_area(wcs.celestial)
    return float(area_deg2) * (np.pi / 180.0) ** 2 

def _combine_function(name: str) -> Callable:
    """
    Map CLI combine choices to a callable suitable for reproject_and_coadd.
    """
    name = name.lower()
    if name == "mean":
        return np.nanmean
    if name == "sum":
        return np.nansum
    if name == "min":
        return np.nanmin
    if name == "max":
        return np.nanmax

    # "first" and "last" are not standard reduce funcs; implement explicitly.
    if name == "first":
        def _first(arr, axis=0):
            # pick first along axis, preserving shape
            return np.take(arr, 0, axis=axis)
        return _first

    if name == "last":
        def _last(arr, axis=0):
            return np.take(arr, -1, axis=axis)
        return _last

    raise ValueError(f"Unknown combine='{name}'")

def _get_reproject_function(mode: str):
    try:
        from reproject import reproject_exact, reproject_interp
    except Exception as e:
        raise RuntimeError(
            "This script expects the 'reproject' package.\n"
            "Install with: pip install reproject\n"
            f"Import error: {e}"
        )

    mode = mode.lower()
    if mode == "exact":
        return reproject_exact
    if mode == "interp":
        return reproject_interp
    raise ValueError("reproject must be 'exact' or 'interp'")

def _find_output_wcs_and_shape(
    input_paths: List[str],
    hdu_in: str | int,
    template: Optional[Path],
    auto_wcs: bool,
    pixel_scale_arcsec: Optional[float] = None,
) -> tuple[WCS, tuple[int, int], fits.Header]:
    """
    Decide output celestial WCS + (ny,nx) shape.

    - If auto_wcs: use find_optimal_celestial_wcs across all inputs
    - Else: use template header/shape

    Returns (out_wcs_celestial, shape_out_xy, header_template_like)
    """
    if auto_wcs:
        try:
            from reproject.mosaicking import find_optimal_celestial_wcs
        except Exception as e:
            raise RuntimeError(
                "Need reproject.mosaicking.find_optimal_celestial_wcs.\n"
                f"Import error: {e}"
            )

        # Build list of (array, wcs) for find_optimal_celestial_wcs
        # Use a dummy array of the right shape for each input so footprint is correct.
        inputs_for_wcs = []
        for p in input_paths:
            data, hdr = _read_hdu(Path(p), hdu_in)
            w = WCS(hdr).celestial
            shape_xy = data.shape[-2:] if data.ndim == 3 else data.shape

            # only the shape is read; no need to zero-fill it
            dummy = np.empty(shape_xy, dtype="f4")
            inputs_for_wcs.append((dummy, w))

        # Optional: force pixel scale (arcsec/pix) if requested
        if pixel_scale_arcsec is not None:
            resolution = (pixel_scale_arcsec / 3600.0)  # deg/pixel
            out_wcs, shape_out = find_optimal_celestial_wcs(
                inputs_for_wcs,
                resolution=resolution
            )
        else:
            out_wcs, shape_out = find_optimal_celestial_wcs(inputs_for_wcs)

        hdr_out = out_wcs.to_header(relax=True)
        return out_wcs, tuple(shape_out), hdr_out

    # Template mode
    if template is None:
        raise ValueError("template must be provided unless --auto-wcs is set")

    tdata, thdr = _read_hdu(template, hdu_in)
    out_wcs = WCS(thdr).celestial
    shape_xy = tdata.shape[-2:] if tdata.ndim == 3 else tdata.shape
    return out_wcs, tuple(shape_xy), thdr


# -----------------------------------------------------------------------------
# Core mosaic routines
# -----------------------------------------------------------------------------

def _coadd_map_with_reproject(
    inputs: List[str] | List[Tuple[np.ndarray, WCS]],
    output_wcs: WCS,
    shape_out: tuple[int, int],
    input_weights: Optional[List[np.ndarray]] = None,
    hdu_in: str | int = "FLUX",
    hdu_weights: Optional[str | int] = "IVAR",
    reproject_mode: str = "exact",
    combine: str = "mean",
    flux_per_pixel: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproject + coadd 2D maps.
    inputs can be:
      - list of file paths
      - list of (array2d, WCS) tuples (used internally for cube planes)
    """
    try:
        from reproject.mosaicking import reproject_and_coadd
    except Exception as e:
        raise RuntimeError(
            "This script expects reproject.\n"
            "Install with: pip install reproject\n"
            f"Import error: {e}"
        )

    reproject_fn = _get_reproject_function(reproject_mode)
    # _combine_function maps the name to a real callable, including the
    # custom 'first'/'last' reducers that reproject does not know by name.
    combine_fn = _combine_function(combine)

    # Normalize inputs into list[(array, wcs)]
    arr_wcs: List[Tuple[np.ndarray, WCS]] = []
    tuple_inputs = False

    for it in inputs:
        if isinstance(it, (tuple, list)) and len(it) == 2:
            tuple_inputs = True
            data, w = it
            arr_wcs.append((np.asarray(data, float), w.celestial if hasattr(w, "celestial") else w))
        else:
            data, hdr = _read_hdu(Path(it), hdu_in)
            data = np.asarray(data, float)
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            w = WCS(hdr).celestial
            arr_wcs.append((np.asarray(data, float), w))

    # Prepare weights for tuple-input case
    w_list = None
    if tuple_inputs:
        if input_weights is not None:
            if len(input_weights) != len(arr_wcs):
                raise ValueError("input_weights length must match inputs length")
            w_list = [np.asarray(wi, float) for wi in input_weights]
    else:
        # file-input case: let reproject_and_coadd read weights via hdu_weights
        w_list = None

    # Flux-per-pixel conservation: convert to SB using pixel solid angle.
    if flux_per_pixel:
        out_area = _pixel_area_sr(output_wcs)

        sb_inputs: List[Tuple[np.ndarray, WCS]] = []
        sb_weights = [] if (tuple_inputs or (hdu_weights is not None)) else None

        for i, (data, w) in enumerate(arr_wcs):
            in_area = _pixel_area_sr(w)
            sb = data / in_area
            sb_inputs.append((sb, w))

            if sb_weights is not None:
                if tuple_inputs:
                    iv = w_list[i] if w_list is not None else None
                    if iv is None:
                        raise ValueError("tuple inputs + flux_per_pixel requires input_weights")
                else:
                    iv = fits.getdata(Path(inputs[i]), hdu_weights)
                    iv = np.asarray(iv, float)
                    if iv.ndim == 3 and iv.shape[0] == 1:
                        iv = iv[0]
                # ivar transforms: ivar_sb = ivar_flux * area^2
                sb_weights.append(np.asarray(iv, float) * (in_area ** 2))

        sb_mosaic, footprint = reproject_and_coadd(
            sb_inputs,
            output_projection=output_wcs,
            shape_out=shape_out,
            input_weights=sb_weights,
            reproject_function=reproject_fn,
            combine_function=combine_fn,
        )

        flux_mosaic = sb_mosaic * out_area
        return flux_mosaic, footprint

    # Non flux-per-pixel path
    if tuple_inputs:
        mosaic, footprint = reproject_and_coadd(
            arr_wcs,
            output_projection=output_wcs,
            shape_out=shape_out,
            input_weights=w_list,
            reproject_function=reproject_fn,
            combine_function=combine_fn,
        )
        return mosaic, footprint

    # File-input path
    mosaic, footprint = reproject_and_coadd(
        [str(p) for p in inputs],
        output_projection=output_wcs,
        shape_out=shape_out,
        input_weights=None,
        hdu_in=hdu_in,
        hdu_weights=hdu_weights if hdu_weights is not None else None,
        reproject_function=reproject_fn,
        combine_function=combine_fn,
    )
    return mosaic, footprint


def _coadd_cube_plane_by_plane(
    input_paths: List[str],
    output_wcs_celestial: WCS,
    shape_out_xy: tuple[int, int],
    hdu_in: str | int = "FLUX",
    weights_hdu: Optional[str | int] = "IVAR",
    reproject_mode: str = "exact",
    combine: str = "mean",
    flux_per_pixel: bool = True,
) -> np.ndarray:
    """
    Mosaic a 3D cube by looping over wavelength index and mosaicing each 2D plane.
    Assumes all cubes share the same spectral axis sampling (same L).
    Output shape: (L, ny, nx)
    """
    tdata, _thdr = _read_hdu(Path(input_paths[0]), hdu_in)
    if tdata.ndim != 3:
        raise ValueError("Expected a 3D cube in selected HDU")

    L = tdata.shape[0]
    ny, nx = shape_out_xy
    out = np.full((L, ny, nx), np.nan, dtype=float)

    for ell in range(L):
        plane_inputs = []
        plane_weights = [] if weights_hdu is not None else None

        for p in input_paths:
            data, hdr = _read_hdu(Path(p), hdu_in)
            w = WCS(hdr).celestial
            plane_inputs.append((np.asarray(data[ell], float), w))

            if weights_hdu is not None:
                wdat = fits.getdata(p, weights_hdu)
                plane_weights.append(np.asarray(wdat[ell], float))

        m, _fp = _coadd_map_with_reproject(
            plane_inputs,
            output_wcs=output_wcs_celestial,
            shape_out=(ny, nx),
            input_weights=plane_weights,
            hdu_in=0,
            hdu_weights=0,
            reproject_mode=reproject_mode,
            combine=combine,
            flux_per_pixel=flux_per_pixel,
        )
        out[ell] = m

    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input FITS files (tiles)")
    ap.add_argument("--output", required=True, help="Output FITS path")
    ap.add_argument("--hdu", default="FLUX", help="Input data HDU name (e.g. FLUX)")
    ap.add_argument("--all-planes", action="store_true",
                help="If input maps are 3D (N,ny,nx), mosaic all planes (e.g. multiple emission lines).")
    ap.add_argument("--plane", type=int, default=0,
                help="If input maps are 3D (N,ny,nx) and --all-planes is not set, mosaic this plane index.")
    ap.add_argument("--weights-hdu", default="IVAR", help="Weights HDU name (e.g. IVAR) or set to 'NONE'")
    ap.add_argument("--mode", choices=["auto", "map", "cube"], default="auto")
    ap.add_argument("--combine", choices=["mean", "sum", "first", "last", "min", "max"], default="mean")
    ap.add_argument("--reproject", choices=["exact", "interp"], default="exact")
    ap.add_argument("--flux-per-pixel", action="store_true",
                    help="Treat inputs as integrated flux per pixel; conserve total flux via area conversion")

    # Output grid control
    ap.add_argument("--template", default=None,
                    help="Template FITS to define output WCS/shape (only used if --auto-wcs is NOT set)")
    ap.add_argument("--auto-wcs", action="store_true",
                    help="Compute output WCS/shape that encloses all inputs (recommended for mosaicing tiles)")
    ap.add_argument("--pixscale", type=float, default=None,
                    help="Optional output pixel scale in arcsec/pixel (only used with --auto-wcs)")

    # Output formatting
    ap.add_argument("--squeeze", action="store_true",
                    help="If mosaic has a singleton leading axis (1,ny,nx), write out 2D (ny,nx)")

    args = ap.parse_args()

    inputs = [str(Path(p)) for p in args.inputs]
    outpath = Path(args.output)
    template = Path(args.template) if args.template else None

    weights_hdu = None if (args.weights_hdu is None or str(args.weights_hdu).upper() == "NONE") else args.weights_hdu

    # Determine mode if auto
    sample_data, _sample_hdr = _read_hdu(Path(inputs[0]), args.hdu)
    if args.mode == "auto":
        mode = "cube" if sample_data.ndim == 3 else "map"
    else:
        mode = args.mode

    # Build output celestial WCS + shape that encloses all inputs (or use template)
    out_wcs_cel, shape_out_xy, hdr_out = _find_output_wcs_and_shape(
        input_paths=inputs,
        hdu_in=args.hdu,
        template=template,
        auto_wcs=args.auto_wcs,
        pixel_scale_arcsec=args.pixscale,
    )

    if mode == "map":
        sample_data, sample_hdr = _read_hdu(Path(inputs[0]), args.hdu)

        # Case A: 2D maps -> existing behavior
        if sample_data.ndim == 2 or (sample_data.ndim == 3 and sample_data.shape[0] == 1 and not args.all_planes):
            mosaic, footprint = _coadd_map_with_reproject(
                inputs,
                output_wcs=out_wcs_cel,
                shape_out=shape_out_xy,
                hdu_in=args.hdu,
                hdu_weights=weights_hdu,
                reproject_mode=args.reproject,
                combine=args.combine,
                flux_per_pixel=args.flux_per_pixel,
            )
            hdr_wcs = out_wcs_cel.to_header(relax=True)
            hdul = fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(np.asarray(mosaic, float).astype("f4"), header=hdr_wcs, name="MOSAIC"),
                fits.ImageHDU(np.asarray(footprint, float).astype("f4"), header=hdr_wcs, name="FOOTPRINT"),
            ])
            hdul.writeto(outpath, overwrite=True)
            print(f"[ok] Wrote map mosaic: {outpath}")
            return

        # Case B: multi-plane maps (N,ny,nx)
        if sample_data.ndim == 3:
            nplanes = sample_data.shape[0]

            if args.all_planes:
                # mosaic every plane
                ny, nx = shape_out_xy
                mos = np.full((nplanes, ny, nx), np.nan, dtype=float)
                fp  = np.zeros((nplanes, ny, nx), dtype=float)

                for k in range(nplanes):
                    plane_inputs = []
                    plane_weights = [] if weights_hdu is not None else None

                    for p in inputs:
                        d, hdr = _read_hdu(Path(p), args.hdu)
                        w = WCS(hdr).celestial

                        # squeeze singleton maps, otherwise take plane k
                        if d.ndim == 3 and d.shape[0] == 1:
                            d2 = np.asarray(d[0], float)
                        else:
                            d2 = np.asarray(d[k], float)

                        plane_inputs.append((d2, w))

                        if weights_hdu is not None:
                            iv = fits.getdata(p, weights_hdu)
                            iv = np.asarray(iv, float)
                            if iv.ndim == 3 and iv.shape[0] == 1:
                                iv2 = iv[0]
                            else:
                                iv2 = iv[k]
                            plane_weights.append(iv2)

                    mk, fpk = _coadd_map_with_reproject(
                        plane_inputs,
                        output_wcs=out_wcs_cel,
                        shape_out=shape_out_xy,
                        input_weights=plane_weights,
                        hdu_in=0,
                        hdu_weights=0,
                        reproject_mode=args.reproject,
                        combine=args.combine,
                        flux_per_pixel=args.flux_per_pixel,
                    )
                    mos[k] = mk
                    fp[k] = fpk

                # Write a 3D mosaic for multi-line maps
                hdr_wcs = out_wcs_cel.to_header(relax=True)
                hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(mos.astype("f4"), header=hdr_wcs, name="MOSAIC"),
                    fits.ImageHDU(fp.astype("f4"), header=hdr_wcs, name="FOOTPRINT"),
                ])
                hdul.writeto(outpath, overwrite=True)
                print(f"[ok] Wrote multi-plane map mosaic: {outpath}")
                return

            else:
                # mosaic a single requested plane
                k = args.plane
                if not (0 <= k < nplanes):
                    raise ValueError(f"--plane {k} out of range for input with {nplanes} planes")

                plane_inputs = []
                plane_weights = [] if weights_hdu is not None else None

                for p in inputs:
                    d, hdr = _read_hdu(Path(p), args.hdu)
                    w = WCS(hdr).celestial
                    d2 = np.asarray(d[k], float) if d.ndim == 3 else np.asarray(d, float)
                    plane_inputs.append((d2, w))

                    if weights_hdu is not None:
                        iv = np.asarray(fits.getdata(p, weights_hdu), float)
                        iv2 = iv[k] if iv.ndim == 3 else iv
                        plane_weights.append(iv2)

                mosaic, footprint = _coadd_map_with_reproject(
                    plane_inputs,
                    output_wcs=out_wcs_cel,
                    shape_out=shape_out_xy,
                    input_weights=plane_weights,
                    hdu_in=0,
                    hdu_weights=0,
                    reproject_mode=args.reproject,
                    combine=args.combine,
                    flux_per_pixel=args.flux_per_pixel,
                )
                hdr_wcs = out_wcs_cel.to_header(relax=True)
                hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(np.asarray(mosaic, float).astype("f4"), header=hdr_wcs, name="MOSAIC"),
                    fits.ImageHDU(np.asarray(footprint, float).astype("f4"), header=hdr_wcs, name="FOOTPRINT"),
                ])
                hdul.writeto(outpath, overwrite=True)
                print(f"[ok] Wrote map mosaic: {outpath}")
                return

        raise ValueError(f"Unexpected map dimensionality: {sample_data.shape}")


    elif mode == "cube":
        # For cubes we only use celestial WCS for planes; spectral axis is assumed aligned across inputs.
        # Output header will be celestial WCS; you can optionally extend this later if you want full 3D WCS.
        cube_mosaic = _coadd_cube_plane_by_plane(
            inputs,
            output_wcs_celestial=out_wcs_cel,
            shape_out_xy=shape_out_xy,
            hdu_in=args.hdu,
            weights_hdu=weights_hdu,
            reproject_mode=args.reproject,
            combine=args.combine,
            flux_per_pixel=args.flux_per_pixel,
        )

        hdr_wcs = out_wcs_cel.to_header(relax=True)
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(cube_mosaic.astype("f4"), header=hdr_wcs, name="MOSAIC"),
        ])
        hdul.writeto(outpath, overwrite=True)
        print(f"[ok] Wrote cube mosaic: {outpath}")

    else:
        raise ValueError("mode must be map or cube")


if __name__ == "__main__":
    main()

