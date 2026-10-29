#!/usr/bin/env python
"""
build.py

Unifying CLI for 
    - DAP multi-line emission maps
    - SFrame cubes
    - Ratio maps derived from map fits outputs

(mosaic with mosaic.py)

python -m ifuCube.build dap-maps   --dap dap/dap-rsp108-sn20-00004163.dap.fits.gz   --fibermap /home/rwhite/lvmcore/metrology/lvm_fiducial_fibermap.yaml   --out lvmDAPMaps-00004163.fits   --lines Halpha Hbeta "[NII]6583" "[OIII]5007" "[SII]6731"   --pix 2.0 --fwhm 3.2 --frad 18.5 --rsigma 3.0 --kernel gauss

Examples:

    # Build DAP maps
    python -m ifuCube.build dap-maps \
        --dap dap/dap-rsp108-sn20-00006112.dap.fits.gz \
        --fibermap /home/rwhite/lvmcore/metrology/lvm_fiducial_fibermap.yaml \
        --out lvmDAPMaps-00004297.fits \
        --lines Halpha [NII]6583 [SII]6716 [SII]6731 \
        --pix 2.0 --fwhm 3.2 --frad 18.5 --rsigma 3 --kernel gauss

    python -m ifuCube.build dap-maps \
        --dap ~/sas/sdsswork/lvm/spectro/analysis/1.2.1/0011XX/11111/60222/00006117/dap-rsp108-sn20-00006117.dap.fits.gz \
        --fibermap /home/rwhite/lvmcore/metrology/lvm_fiducial_fibermap.yaml \
        --out lvmDAPMaps-00006117.fits \
        --lines Halpha [NII]6583 [OIII]5007 \
        --pix 2.0 --fwhm 3.2 --frad 18.5 --rsigma 3 --kernel gauss
    
    # Build cube from SFrame
    python -m ifuCube.build cube \
        --sframe /home/rwhite/sas/sdsswork/lvm/spectro/redux/1.3.2/0011XX/11111/60191/lvmSFrame-00004297.fits \
        --fibermap /home/rwhite/lvmcore/metrology/lvm_fiducial_fibermap.yaml \
        --out lvmCubeClean-00004297.fits \
        --pix 15.0 --fwhm 35 --frad 18.5 --rsigma 3 --kernel gauss

    # Build ratio map from a multi-line MAP FITS (output of dap-maps)
    python build.py ratio \
        --in lvmDAPMaps-00004297.fits \
        --num "[NII]6583" \
        --den "Halpha" \
        --out ratio_NII_HA-00004297.fits


"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Sequence

from astropy.io import fits

from ifuCube.io_lvm import read_dap_multi, read_lvm_sframe, LINE_DEFS
from ifuCube.mapcat_build import build_map, save_map_to_fits
from ifuCube.cube_build import build_cube, save_cube_to_fits
from ifuCube.ratio import build_ratio_from_map_fits


# Helpers 
def _ensure_parent(path:Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

# Sub commands

def cmd_dap_maps(args: argparse.Namespace) -> None:
    dap_path = Path(args.dap)
    fibermap_yaml = Path(args.fibermap) if args.fibermap else None
    out_path = Path(args.out)
    _ensure_parent(out_path)

    fibers = read_dap_multi(
        dap_path=dap_path,
        fibermap_yaml=fibermap_yaml,
        lines=args.lines,
        extra_cols=args.extra_cols,
        kinematics=not args.no_kinematics,
        slitmap_path=args.slitmap,
        mask_source=args.mask_source,
        assume_all_science={"auto": "auto", "yes": True,
                            "no": False}[args.assume_all_science],
    )

    result = build_map(
        fibers=fibers,
        pixel_size_arcsec=args.pix,
        fwhm_arcsec=args.fwhm,
        fiber_radius_arcsec=args.frad,
        r_sigma_mult=args.rsigma,
        kernel_mode=args.kernel,
        wcs_strategy="fit",
    )

    save_map_to_fits(
        result=result,
        path=str(out_path),
        extra_header={
            "PIXSIZE": (args.pix, "arcsec per spaxel"),
            "FWHM_ARC": (args.fwhm, "kernel FWHM arcsec"),
            "FRAD_ARC": (args.frad, "fibre radius arcsec"),
            "RSMULT": (args.rsigma, "sigma truncation mult"),
            "KERNEL": (args.kernel, "kernel mode"),
        },
    )
    print(f" Emission maps written to {out_path}")

def cmd_ratio(args) -> None:
    """
    Build ratio FITS from an existing multi-line map FITS.
    """
    in_map = Path(args.in_map)
    out_ratio = Path(args.out)

    build_ratio_from_map_fits(
        map_fits=in_map,
        numerator=args.num,
        denominator=args.den,
        out_fits=out_ratio,
        map_hdu=args.map_hdu,
        ivar_hdu=args.ivar_hdu,
        neff_hdu=args.neff_hdu,
        lines_hdu=args.lines_hdu,
        min_den=args.min_den,
    )
    print(f"[✓] Ratio map written to {out_ratio}  ({args.num}/{args.den})")

def cmd_cube(args: argparse.Namespace) -> None:
    sframe_path = Path(args.sframe)
    fibermap_yaml = Path(args.fibermap) if args.fibermap else None
    out_path = Path(args.out)
    _ensure_parent(out_path)

    fibers = read_lvm_sframe(
        sframe_path=sframe_path,
        fibermap_yaml=fibermap_yaml,
        mask_source=args.mask_source,
    )

    result = build_cube(
        fibers=fibers,
        pixel_size_arcsec=args.pix,
        fwhm_arcsec=args.fwhm,
        fiber_radius_arcsec=args.frad,
        r_sigma_mult=args.rsigma,
        kernel_mode=args.kernel,
        wcs_strategy="fit",
    )

    save_cube_to_fits(
        result=result,
        path=str(out_path),
        extra_header={
            "PIXSIZE": (args.pix, "arcsec per spaxel"),
            "FWHM_ARC": (args.fwhm, "kernel FWHM arcsec"),
            "FRAD_ARC": (args.frad, "fibre radius arcsec"),
            "RSMULT": (args.rsigma, "sigma truncation mult"),
            "KERNEL": (args.kernel, "kernel mode"),
        },
    )
    print(f" Cube written to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build IFU cubes/maps/ratios.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- dap-maps ----
    p_map = sub.add_parser("dap-maps", help="Build multi-line DAP emission maps")
    p_map.add_argument("--dap", required=True, help="DAP file path")
    p_map.add_argument("--fibermap", default=None,
                       help="fiducial fibermap YAML. OPTIONAL and a last resort: "
                            "targettype/fibstatus are preferred from the DAP table "
                            "or --slitmap, which describe this exposure")
    p_map.add_argument("--slitmap", default=None,
                       help="FITS file with a SLITMAP extension (normally the sframe "
                            "this DAP file came from); used if the DAP table itself "
                            "has no targettype/fibstatus")
    p_map.add_argument("--assume-all-science", dest="assume_all_science",
                       choices=["auto", "yes", "no"], default="auto",
                       help="what to do when no source has targettype (e.g. a DAP PT "
                            "table). 'auto' (default) decides from the row count: a "
                            "table larger than the science-fibre complement must still "
                            "hold sky fibres and is refused. 'yes' forces, 'no' always "
                            "raises")
    p_map.add_argument("--mask-source", dest="mask_source", default="auto",
                       choices=["auto", "table", "slitmap", "yaml"],
                       help="pin the mask to one source instead of the auto chain")
    p_map.add_argument("--out", required=True, help="Output FITS map filename")

    p_map.add_argument("--lines", nargs="+", default=["Halpha"],
                       help=f"Logical line names; choose from {sorted(LINE_DEFS)}")
    p_map.add_argument("--extra-cols", nargs="+", default=None,
                       help='Extra DAP columns for the FIBERCAT table, names or wildcards, e.g. "vel_*". '
                            'For non-per-line quantities; each requested line already brings its own kinematics.')
    p_map.add_argument("--no-kinematics", action="store_true",
                       help="Do not add each line's velocity/dispersion (and errors) to FIBERCAT. "
                            "By default they are derived from the line's flux column stem "
                            "(flux_Halpha_6562.85 -> vel_Halpha_6562.85, ...) and any that the "
                            "DAP file does not have are reported and skipped.")
    p_map.add_argument("--pix", type=float, default=2.0, help="Spaxel size in arcsec")
    p_map.add_argument("--fwhm", type=float, default=3.2, help="Gaussian kernel FWHM in arcsec (set from fiber spacing ~35, not from --pix)")
    p_map.add_argument("--frad", type=float, default=18.5, help="Fibre radius in arcsec")
    p_map.add_argument("--rsigma", type=float, default=3.0, help="Sigma multiplier for kernel truncation")
    p_map.add_argument("--kernel", choices=["gauss", "tophat"], default="gauss")

    p_map.set_defaults(func=cmd_dap_maps)

    # ---- cube ----
    p_cube = sub.add_parser("cube", help="Build a cube from an LVM SFrame")
    p_cube.add_argument("--sframe", required=True, help="SFrame FITS path")
    p_cube.add_argument("--fibermap", default=None,
                        help="fiducial fibermap YAML. OPTIONAL: the sframe's own "
                             "SLITMAP normally carries targettype/fibstatus and is "
                             "preferred, since it describes this exposure")
    p_cube.add_argument("--mask-source", dest="mask_source", default="auto",
                        choices=["auto", "table", "yaml"],
                        help="pin the mask to one source ('table' = the SLITMAP)")
    p_cube.add_argument("--out", required=True, help="Output FITS cube filename")

    p_cube.add_argument("--pix", type=float, default=2.0, help="Spaxel size in arcsec")
    p_cube.add_argument("--fwhm", type=float, default=3.2, help="Gaussian kernel FWHM in arcsec (set from fiber spacing ~35, not from --pix)")
    p_cube.add_argument("--frad", type=float, default=18.5, help="Fibre radius in arcsec")
    p_cube.add_argument("--rsigma", type=float, default=3.0, help="Sigma multiplier for kernel truncation")
    p_cube.add_argument("--kernel", choices=["gauss", "tophat"], default="gauss")

    p_cube.set_defaults(func=cmd_cube)

    # ---- ratio ----
    p_ratio = sub.add_parser("ratio", help="Build a ratio FITS from an existing multi-line map FITS")
    p_ratio.add_argument("--in-map", required=True, help="Input multi-line map FITS (from save_map_to_fits)")
    p_ratio.add_argument("--out", required=True, help="Output ratio FITS filename")
    p_ratio.add_argument("--num", required=True, help="Numerator line name (must match LINES.LINE_NAME)")
    p_ratio.add_argument("--den", required=True, help="Denominator line name (must match LINES.LINE_NAME)")
    p_ratio.add_argument("--min-den", type=float, default=0.0,
                         help="Require |den| > min_den to form ratio")

    # allow HDU name overrides (defaults match your save_map_to_fits layout)
    p_ratio.add_argument("--map-hdu", default="FLUX", help="HDU for the line maps (default: MAP)")
    p_ratio.add_argument("--ivar-hdu", default="IVAR", help="HDU for IVAR maps (default: IVAR)")
    p_ratio.add_argument("--neff-hdu", default="NEFF", help="HDU for NEFF map (default: NEFF)")
    p_ratio.add_argument("--lines-hdu", default="LINES", help="HDU for line metadata table (default: LINES)")

    p_ratio.set_defaults(func=cmd_ratio)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
