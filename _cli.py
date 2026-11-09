#!/usr/bin/env python

"""
_cli.py
-------
Argument groups shared by the cube_build and mapcat_build command lines.

Both builders take the same grid and kernel parameters, and both can
optionally fill dead fibres first. Declaring those flags twice would put the
two CLIs out of step the first time a default changed -- the same failure
that put LVM_FILL_FACTOR in two files -- so they are declared once here.

Internal: the leading underscore marks this as package plumbing, not API.
Nothing but the CLIs should import it.
"""

from __future__ import annotations
from typing import Any, Dict

# LVM defaults. 9" pixels give ~4 per 35" FWHM, which is comfortably Nyquist;
# the FWHM is set from the FIBRE SPACING (37"), not the pixel size.
DEFAULT_PIXEL_ARCSEC = 9.0
DEFAULT_FWHM_ARCSEC = 35.0
DEFAULT_FIBER_RADIUS_ARCSEC = 17.65


def add_grid_args(ap) -> None:
    """Grid and kernel geometry, forwarded straight to the builder."""
    g = ap.add_argument_group("grid and kernel")
    g.add_argument("--pixel", type=float, default=DEFAULT_PIXEL_ARCSEC,
                   metavar="ARCSEC",
                   help="output pixel size in arcsec (default: %(default)s). "
                        "File size goes as the SQUARE of this, so 2\" is ~20x "
                        "the bytes of 9\" for no extra information at LVM's "
                        "~35\" resolution.")
    g.add_argument("--fwhm", type=float, default=DEFAULT_FWHM_ARCSEC,
                   metavar="ARCSEC",
                   help="kernel FWHM in arcsec (default: %(default)s). Set "
                        "this from the FIBRE SPACING (~37\"), not the pixel "
                        "size: a kernel narrower than the spacing leaves "
                        "holes between fibres however fine the grid.")
    g.add_argument("--fiber-radius", type=float,
                   default=DEFAULT_FIBER_RADIUS_ARCSEC, metavar="ARCSEC",
                   help="fibre radius in arcsec (default: %(default)s); sets "
                        "grid padding and hull expansion")
    g.add_argument("--kernel", choices=["gauss", "tophat"], default="gauss",
                   help="kernel shape (default: %(default)s)")
    g.add_argument("--r-sigma-mult", type=float, default=3.0, metavar="N",
                   help="kernel truncation radius in sigma (default: "
                        "%(default)s)")
    g.add_argument("--max-dropped-frac", type=float, default=0.02,
                   metavar="FRAC",
                   help="refuse to build if more than this fraction of fibres "
                        "would have their flux discarded (default: "
                        "%(default)s). Guards against a silent flux leak from "
                        "a mis-specified grid.")


def add_fill_args(ap) -> None:
    """Optional imputation of dead fibres, applied before building."""
    g = ap.add_argument_group("hole filling (optional)")
    g.add_argument("--fill", action="store_true",
                   help="impute dead fibres from their neighbours before "
                        "building. FOR DISPLAY: it adds no information. "
                        "Ratios need no filling (a dead fibre is dead in "
                        "every line, so the deficit cancels), and aperture "
                        "photometry is better fixed by dividing by COVERAGE. "
                        "Imputed fibres are flagged in FIBERCAT.")
    g.add_argument("--fill-method",
                   choices=["idw", "plane", "mean", "nearest"], default="idw",
                   help="imputation method (default: %(default)s)")
    g.add_argument("--fill-k", type=int, default=6, metavar="N",
                   help="neighbours per estimate (default: %(default)s)")
    g.add_argument("--max-gap-spacings", type=float, default=1.35,
                   metavar="N",
                   help="refuse to impute a fibre whose neighbours average "
                        "further than this, in fibre spacings (default: "
                        "%(default)s). Excludes the interiors of contiguous "
                        "dead patches, where interpolation is a guess.")


def add_common_args(ap) -> None:
    """Flags every builder CLI has."""
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress progress output")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="fail instead of replacing an existing output file")


def grid_kwargs(args) -> Dict[str, Any]:
    """The subset of `args` that build_map and build_cube both accept."""
    return dict(
        pixel_size_arcsec=args.pixel,
        fwhm_arcsec=args.fwhm,
        fiber_radius_arcsec=args.fiber_radius,
        kernel_mode=args.kernel,
        r_sigma_mult=args.r_sigma_mult,
        max_dropped_frac=args.max_dropped_frac,
    )


def maybe_fill(fibers, args, verbose: bool = True):
    """Apply fill_missing_fibers if --fill was given; otherwise pass through.

    Imported lazily so that a plain build does not pay for scipy's cKDTree
    import, and so a broken fill.py cannot stop someone building a map.
    """
    if not getattr(args, "fill", False):
        return fibers
    from .fill import fill_missing_fibers
    return fill_missing_fibers(
        fibers,
        method=args.fill_method,
        k=args.fill_k,
        max_gap_spacings=args.max_gap_spacings,
        verbose=verbose,
    )


__all__ = ["add_grid_args", "add_fill_args", "add_common_args",
           "grid_kwargs", "maybe_fill",
           "DEFAULT_PIXEL_ARCSEC", "DEFAULT_FWHM_ARCSEC",
           "DEFAULT_FIBER_RADIUS_ARCSEC"]
