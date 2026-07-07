#!/usr/bin/env python

"""
ifuCube: modular flux conserving IFU cube builder

Minimal. general framework for building 3D data cubes from IFU/fiber data
contained in an RSS file using a flux-conserving Shepard style interpolation.

-Instrument agnostic: should work with generic numpy arrays
-Flux-conserving: projection operator A from fibers to pixels
has column normalized weights, so each fibers total flux is preserved when
redistributed to the spaxels (up to floating point noise)
-Modular: separate modules for input/output, geometry, kernels, cube building,
diagnostics, and region handling.
"""

from .io_readers import FiberSet, fibers_from_arrays
from .geometry import make_grid_from_points, fit_celestial_wcs, make_wcs3d
from .kernels import build_projection_A, splat_cube, ivar_through_A, neff_from_A, backproject
from .cube_build import build_cube, save_cube_to_fits
from .diagnostics import global_flux_ratio
from .mapcat_build import build_map, save_map_to_fits
from .regions import (
    fibers_in_polygon,
    fibers_in_circle,
    region_spectrum_from_fibers
)
