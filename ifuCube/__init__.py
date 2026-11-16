"""
ifuCube
-------
Flux-conserving cubes, emission-line maps and mosaics from fibre IFU data.

Import the submodule you need:

    from ifuCube.io_lvm       import read_dap_multi
    from ifuCube.mapcat_build import build_map, read_map
    from ifuCube.cube_build   import build_cube, read_cube

NOTHING IS IMPORTED HERE ON PURPOSE. It would be friendlier to re-export the
main functions so that `from ifuCube import build_map` worked, but doing that
runs every submodule at `import ifuCube` time, and io_lvm needs `lvmdap`,
which is not on PyPI. A plain `import ifuCube` would then fail on any machine
without it -- including one that only builds maps from an existing FiberSet,
or only draws figures. Optional dependencies have to stay optional, so the
cost is one extra word at each import site.
"""

__version__ = "0.1.0"

# The submodules, for discoverability (`ifuCube.__all__`). Listing a name here
# does not import it.
__all__ = [
    "constants",
    "cube_build",
    "diagnostics",
    "fill",
    "geometry",
    "healpix_build",
    "io_lvm",
    "io_readers",
    "kernels",
    "mapcat_build",
    "mosaic",
    "plotting",
    "projection",
    "ratio",
    "regions",
]
