#!/usr/bin/env python

"""
constants.py
------------
Instrument facts for the SDSS-V Local Volume Mapper science IFU.

This module exists so that nothing else has to own these numbers. It was
previously `LVM_FILL_FACTOR` defined in mapcat_build.py and imported by
cube_build.py inside a try/except with a hardcoded fallback -- which meant
two copies of the same constant that could silently disagree, and a cube
module that depended on the map module for no other reason.

Imports numpy and nothing else, so any module in the package can import it
without creating a cycle.

Reference: Drory et al. 2024, AJ 168, 198 (the LVM instrument paper).
"""

from __future__ import annotations

import numpy as np

# --- Science IFU geometry ---------------------------------------------------

#: Projected diameter of one science fiber on the sky.
LVM_FIBER_DIAMETER_ARCSEC = 35.3

#: Center-to-center spacing of the hexagonal fiber lattice.
LVM_FIBER_PITCH_ARCSEC = 37.0

#: Number of science fibers in the IFU (25 rings of a hex lattice, fed to
#: three spectrographs through three rhombic slit segments).
N_SCIENCE_FIBERS = 1801

#: Total fibers including the sky and standard-star fibers that flank each
#: slit block. A DAP table with this many rows has NOT been cut to science.
N_ALL_FIBERS = 1944


# --- Derived ----------------------------------------------------------------

def _fill_factor(diameter_arcsec: float, pitch_arcsec: float) -> float:
    """
    Fraction of the sky actually covered by fiber apertures.

    One fiber of diameter d has area pi*(d/2)**2. On a hexagonal lattice of
    pitch p, each fiber owns a hexagonal cell whose area is

        A_cell = (sqrt(3)/2) * p**2

    (a regular hexagon of center-to-center spacing p; equivalently
    (3*sqrt(3)/2) * s**2 for side s = p/sqrt(3)). The ratio is therefore

        f = pi * (d/2)**2 / ((sqrt(3)/2) * p**2)

    For the LVM numbers above this is 0.825477: about 17.5% of the sky
    between the fibers is never measured. Light falling there is lost, so a
    map that spreads each fiber's flux over its whole cell reads ~0.83x the
    true sky surface brightness on smooth emission.

    The factor is a pure constant -- independent of pixel size, kernel width
    and everything else -- so it cancels in any LVM-internal comparison and
    in any line ratio. It matters only when comparing LVM surface
    brightnesses against another survey or a model in absolute units.
    """
    aperture = np.pi * (diameter_arcsec / 2.0) ** 2
    cell = (np.sqrt(3.0) / 2.0) * pitch_arcsec ** 2
    return float(aperture / cell)


#: Aperture fill factor, 0.825477 for the nominal LVM geometry.
LVM_FILL_FACTOR = _fill_factor(LVM_FIBER_DIAMETER_ARCSEC,
                               LVM_FIBER_PITCH_ARCSEC)

#: Radius, for callers that want it instead of the diameter.
LVM_FIBER_RADIUS_ARCSEC = LVM_FIBER_DIAMETER_ARCSEC / 2.0


__all__ = [
    "LVM_FIBER_DIAMETER_ARCSEC",
    "LVM_FIBER_RADIUS_ARCSEC",
    "LVM_FIBER_PITCH_ARCSEC",
    "LVM_FILL_FACTOR",
    "N_SCIENCE_FIBERS",
    "N_ALL_FIBERS",
]
