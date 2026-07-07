#!/usr/bin/env python

"""
geometry.py
-----------
Spatial geometry and WCS construction

- Builds a regular pixel grid (xg, yg) covering the fibers
- Construct an "inside IFU" mask via convex hull
- Fit a 2D celestial WCS from fiber (x,y) and RA/dec
- Embed the 2D WCS into a 3D (RA, dec, lambda) WCS.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points as _fit_wcs

def make_grid_from_points(
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    pixel_size_arcsec: float,
    fiber_radius_arcsec: float = 18.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds a regular pixel grid that covers all fibers and an "inside IFU" mask
    based on the convex hull of the fiber positions.

    Math
    ----
    - Fibers define a set of points (xi, yi)
    - Plane region they cover is well approximated by the convex hull
    - Expand the hull outward slightly so the grid doesn't clip fibers
    - Every pixel center inside this expanded hull is considers "inside"

    Parameters
    ----------
    x_arcsec, y_arcsec : np.ndarray
        1D array of fiber positions (arcsec)
    pixel_size_arcsec : float
        Desiered spaxel size (arcsec)
    margin_pixels: float
        How many pixel widths of extra padding to add around the hull

    Returns 
    -------
    xg, yg : np.ndarray
        1D arrays of pixel-center coordinates (arcsec)
    inside_mask : np.ndarray (bool)
        2D mask (ny, nx). True = inside IFU footprint
    """
    x = np.asarray(x_arcsec, float)
    y = np.asarray(y_arcsec, float)
    ps = float(pixel_size_arcsec)
    #R = float(fiber_radius_arcsec)

    # construct grid using max x and y values and pad by fiber radius
    x_min = x.min() #- R
    x_max = x.max() #+ R
    y_min = y.min() #- R
    y_max = y.max() #+ R

    xg = np.arange(x_min, x_max + ps, ps)
    yg = np.arange(y_min, y_max + ps, ps)

    if x.size < 3:
        # degenerate case: 1-2 points only. Mark all grid pixels as inside
        X, Y = np.meshgrid(xg, yg)
        inside = np.ones_like(X, dtype=bool)
        return xg, yg, inside
    
    points = np.column_stack([x,y])
    hull = ConvexHull(points)
    poly = points[hull.vertices]

    # expand polygon away from its centroid by 'pad' arcsecs
    cx, cy = poly.mean(axis=0)
    v = poly - np.array([cx, cy])
    r = np.hypot(v[:, 0], v[:, 1]) + 1e-9 # avoid divide by 0
    scale = (r + 2) / r # got rid of scale R, scale = (r + R)/r, works fine like this! 
    poly_expanded = np.column_stack([cx + v[:, 0] * scale,
                                     cy + v[:, 1] * scale])

    X, Y = np.meshgrid(xg, yg)
    path = MplPath(poly_expanded, closed=True)
    inside = path.contains_points(
        np.column_stack([X.ravel(), Y.ravel()])
    ).reshape(Y.shape)

    return xg, yg, inside

def fit_celestial_wcs(
    x_arcsec : np.ndarray,
    y_arcsec : np.ndarray,
    ra_deg : np.ndarray,
    dec_deg : np.ndarray,
    pixel_size_arcsec : float, 
) -> WCS:
    """
    Fit a 2D celestial WCS (RA, Dec) from fiber plane coords (x, y) and
    sky coords (RA, Dec) using astropy's fit_wcs_from_points.

    Conceptually solving for a mapping:

    (x, y) -> (RA, dec)

    where (x, y) are in arcsec on a tangent plane and (RA, dec) are the
    true sky positions of each fiber.

    We first convert (x, y) into approximate "pixel" units by dividing by
    the desired pixel size. The absolute origin doesn't matter as long as the mapping
    is consistent

    Returns
    -------
    wcs2d : astropy.wcs.WCS
        2D celestial WCS object
    """

    x = np.asarray(x_arcsec, float)
    y = np.asarray(y_arcsec,float)
    ra = np.asarray(ra_deg, float)
    dec = np.asarray(dec_deg, float)

    if not (x.shape == y.shape == ra.shape == dec.shape):
        raise ValueError("x,y,ra,dec, must have the same shape")

    ps = float(pixel_size_arcsec)
    x_pix = x / ps 
    y_pix = y / ps

    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    try:
        wcs2d = _fit_wcs(
            xy=np.column_stack([x_pix, y_pix]),
            world_coords=sky,
            projection="TAN",
        )
        return wcs2d
    except Exception:
        pass # older astropy signature
    
    try:
        wcs2d = _fit_wcs(
            (x_pix, y_pix),
            world_coords=sky,
            projection="TAN",
        )
        return wcs2d
    except Exception as e:
        raise RuntimeError(
            f"fit_wcs_from_points failed for both API versions: {e}"
        )


def make_wcs3d(celestial_wcs: WCS, wave: np.ndarray) -> WCS:
    """
    Embed a 2D celestial WCS into a 3D (RA, Dec, lambda) WCS.

    We keep the spatial part from celestial_wcs and add a spectral axis with
    a simple linear relation

        lambda(i) = CRVAL3 + (i + 1 - CRPIX3) * CD3_3

    where CD3_# is set to the median wavelength step. 

    Parameters
    ----------
    celestial_wcs : WCS
        Fitted 2D RA/Dec WCS (naxis=2)
    wave : np.ndarray
        1D array of wavelengths (wave_n)

    Returns
    -------
    w3d : WCS
        3D WCS with axes (RA, Dec, lambda)
    """

    wave = np.asarray(wave, float)
    if wave.ndim != 1 or wave.size < 2: 
        raise ValueError("wave must be 1D with at least 2 elements")
    
    w3d = WCS(naxis=3)
    h2 = celestial_wcs.to_header(relax=True)

    # axis types
    w3d.wcs.ctype = [
        h2.get("CTYPE1", "RA---TAN"),
        h2.get("CTYPE2", "DEC--TAN"),
        "WAVE"
    ]

    # Reference values
    w3d.wcs.crval = [
        celestial_wcs.wcs.crval[0],
        celestial_wcs.wcs.crval[1],
        float(wave[0]),
    ]

    # Reference pixels: keep spatial CRPIX, spectral at pixel 1
    w3d.wcs.crpix = [
        celestial_wcs.wcs.crpix[0],
        celestial_wcs.wcs.crpix[1],
        1.0,
    ]

    # Build 3x3 CD matrix
    if getattr(celestial_wcs.wcs, "cd", None) is not None:
        cd2 = celestial_wcs.wcs.cd
    elif np.any(celestial_wcs.wcs.cdelt):
        cd2 = np.diag(celestial_wcs.wcs.cdelt[:2])
    else:
        # fallback: 1 arcsec/pixel nominal
        scale = 1.0 / 3600.0
        cd2 = np.array([[-scale, 0.0], [0.0, scale]])

    cd3 = np.zeros((3,3), float)
    cd3[:2, :2] = cd2
    cd3[2, 2] = np.median(np.diff(wave))

    # ensure non-singular
    if np.linalg.det(cd3) == 0.0:
        scale = 1.0 / 3600.0
        cd3 = np.diag([-scale, scale, np.median(np.diff(wave))])

    w3d.wcs.cd = cd3
    return w3d
    