#!/usr/bin/env python

"""
geometry.py
-----------
Spatial geometry and WCS construction

- Builds a regular pixel grid (xg, yg) covering the fibers
- Construct an "inside IFU" mask via convex hull
- Fit a 2D celestial WCS from fiber (x,y) and RA/dec
- Embed the 2D WCS into a 3D (RA, dec, lambda) WCS.

IMPORTANT -- pixel coordinate convention
----------------------------------------
The image array produced by the builders has its element [0, 0] at the
tangent-plane position (xg[0], yg[0]), which is NOT (0, 0) arcsec -- it is
the bottom-left corner of the padded footprint, i.e. roughly minus half the
field width. Anything that converts between arcsec and *array pixel* units
must therefore use

    x_pix = (x_arcsec - xg[0]) / pixel_size_arcsec
    y_pix = (y_arcsec - yg[0]) / pixel_size_arcsec

and not simply x_arcsec / pixel_size_arcsec. Using the latter shifts the
fitted WCS relative to the array by (-xg[0], -yg[0]), which for an LVM
pointing is about a third of a degree -- larger than the field of view.
`fit_celestial_wcs` takes x0_arcsec / y0_arcsec for exactly this reason;
pass xg[0] and yg[0] from `make_grid_from_points`.
"""

from __future__ import annotations
from typing import Tuple, Optional
import warnings

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points as _fit_wcs
from astropy.wcs.utils import proj_plane_pixel_area

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
    - Every pixel center inside this expanded hull is considered "inside"

    Parameters
    ----------
    x_arcsec, y_arcsec : np.ndarray
        1D array of fiber positions (arcsec)
    pixel_size_arcsec : float
        Desired spaxel size (arcsec)
    fiber_radius_arcsec : float
        Fiber radius (arcsec). Used twice: as the padding added around the
        fiber bounding box, and as the distance the convex hull is pushed
        outward from its centroid so edge fibers are not clipped.

    Returns
    -------
    xg, yg : np.ndarray
        1D arrays of pixel-center coordinates (arcsec). NOTE: xg[0], yg[0]
        are the coordinates of array element [0, 0] and are generally large
        and negative -- see the module docstring.
    inside_mask : np.ndarray (bool)
        2D mask (ny, nx). True = inside IFU footprint
    """
    x = np.asarray(x_arcsec, float)
    y = np.asarray(y_arcsec, float)
    ps = float(pixel_size_arcsec)
    R = float(fiber_radius_arcsec)

    # construct grid using max x and y values and pad by fiber radius
    x_min = x.min() - R
    x_max = x.max() + R
    y_min = y.min() - R
    y_max = y.max() + R

    xg = np.arange(x_min, x_max + ps, ps)
    yg = np.arange(y_min, y_max + ps, ps)

    if x.size < 3:
        # degenerate case: 1-2 points only. Mark all grid pixels as inside
        X, Y = np.meshgrid(xg, yg)
        inside = np.ones_like(X, dtype=bool)
        return xg, yg, inside

    points = np.column_stack([x, y])
    hull = ConvexHull(points)
    poly = points[hull.vertices]

    # expand polygon away from its centroid by 'pad' arcsecs
    cx, cy = poly.mean(axis=0)
    v = poly - np.array([cx, cy])
    r = np.hypot(v[:, 0], v[:, 1]) + 1e-9 # avoid divide by 0
    scale = (r + R) / r # got rid of scale R, scale = (r + R)/r, works fine like this!
    poly_expanded = np.column_stack([cx + v[:, 0] * scale,
                                     cy + v[:, 1] * scale])

    X, Y = np.meshgrid(xg, yg)
    poly_expanded = np.vstack([poly_expanded, poly_expanded[:1]])
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
    x0_arcsec : float = 0.0,
    y0_arcsec : float = 0.0,
    max_residual_arcsec : float = 1.0,
) -> WCS:
    """
    Fit a 2D celestial WCS (RA, Dec) from fiber plane coords (x, y) and
    sky coords (RA, Dec) using astropy's fit_wcs_from_points.

    Conceptually solving for a mapping:

        (array pixel) -> (RA, dec)

    The fit is performed in *array pixel* units, obtained from the
    tangent-plane arcsec positions as

        x_pix = (x_arcsec - x0_arcsec) / pixel_size_arcsec
        y_pix = (y_arcsec - y0_arcsec) / pixel_size_arcsec

    so x0_arcsec / y0_arcsec MUST be the coordinates of array element
    [0, 0] -- i.e. xg[0] and yg[0] from make_grid_from_points. The origin is
    emphatically NOT arbitrary: the returned WCS is written into the FITS
    header of an array whose [0, 0] element sits at (xg[0], yg[0]), and a
    WCS fitted in any other pixel system is offset from that array by the
    difference. For an LVM pointing that offset is ~0.35 deg, larger than
    the field itself.

    Parameters
    ----------
    x_arcsec, y_arcsec : (K,)
        Fiber tangent-plane positions (arcsec).
    ra_deg, dec_deg : (K,)
        Fiber sky positions (degrees, ICRS).
    pixel_size_arcsec : float
        Output pixel scale.
    x0_arcsec, y0_arcsec : float
        Tangent-plane coordinates of array element [0, 0]; pass xg[0], yg[0].
        The defaults of 0.0 reproduce the old (incorrect) behaviour and are
        kept only so the signature stays backward compatible -- always pass
        the grid origin.
    max_residual_arcsec : float
        If the RMS of the fit residuals exceeds this, a warning is issued.
        A healthy TAN fit over an LVM field lands well under 0.05 arcsec;
        anything large means the positions and the sky coordinates disagree.

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
    # Reference to the array origin, NOT to (0, 0) arcsec. See docstring.
    x_pix = (x - float(x0_arcsec)) / ps
    y_pix = (y - float(y0_arcsec)) / ps

    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    wcs2d = None
    # astropy changed this signature: newer versions take a 2-tuple of
    # arrays, older ones took a single stacked (N, 2) array. Catch only
    # TypeError so a genuine problem with the inputs still propagates.
    try:
        wcs2d = _fit_wcs(
            xy=(x_pix, y_pix),
            world_coords=sky,
            projection="TAN",
        )
    except TypeError:
        try:
            wcs2d = _fit_wcs(
                xy=np.column_stack([x_pix, y_pix]),
                world_coords=sky,
                projection="TAN",
            )
        except TypeError as e:
            raise RuntimeError(
                f"fit_wcs_from_points failed for both API versions: {e}"
            )

    # Verify the fit actually reproduces the input positions. This is cheap
    # and it is the check that would have caught the pixel-origin bug.
    fitted = wcs2d.pixel_to_world(x_pix, y_pix)
    resid = fitted.separation(sky).to(u.arcsec).value
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if rms > float(max_residual_arcsec):
        warnings.warn(
            f"fit_celestial_wcs: RMS residual {rms:.3f} arcsec exceeds "
            f"{max_residual_arcsec} arcsec (max {resid.max():.3f}). The WCS "
            "does not describe these fiber positions well -- check that "
            "x0_arcsec/y0_arcsec are xg[0]/yg[0] and that ra/dec match x/y.",
            RuntimeWarning,
        )
    return wcs2d


def pixel_area_arcsec2(wcs2d: WCS) -> float:
    """
    Solid angle of one pixel, in square arcseconds, from a celestial WCS.

    Used to convert between the two normalization conventions:

        surface brightness = flux_per_pixel / pixel_area
        flux_per_pixel     = surface_brightness * pixel_area

    Taken from the WCS (via the CD/PC matrix) rather than from a stored
    pixel-size keyword, so it stays correct for a rotated or slightly
    skewed projection.

    Parameters
    ----------
    wcs2d : WCS
        Celestial WCS (2D, or one with a .celestial part).

    Returns
    -------
    area : float
        Pixel solid angle in arcsec**2.
    """
    cel = wcs2d.celestial if wcs2d.naxis != 2 else wcs2d
    return float(proj_plane_pixel_area(cel)) * 3600.0 ** 2


def make_wcs3d(celestial_wcs: WCS, wave: np.ndarray,
               warn_nonlinear: bool = True) -> WCS:
    """
    Embed a 2D celestial WCS into a 3D (RA, Dec, lambda) WCS.

    We keep the spatial part from celestial_wcs and add a spectral axis with
    a simple linear relation

        lambda(i) = CRVAL3 + (i + 1 - CRPIX3) * CD3_3

    where CD3_3 is set to the median wavelength step.

    Because a single CD3_3 can only describe a *linear* wavelength axis, a
    non-uniform grid (e.g. logarithmic) cannot be represented exactly. The
    exact grid is always written separately as the WAVE table HDU by
    save_cube_to_fits; `warn_nonlinear` checks the header description is
    honest and warns if it is not.

    Parameters
    ----------
    celestial_wcs : WCS
        Fitted 2D RA/Dec WCS (naxis=2)
    wave : np.ndarray
        1D array of wavelengths (wave_n)
    warn_nonlinear : bool
        Warn if the wavelength grid is not linear to within 1% of the
        median step.

    Returns
    -------
    w3d : WCS
        3D WCS with axes (RA, Dec, lambda)
    """

    wave = np.asarray(wave, float)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("wave must be 1D with at least 2 elements")

    dwave = np.diff(wave)
    step = float(np.median(dwave))
    if warn_nonlinear and step != 0.0:
        spread = float(np.ptp(dwave))
        if spread > 0.01 * abs(step):
            warnings.warn(
                f"make_wcs3d: wavelength steps vary by {spread:.4g} about a "
                f"median of {step:.4g} ({100*spread/abs(step):.1f}%). A single "
                "CD3_3 describes only a linear axis, so the 3D WCS spectral "
                "axis is approximate -- use the WAVE table HDU for exact "
                "wavelengths.",
                RuntimeWarning,
            )

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
    cd3[2, 2] = step

    # ensure non-singular
    if np.linalg.det(cd3) == 0.0:
        scale = 1.0 / 3600.0
        cd3 = np.diag([-scale, scale, step if step != 0.0 else 1.0])

    w3d.wcs.cd = cd3
    return w3d

def radec_to_tangent_arcsec(ra_deg, dec_deg):
    """Exact tangent-plane offsets (arcsec) about the field's mean direction.

    The mean direction is computed by averaging unit vectors rather than
    averaging RA/Dec, so it is well behaved near the poles and across the
    RA = 0 wrap.
    """
    c = SkyCoord(np.asarray(ra_deg) * u.deg, np.asarray(dec_deg) * u.deg, frame="icrs")
    v = c.cartesian.xyz.value.mean(axis=1)
    center = SkyCoord(np.arctan2(v[1], v[0]) * u.rad,
                      np.arctan2(v[2], np.hypot(v[0], v[1])) * u.rad, frame="icrs")
    off = c.transform_to(center.skyoffset_frame())
    return off.lon.to(u.arcsec).value, off.lat.to(u.arcsec).value
