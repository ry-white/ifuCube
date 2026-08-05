#!/usr/bin/env python

"""
cross_match.py
----------
A not so Robust : 

Cross-match lvm drp and WISE HII catalogs using a 
radius-based search. The cross-match is performed to identify HII regions in the LVM DRP data that correspond to known 
HII regions in the WISE catalog. 
The output is a list of matched regions with their respective properties.   

There is also a function to download all Sframes, and/or dap files via SDSS Access. The function will download all files that match
coordinates within the searched radius for the cross-matched WISE HII objects and LVM pointings and save them to a specified directory.
"""

from future__ import annotations
from typing import Sequence, Tuple  

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from sdss_access import Access
from sdss_access import Path

def cross_match_hii(radius_degree: float = 0.5) -> Table:
    """
    Cross-match LVM DRP and WISE HII catalogs using a radius-based search.

    Parameters
    ----------
    radius_arcsec : float
        Search radius in arcseconds for cross-matching

    Returns
    -------
    matched_table : astropy.table.Table
        Table containing matched regions with their respective properties
    """
    wise = Table.read("data/wise_hII.fits")  # Load WISE HII catalog
    drpall = Table.read("data/drpall-1.2.1.fits")  # Load LVM DRP catalog

    wise_coord = SkyCoord(ra=wise['_RA_icrs'], dec=wise_hii_catalog['_DE_icrs'])
    drp_coord = SkyCoord(ra=drpall['sci_ra'] * u.degree, dec=drpall['sci_dec'] * u.degree)

    # match WISE HII regions to LVM DRP regions within the specified radius, 
    # here we use a radius of 0.5 degrees (30 arcminutes) for the cross-match
    # as that's about LVM's own field of view (~0.5 degrees)
    idx_lvm, idx_wise, sep2d, _ = wise_coord.search_around_sky(drp_coord, seplimit=0.5 * u.deg)

    # matched_obs has the same structure as drpall table, but only contains matched objects
    # obviously... 
    matched_obs = drpall[np.unique(idx_lvm)]

    # now to download all Sframes and/or dap files via SDSS Access

    











