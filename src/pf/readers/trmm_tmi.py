"""Reader for the GPM-reprocessed TRMM TMI 1C product (``GPM_1CTRMMTMI``, V07).

The GPM-reprocessed TRMM TMI product is HDF5 in the *same* structure as the GPM
1C-GMI granule, so :class:`TrmmTmiReader` reuses
:class:`~pf.readers.gpm_gmi.GpmGmiReader` verbatim, overriding only the swath
group and PCT channel indices.

TMI carries its 85.5 GHz channels in the ``S3`` group: ``S3/Tc`` is
``(nscan, nray, 2)`` with **85.5V at last-axis index 0** and **85.5H at index
1**. All geolocation/time/spacecraft paths (``S3/Latitude``, ``S3/Longitude``,
``S3/ScanTime/*``, ``S3/SCstatus/*``) follow the same layout as GMI's ``S1``.
The PCT formula (``PCT_A*V - PCT_B*H``) and the V > H channel-order self-check
are inherited unchanged.

Unlike GMI (whose 36.5 GHz channels share the S1 swath with 89 GHz), TMI's
37 GHz channels live in the **separate** ``S2`` group at ``S2/Tc`` last-axis
indices **3 (37V)** and **4 (37H)** — with its own geolocation and spacecraft
sub-track. :class:`~pf.readers.gpm_gmi.GpmGmiReader` reads that swath
independently when ``pct37_swath`` differs from ``pct_swath``.
"""

from __future__ import annotations

from pf.readers.gpm_gmi import GpmGmiReader


class TrmmTmiReader(GpmGmiReader):
    """Reader for the GPM-reprocessed TRMM TMI 1C S3 swath (85.5 GHz PCT)."""

    short_name: str = "GPM_1CTRMMTMI"
    mission: str = "TRMM"
    swath: str = "S3"
    pct_swath: str = "S3"
    pct_v_idx: int = 0  # 85.5V
    pct_h_idx: int = 1  # 85.5H

    # 37 GHz lives in the separate S2 swath (own geolocation), not S3.
    pct37_swath: str = "S2"
    pct37_v_idx: int = 3  # 37V
    pct37_h_idx: int = 4  # 37H
