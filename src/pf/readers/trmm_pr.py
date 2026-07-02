"""Reader for the GPM-reprocessed TRMM PR 2A product (``GPM_2APR``, V07).

The GPM-reprocessed TRMM PR product is HDF5 in the *same* structure as the GPM
2A-DPR FS swath, so :class:`TrmmPrReader` reuses
:class:`~pf.readers.gpm_ku.GpmKuReader` verbatim, overriding only the product
``short_name`` and ``mission``.

TRMM-reprocessed specifics that make this a no-op subclass:

* the ``FS`` swath group and all ``FS/*`` paths exist identically;
* 176 range bins at 125 m spacing (same as GPM Ku);
* reflectivity fields are 2-D ``(nscan, nray)`` / 3-D ``(nscan, nray, 176)``
  with **no** trailing dual-frequency axis, so the inherited ``_select_ku``
  guard (``ndim >= 3 and shape[-1] == 2``) never fires and is a safe no-op.

The orbit regex, ``ScanTime`` composition, and ``_height_3d`` are inherited.
"""

from __future__ import annotations

from pf.readers.gpm_ku import GpmKuReader


class TrmmPrReader(GpmKuReader):
    """Reader for the GPM-reprocessed TRMM PR 2A FS swath (``GPM_2APR``)."""

    short_name: str = "GPM_2APR"
    mission: str = "TRMM"
