"""Abstract base class for swath readers.

A :class:`SwathReader` turns an on-disk granule (a local HDF5 file path) into a
fully-populated :class:`~pf.swath.Swath` for a single sensor/product. Concrete
subclasses (e.g. :class:`~pf.readers.gpm_ku.GpmKuReader`) bind the mission and
NASA short-name and implement the product-specific HDF5 decoding.

Readers are side-effect free: :meth:`SwathReader.read` performs a *pure* read of
a local file (no network access, no temporary-file management) and
:meth:`SwathReader.orbit_of` is a deterministic parse. Orchestration concerns
such as downloading and cleanup live in :mod:`pf.granule`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pf.swath import Swath


class SwathReader(ABC):
    """Abstract interface for reading one sensor/product granule into a Swath.

    Attributes
    ----------
    short_name : str
        NASA short-name of the product this reader handles (e.g. ``"GPM_2ADPR"``).
    mission : str
        Mission identifier (e.g. ``"GPM"``), matching a key of
        :data:`pf.config.MISSION_CODE`.
    """

    #: NASA short-name of the product handled by this reader.
    short_name: str = ""
    #: Mission identifier handled by this reader.
    mission: str = ""

    @abstractmethod
    def read(self, path: str) -> Swath:
        """Read a local granule file into a fully-populated :class:`Swath`.

        This is a pure read: it must not download files, create temporary
        directories, or otherwise perform network/filesystem side effects beyond
        opening ``path`` for reading.

        Parameters
        ----------
        path : str
            Absolute path to a local granule file (HDF5 for GPM).

        Returns
        -------
        pf.swath.Swath
            A swath with all Phase-1 fields populated.
        """
        raise NotImplementedError

    @abstractmethod
    def orbit_of(self, granule_or_filename: Any) -> int:
        """Return the integer orbit number for a granule or filename.

        Deterministic parse of the product naming convention. Accepts either a
        plain filename/path string or a granule object (e.g. an
        :class:`earthaccess.DataGranule`) from which a filename can be derived.

        Parameters
        ----------
        granule_or_filename : str or object
            A filename/path string or a granule handle.

        Returns
        -------
        int
            The orbit number.

        Raises
        ------
        ValueError
            If an orbit number cannot be parsed.
        """
        raise NotImplementedError
