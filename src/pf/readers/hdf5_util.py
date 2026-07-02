"""Stateless helpers for reading and decoding HDF5 swath variables.

These functions are the *decode* inverse of the encode path in
``gpm-ingest/ingest_gpm_dpr.py``: raw 2A-DPR variables are already physical
floats, but the reader still honours any HDF5 ``scale_factor`` / ``add_offset``
attributes (present in encoded stores) and masks fill sentinels to ``NaN``.

All functions are side-effect free and never mutate their inputs. Group paths
are root-relative and include the swath prefix (e.g. ``"FS/SLV/zFactorFinal"``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pf.config import FILL_SENTINELS


def has_path(f: Any, group_path: str) -> bool:
    """Return whether ``group_path`` exists in the open HDF5 file ``f``."""
    return group_path in f


def read_var(f: Any, group_path: str, *, dtype: Any = np.float32) -> np.ndarray:
    """Read a dataset into memory as ``dtype``.

    Parameters
    ----------
    f : h5py.File or mapping
        Open HDF5 file (or compatible mapping).
    group_path : str
        Root-relative dataset path.
    dtype : numpy dtype, optional
        Target dtype for the returned array (default ``float32``).

    Returns
    -------
    ndarray

    Raises
    ------
    KeyError
        If ``group_path`` is absent.
    """
    if group_path not in f:
        raise KeyError(group_path)
    return np.asarray(f[group_path][:], dtype=dtype)


def decode_fill(
    arr: np.ndarray,
    *,
    sentinels: tuple[float, ...] = FILL_SENTINELS,
    atol: float = 0.05,
) -> np.ndarray:
    """Return a ``float32`` copy of ``arr`` with fill sentinels set to ``NaN``.

    Parameters
    ----------
    arr : ndarray
        Input array (not mutated).
    sentinels : tuple of float, optional
        Fill values to mask (default :data:`pf.config.FILL_SENTINELS`).
    atol : float, optional
        Absolute tolerance used by :func:`numpy.isclose` (default ``0.05``).

    Returns
    -------
    ndarray, float32
        A copy with masked sentinels replaced by ``NaN``.
    """
    out = np.array(arr, dtype=np.float32)
    mask = np.zeros(out.shape, dtype=bool)
    for fv in sentinels:
        mask |= np.isclose(out, fv, atol=atol, rtol=0.0)
    out[mask] = np.nan
    return out


def read_float(
    f: Any,
    group_path: str,
    *,
    sentinels: tuple[float, ...] = FILL_SENTINELS,
) -> np.ndarray:
    """Read a float field, applying scale/offset attrs and masking fills.

    Sentinel masking is performed on the *raw* values before applying any
    ``scale_factor`` / ``add_offset`` attributes, matching the encode order
    in ``ingest_gpm_dpr.py``.

    Parameters
    ----------
    f : h5py.File or mapping
        Open HDF5 file.
    group_path : str
        Root-relative dataset path.
    sentinels : tuple of float, optional
        Fill values to mask (default :data:`pf.config.FILL_SENTINELS`).

    Returns
    -------
    ndarray, float32
        Decoded, fill-masked, physically-scaled values.

    Raises
    ------
    KeyError
        If ``group_path`` is absent.
    """
    if group_path not in f:
        raise KeyError(group_path)
    dset = f[group_path]
    out = decode_fill(dset[:], sentinels=sentinels)

    attrs = getattr(dset, "attrs", {})
    scale_factor = attrs.get("scale_factor") if hasattr(attrs, "get") else None
    add_offset = attrs.get("add_offset") if hasattr(attrs, "get") else None
    if scale_factor is not None:
        out = out * np.float32(scale_factor)
    if add_offset is not None:
        out = out + np.float32(add_offset)
    return out.astype(np.float32, copy=False)


def read_int(f: Any, group_path: str, dtype: Any = np.int32) -> np.ndarray:
    """Read an integer field unchanged (no fill masking, no scaling).

    Parameters
    ----------
    f : h5py.File or mapping
        Open HDF5 file.
    group_path : str
        Root-relative dataset path.
    dtype : numpy dtype, optional
        Target integer dtype (default ``int32``).

    Returns
    -------
    ndarray

    Raises
    ------
    KeyError
        If ``group_path`` is absent.
    """
    if group_path not in f:
        raise KeyError(group_path)
    return np.asarray(f[group_path][:], dtype=dtype)
