"""Connected-component labeling of radar precipitation features (RPFs).

A "radar precipitation feature" is a contiguous region of near-surface
reflectivity at or above a threshold, retained only if its area-weighted extent
(summed per-pixel footprint area, not pixel count) meets a minimum. Unlike the
storm-cell detector this is derived from, there is **no 40-dBZ core gate**:
retention is purely threshold + minimum area.

Array adjacency in the ``(nscan, nray)`` reference frame is treated as physical
swath contiguity. The returned ``local_label`` values are stable under
scikit-image's row-major labeling, which makes the downstream feature_id
deterministic.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage.measure import label as _sk_label

from pf.config import CONNECTIVITY, DBZ_THRESHOLD, MIN_AREA_KM2, MIN_PIXELS
from pf.swath import Swath


def label_rpf(
    swath: Swath,
    dbz_thresh: float = DBZ_THRESHOLD,
    min_area_km2: float = MIN_AREA_KM2,
    min_pixels: int = MIN_PIXELS,
    connectivity: int = CONNECTIVITY,
) -> tuple[npt.NDArray[np.int32], list[tuple[int, float]]]:
    """Label and area-filter radar precipitation features in a swath.

    Parameters
    ----------
    swath : pf.swath.Swath
        Swath providing ``near_sfc_dbz`` (the labeling field) and ``pixel_area``
        (per-pixel footprint km^2 used for the area-weighted retention test).
    dbz_thresh : float, optional
        Near-surface reflectivity threshold (dBZ). Pixels with finite
        ``near_sfc_dbz >= dbz_thresh`` form the binary mask.
    min_area_km2 : float, optional
        Minimum area-weighted feature area (km^2) for retention.
    min_pixels : int, optional
        Minimum number of member pixels for retention. A component is kept iff
        ``npixels >= min_pixels`` **and** ``area_km2 >= min_area_km2``.
    connectivity : int, optional
        Pixel connectivity for :func:`skimage.measure.label` (1 = 4-connected,
        2 = 8-connected).

    Returns
    -------
    labeled : numpy.ndarray of int32, shape (nscan, nray)
        Labeled image in which **only retained features are non-zero**; all
        background and rejected (too-small) labels are zeroed.
    kept : list of (int, float)
        ``(local_label, area_km2)`` pairs for the retained features, sorted by
        ``local_label`` ascending.
    """
    dbz = swath.near_sfc_dbz
    mask = np.isfinite(dbz) & (dbz >= dbz_thresh)

    labeled, n_labels = _sk_label(
        mask, connectivity=connectivity, return_num=True
    )
    labeled = labeled.astype(np.int32, copy=False)

    if n_labels == 0:
        return np.zeros_like(labeled), []

    # Area-weighted extent per label: sum the per-pixel footprint area over the
    # pixels belonging to each label (NOT a raw pixel count).
    label_ids = np.arange(1, n_labels + 1)
    areas = ndimage.sum_labels(
        swath.pixel_area, labels=labeled, index=label_ids
    )
    # Raw member-pixel count per label (index 0 = background; drop it).
    npixels = np.bincount(labeled.ravel(), minlength=n_labels + 1)[1:]

    kept: list[tuple[int, float]] = []
    keep_ids: list[int] = []
    for lid, area, npix in zip(label_ids, areas, npixels):
        if npix >= min_pixels and area >= min_area_km2:
            kept.append((int(lid), float(area)))
            keep_ids.append(int(lid))

    # Zero out every label that was not retained.
    out = np.where(np.isin(labeled, keep_ids), labeled, 0).astype(np.int32)

    kept.sort(key=lambda item: item[0])
    return out, kept


def touches_edge(labeled: npt.NDArray[np.integer], local_label: int) -> bool:
    """Return whether a labeled feature touches the swath edge.

    A feature "touches the edge" if any of its member pixels lies on the first
    or last scan (row 0 / row -1) or the first or last ray (column 0 /
    column -1) of the ``(nscan, nray)`` array.

    Parameters
    ----------
    labeled : numpy.ndarray
        Labeled image.
    local_label : int
        Label whose edge contact is tested.

    Returns
    -------
    bool
        ``True`` if the feature touches any array border, else ``False``.
    """
    member = labeled == local_label
    if not member.any():
        return False
    return bool(
        member[0, :].any()
        or member[-1, :].any()
        or member[:, 0].any()
        or member[:, -1].any()
    )
