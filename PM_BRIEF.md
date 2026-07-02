# PMM High-Resolution Precipitation Radar Atlas — Program Brief

**What it is.** A publicly served, interactive web Atlas and an open, citable dataset that
turn nearly three decades of NASA spaceborne precipitation-radar observations into the
highest-resolution precipitation climatology of its kind — **0.05° (~5 km), near-global**.
It is built directly from the orbital swaths of the **TRMM Precipitation Radar (1997–2014)**
and the **GPM Dual-frequency Precipitation Radar / DPR, Ku band (2014–present)**, using the
algorithm near-surface precipitation rate (`precipRateNearSurface`, V07). Because every
observed radar pixel is counted, the sampling denominator and the precipitation fields are
physically consistent — yielding clean maps of precipitation **frequency**, **conditional
intensity**, and **annual accumulation** for TRMM, GPM, and a pooled (COMBINED) record.
Users explore it live in the browser (tiles rendered on demand) or download the full Zarr
store under a DOI (CC-BY-4.0).

**Legacy & value to NASA.** The Atlas extends the approach of **Nesbitt & Anders (2009,
GRL)** — fine-scale TRMM-radar rainfall climatology — across the **full TRMM→GPM radar era**,
version-uniform and reproducible. It demonstrates the unique, enduring return on NASA's
investment in spaceborne precipitation *radar*: resolving orographic, coastal, land/lake,
and diurnal precipitation structure that coarser merged products cannot. As an open,
DOI-citable, continuously extensible resource, it serves the research, applications, and
education communities, anchors a growing precipitation-feature and diurnal-climatology
framework, and is positioned to migrate to GPM **V08** as that reprocessing completes —
preserving the radar record's scientific legacy for the post-mission era.

*Developed at the University of Illinois Urbana-Champaign (PI S. W. Nesbitt) with support
from the NASA Precipitation Measurement Missions and Weather programs.*
*Live Atlas: https://huggingface.co/spaces/snesbitt/pf-grid-tiles-app ·
Dataset DOI: https://doi.org/10.57967/hf/9189*

## Unique aspect of the development (how it differs from what's currently available)

Today's widely used precipitation climatologies (IMERG, GPCP, CMORPH) are merged
passive-microwave/IR products at coarser resolution (~0.1°+) that smear fine structure and
conflate *how often* it rains with *how hard*. Existing spaceborne-radar climatologies are
typically coarser, single-mission, or fixed-period. This development differs in four ways:

1. **Gridded directly from the orbital radar swaths at native 0.05° (~5 km)** with an
   explicit, physically-consistent **sampling ("views") denominator** — so precipitation
   **frequency, conditional intensity, and accumulation are cleanly separated**, not just rate.
2. **One version-uniform (V07) framework spanning the full TRMM→GPM radar era (1997–present)**,
   with GPM, TRMM, and a pooled member on a shared, cell-aligned grid.
3. **Open, DOI-citable, and interactive** — a live tile-served Atlas plus a downloadable Zarr,
   fully reproducible from the swaths, not a static figure set.
4. **Extensible foundation** — the same pipeline yields a diurnal (UTC-hour) climatology and
   per-feature environmental (ERA5 CAPE/shear) attribution, beyond any current product.

It extends the fine-scale TRMM-radar approach of Nesbitt & Anders (2009) to the GPM era,
open and continuously updatable.

## Significance for Earth system science

Precipitation is the central coupling in the water and energy cycles, and its fine spatial
and diurnal structure — orographic gradients, coastlines, land/lake/river contrasts, SST
fronts, land/sea breezes — governs regional climate, hydrology, and ecosystems yet is exactly
what coarse products miss. Spaceborne precipitation radar is the only instrument that measures
this structure directly and globally, and this Atlas realizes its full spatial information
across three decades.

That enables high-value science: **evaluation of convection-permitting and climate models**
(the diurnal cycle and orographic/coastal precipitation are persistent model biases),
**process studies** of convective organization and light-rain detectability,
**frequency-vs-intensity decomposition** for variability and trend analysis across the
TRMM→GPM record, and **hydrologic and water-resource applications** in mountainous and
data-sparse regions. By preserving sampling rigor and a version-uniform record, it provides a
defensible observational benchmark bridging weather and climate scales — sustaining the
scientific return of NASA's precipitation-radar missions into the post-mission era.
