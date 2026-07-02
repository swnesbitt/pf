---
title: PMM High Resolution Precipitation Radar Atlas
emoji: 🌧️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PF swath-grid climatology — xpublish-tiles map server

[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9189-blue)](https://doi.org/10.57967/hf/9189)

Interactive slippy-map of the **TRMM PR / GPM Ku precipitation-feature swath-grid
climatology**. Dataset DOI: **[10.57967/hf/9189](https://doi.org/10.57967/hf/9189)**. Map tiles are rendered on the fly by
[`xpublish-tiles`](https://github.com/xpublish-community/xpublish-tiles) from a
precomputed Zarr, served by FastAPI, with a small Leaflet front-end at `/`.

**36 fields** = 12 quantities × 3 members, **annual** (summed over all months &
UTC hours), on the native **0.05°** shared ±68° grid. The quantities cover total
precipitation plus a **convective / stratiform** split (radar 2A `typePrecip`
classification):

| quantity | definition | units |
|---|---|---|
| `rate` | Σrain / Nviews (unconditional) | mm/hr |
| `freq` | Nraining / Nviews | fraction |
| `intensity` | Σrain / Nraining (conditional) | mm/hr |
| `rain` | annual accumulation = rate × 8766 h/yr | mm/year |
| `raining_views` | raining pixel-views | count |
| `views` | radar pixel-views (denominator) | count |
| `conv_rain` | **convective** annual accumulation | mm/year |
| `strat_rain` | **stratiform** annual accumulation | mm/year |
| `conv_freq` | convective frequency = Nraining_conv / Nviews | fraction |
| `strat_freq` | stratiform frequency = Nraining_strat / Nviews | fraction |
| `conv_intensity` | convective conditional rate | mm/hr |
| `strat_intensity` | stratiform conditional rate | mm/hr |

Members: **GPM** (Ku, 2014–2026), **TRMM** (PR, 1997–2015, zero poleward of ±38°),
**COMBINED** (GPM+TRMM pooled). The map viewer exposes 11 of these as a grouped
quantity dropdown (All precipitation / Convective / Stratiform); `rate` is in the
store but not the dropdown (`rain` is the user-facing accumulation field).

## How the data is supplied

The Space does **not** ship the Zarr in the image. At startup `app.py` pulls
`pf_tiles.zarr` from a **HuggingFace Dataset repo** named by the `HF_DATASET_REPO`
secret/variable, via `huggingface_hub.snapshot_download`. The store is ~750 MB.

### 1. Push the precomputed Zarr to a Dataset repo

```bash
pip install huggingface_hub
huggingface-cli login           # needs a write token

huggingface-cli repo create pf-grid-tiles --repo-type dataset
huggingface-cli upload pf-grid-tiles \
    /data/scratch/a/snesbitt/pf_tiles.zarr pf_tiles.zarr \
    --repo-type dataset
```

(or `HfApi().upload_folder(folder_path=".../pf_tiles.zarr",
path_in_repo="pf_tiles.zarr", repo_id="<user>/pf-grid-tiles",
repo_type="dataset")`.)

### 2. Create the Docker Space

```bash
huggingface-cli repo create pf-grid-tiles-app --repo-type space --space_sdk docker
git clone https://huggingface.co/spaces/<user>/pf-grid-tiles-app
cp app.py requirements.txt Dockerfile README.md pf-grid-tiles-app/
cd pf-grid-tiles-app && git add -A && git commit -m "tile server" && git push
```

### 3. Point the Space at the Dataset

In the Space **Settings → Variables**, set:

```
HF_DATASET_REPO = <user>/pf-grid-tiles
```

(If the dataset is private, also add a `HF_TOKEN` secret with read access — the
`huggingface_hub` client picks it up automatically.)

The Space builds, pulls the Zarr once, and serves the map at its root URL.

## Run locally

```bash
pip install -r requirements.txt
TILES_ZARR=/data/scratch/a/snesbitt/pf_tiles.zarr uvicorn app:app --port 7860
# open http://localhost:7860/
```

Tile endpoint (OGC/XYZ):
`/tiles/WebMercatorQuad/{z}/{y}/{x}?variables={member}_{quantity}&style=raster/{cmap}&width=256&height=256`
