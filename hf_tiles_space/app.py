"""xpublish-tiles map server for the PF swath-grid climatology.

Serves OGC/XYZ map tiles of 6 quantities x 3 members (GPM / TRMM / COMBINED),
annual, on the 0.05 deg grid, from a precomputed Zarr. Designed to run as a
HuggingFace **Docker Space** (port 7860): the data is pulled once from a HF
Dataset repo at startup, then tiles are rendered on the fly by xpublish-tiles.

Every field is rendered with the **ChaseSpectral** weather colormap (cmweather)
on a FIXED per-variable color range = the [0.01, 0.99] data quantiles
(precomputed in tile_ranges.json), so all tiles of a field share one consistent
scale and the colorbar is accurate.

Env vars:
  HF_DATASET_REPO   HF dataset repo holding pf_tiles.zarr (e.g. "user/pf-grid-tiles")
  TILES_ZARR        local path to pf_tiles.zarr (overrides the HF pull, for local runs)
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.colors import to_hex
import numpy as np
import xarray as xr
import xpublish
import cmweather  # noqa: F401  registers ChaseSpectral & other weather colormaps
from fastapi.responses import HTMLResponse
from xpublish_tiles.xpublish.tiles import TilesPlugin

CMAP = "ChaseSpectral"

# --- locate the precomputed tile Zarr ---------------------------------------
DATA = os.environ.get("TILES_ZARR", "")
if not DATA or not os.path.exists(DATA):
    repo = os.environ.get("HF_DATASET_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=repo, repo_type="dataset",
                                  allow_patterns=["pf_tiles_ms.zarr/**"])
        DATA = os.path.join(local, "pf_tiles_ms.zarr")
    else:
        DATA = "pf_tiles_ms.zarr"

# Multiscale pyramid (DataTree): xpublish-tiles picks the resolution level per zoom,
# so low-zoom tiles read PRE-AVERAGED coarse levels instead of on-the-fly coarsening
# (which over-coarsened -> datashader quadmesh drew cells as rectangles = blockiness).
# The finest level (0) also backs the RANGES fallback + value-at-cursor sampling.
from xpublish_tiles.multiscale import assign_leaf_xpublish_ids
tree = xr.open_datatree(DATA, engine="zarr", consolidated=False)
assign_leaf_xpublish_ids(tree)
ds = tree["0"].to_dataset()

MEMBERS = ["GPM", "TRMM", "COMBINED"]
QUANTS = ["freq", "intensity", "rain", "raining_views", "views",
          "conv_rain", "strat_rain", "conv_freq", "strat_freq",
          "conv_intensity", "strat_intensity", "conv_rain_frac", "conv_pixel_frac",
          "echotop20", "echotop30", "echotop40",
          "freq_gt25", "freq_gt50", "freq_gt75", "freq_gt100",
          "eps_conv", "eps_strat", "nw_conv", "nw_strat", "dm_conv", "dm_strat"]
# quantity dropdown grouped into <optgroup>s: total metrics, heavy rain, then conv,
# then strat (each group roughly ordered rain -> freq -> intensity -> structure/DSD)
QGROUPS = [
    ("All precipitation", ["freq", "intensity", "rain", "raining_views", "views"]),
    ("Heavy rain frequency", ["freq_gt25", "freq_gt50", "freq_gt75", "freq_gt100"]),
    ("Convective", ["conv_rain", "conv_freq", "conv_intensity",
                    "conv_rain_frac", "conv_pixel_frac",
                    "echotop20", "echotop30", "echotop40",
                    "eps_conv", "nw_conv", "dm_conv"]),
    ("Stratiform", ["strat_rain", "strat_freq", "strat_intensity",
                    "eps_strat", "nw_strat", "dm_strat"]),
]
QLABEL = {"freq": "precipitation frequency",
          "intensity": "conditional intensity (mm/hr)", "rain": "annual mean precipitation (mm/year)",
          "raining_views": "precipitation views", "views": "views (denominator)",
          "conv_rain": "convective precipitation (mm/year)",
          "strat_rain": "stratiform precipitation (mm/year)",
          "conv_freq": "convective frequency", "strat_freq": "stratiform frequency",
          "conv_intensity": "convective intensity (mm/hr)",
          "strat_intensity": "stratiform intensity (mm/hr)",
          "conv_rain_frac": "convective rainfall fraction",
          "conv_pixel_frac": "convective area fraction",
          "echotop20": "convective 20 dBZ echo-top height (m)",
          "echotop30": "convective 30 dBZ echo-top height (m)",
          "echotop40": "convective 40 dBZ echo-top height (m)",
          "freq_gt25": "frequency of rain ≥ 25 mm/hr",
          "freq_gt50": "frequency of rain ≥ 50 mm/hr",
          "freq_gt75": "frequency of rain ≥ 75 mm/hr",
          "freq_gt100": "frequency of rain ≥ 100 mm/hr",
          "eps_conv": "convective DSD ε (epsilon)",
          "eps_strat": "stratiform DSD ε (epsilon)",
          "nw_conv": "convective Nw (log₁₀ m⁻³mm⁻¹)", "nw_strat": "stratiform Nw (log₁₀ m⁻³mm⁻¹)",
          "dm_conv": "convective Dm (mm)", "dm_strat": "stratiform Dm (mm)"}
# clean human labels for the quantity dropdown (option value stays the data key)
QDISPLAY = {"freq": "frequency", "intensity": "conditional intensity", "rain": "mean precipitation",
            "raining_views": "precipitation views", "views": "views",
            "conv_rain": "convective precipitation", "strat_rain": "stratiform precipitation",
            "conv_freq": "convective frequency", "strat_freq": "stratiform frequency",
            "conv_intensity": "convective intensity", "strat_intensity": "stratiform intensity",
            "conv_rain_frac": "convective rainfall fraction",
            "conv_pixel_frac": "convective area fraction",
            "echotop20": "convective 20 dBZ echo top", "echotop30": "convective 30 dBZ echo top",
            "echotop40": "convective 40 dBZ echo top",
            "freq_gt25": "freq ≥ 25 mm/hr", "freq_gt50": "freq ≥ 50 mm/hr",
            "freq_gt75": "freq ≥ 75 mm/hr", "freq_gt100": "freq ≥ 100 mm/hr",
            "eps_conv": "convective epsilon", "eps_strat": "stratiform epsilon",
            "nw_conv": "convective Nw", "nw_strat": "stratiform Nw",
            "dm_conv": "convective Dm", "dm_strat": "stratiform Dm"}
# plain-language explanations shown by the "?" next to the quantity selector
LAYMAN = {
  "freq": "How OFTEN precipitation falls here — the fraction of radar snapshots that "
          "caught any precipitation. High in the wet tropics, near zero over deserts.",
  "intensity": "How HARD precipitation falls when it IS falling — the average rate "
               "counting only the precipitating snapshots. Storm-prone places score high "
               "even if precipitation is rare.",
  "rain": "The estimated TOTAL precipitation in a typical year (millimetres per year), "
          "built from the average rate. This is the classic 'how wet is it' map.",
  "raining_views": "Number of satellite pixels in each box with precipitation.",
  "views": "Number of satellite pixels in each box — the sampling count behind every "
           "other field.",
  "conv_rain": "Annual precipitation from CONVECTIVE systems — intense, vertically "
               "developed showers and thunderstorms. Dominates the deep tropics and "
               "warm-season continents.",
  "strat_rain": "Annual precipitation from STRATIFORM systems — widespread, gentler "
                "precipitation, often the broad trailing/anvil regions of storms and "
                "mid-latitude fronts.",
  "conv_freq": "How OFTEN convective (intense, showery) precipitation falls here.",
  "strat_freq": "How OFTEN stratiform (widespread, gentle) precipitation falls here.",
  "conv_intensity": "Average rate of CONVECTIVE precipitation when it falls — highest in "
                    "the deep tropics and over warm continents.",
  "strat_intensity": "Average rate of STRATIFORM precipitation when it falls — far steadier "
                     "and lighter than convective.",
  "conv_rain_frac": "Of the TOTAL rainfall here, the FRACTION that comes from convective "
                    "(intense/showery) systems rather than stratiform — convective "
                    "accumulation divided by total accumulation.",
  "conv_pixel_frac": "Of the precipitating pixels here, the FRACTION that are convective "
                     "(intense/showery) rather than stratiform. ~1 in deep-convective "
                     "regions, low where widespread stratiform rain dominates.",
  "echotop20": "How TALL convective storms reach — the average height of the 20 dBZ radar "
               "echo top in convective pixels. A measure of storm depth; tallest over the "
               "deep tropics and warm-season continents.",
  "echotop30": "Average height of the 30 dBZ convective echo top — a stronger-reflectivity "
               "(deeper precipitation core) version of storm depth than the 20 dBZ top.",
  "echotop40": "Average height of the 40 dBZ convective echo top — the depth of the most "
               "intense precipitation cores; high tops flag vigorous, often hail-bearing storms.",
  "freq_gt25": "How OFTEN near-surface rain reaches at least 25 mm/hr — the occurrence rate "
               "of heavy rain. Concentrated in the convective tropics and monsoon regions.",
  "freq_gt50": "How OFTEN near-surface rain reaches at least 50 mm/hr — intense, "
               "flooding-grade rates that are rare outside deep convection.",
  "freq_gt75": "How OFTEN near-surface rain reaches at least 75 mm/hr — very intense rates, "
               "found almost only in vigorous convective storms.",
  "freq_gt100": "How OFTEN near-surface rain reaches at least 100 mm/hr — extreme rates, the "
                "rarest heavy-rain class, in the most intense convective cores.",
  "eps_conv": "Average DSD 'epsilon' adjustment in CONVECTIVE rain — the retrieval's "
              "drop-size correction factor (≈1). Higher values mean larger drops than the "
              "default size–rate relation assumes.",
  "eps_strat": "Average DSD 'epsilon' adjustment in STRATIFORM rain — the drop-size "
               "correction factor (≈1), typically a touch lower than in convective rain.",
  "nw_conv": "Average normalized drop concentration Nw, as log₁₀(Nw) with Nw in mm⁻¹ m⁻³, for "
             "CONVECTIVE rain — how MANY drops there are for a given size. Higher = more numerous drops.",
  "nw_strat": "Average normalized drop concentration log₁₀(Nw) for STRATIFORM rain.",
  "dm_conv": "Average mass-weighted mean drop diameter Dm (mm) in CONVECTIVE rain — how BIG "
             "the typical raindrops are. Larger in vigorous convection.",
  "dm_strat": "Average mass-weighted mean drop diameter Dm (mm) in STRATIFORM rain — the "
              "drop-size signature of widespread, gentler precipitation.",
}

# --- fixed per-variable color ranges (the [0.01, 0.99] quantiles) -----------
# Prefer the precomputed sidecar; fall back to computing from the dataset.
HERE = os.path.dirname(os.path.abspath(__file__))
RANGES = {}
_sidecar = os.path.join(HERE, "tile_ranges.json")
if os.path.exists(_sidecar):
    with open(_sidecar) as f:
        RANGES = json.load(f)
else:
    for v in ds.data_vars:
        vals = ds[v].values.ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            q001, q999 = (float(x) for x in np.quantile(vals, [0.001, 0.999]))
        else:
            q001, q999 = 0.0, 1.0
        if not (q999 > q001):
            q001, q999 = 0.0, (q999 if q999 > 0 else 1.0)
        RANGES[v] = {"vmin": q001, "vmax": q999,
                     "units": ds[v].attrs.get("units", ""),
                     "long_name": ds[v].attrs.get("long_name", v)}

# --- ChaseSpectral as CSS gradient stops for the legend ---------------------
_cmap = mpl.colormaps.get_cmap(CMAP)
CMAP_STOPS = [to_hex(_cmap(x)) for x in np.linspace(0, 1, 16)]

rest = xpublish.SingleDatasetRest(tree, plugins={"tiles": TilesPlugin()})
app = rest.app

# --- the Leaflet front-end at / ---------------------------------------------
INDEX = """<!doctype html><html><head><meta charset='utf-8'>
<title>PMM High Resolution Precipitation Radar Atlas</title>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<style>
 html,body{margin:0;height:100%;font-family:system-ui,Segoe UI,Roboto,sans-serif}
 #map{height:100%;background:#000}
 .panel{position:absolute;z-index:1000;top:10px;left:50px;background:#111e;color:#eee;
   padding:12px 14px;border-radius:10px;font-size:13px;line-height:1.7;max-width:340px;
   box-shadow:0 2px 12px #0008}
 .panel h1{font-size:14px;margin:0 0 4px;color:#7fd;font-weight:600}
 .panel .sub{color:#9ab;font-size:11px;margin-bottom:8px}
 .panel select,.panel input{font:13px system-ui;margin-left:6px;vertical-align:middle}
 #qhelp{display:inline-block;width:17px;height:17px;line-height:17px;text-align:center;
   border-radius:50%;background:#1a3a5a;color:#cdf;border:1px solid #6ad;font-size:12px;
   font-weight:700;cursor:pointer;margin-left:6px;vertical-align:middle;user-select:none}
 #qhelp:hover{background:#2a5a8a}
 #qpop{display:none;background:#0d1826;color:#dde;border:1px solid #355;border-radius:7px;
   padding:8px 10px;margin-top:6px;font-size:12px;line-height:1.5;max-width:312px}
 #qpop.show{display:block}
 #qpop b{color:#8ef}
 #desc{color:#9cf;font-size:12px;margin-top:4px}
 /* colorbar */
 #cbar{height:14px;border-radius:3px;margin:8px 0 2px;border:1px solid #0006}
 #cbticks{display:flex;justify-content:space-between;font-size:10px;color:#bcd}
 #cbunits{font-size:10px;color:#9ab;text-align:center;margin-top:1px}
 .row{margin-top:6px}
 .btn{background:#244;color:#bff;border:1px solid #488;border-radius:6px;
   padding:4px 8px;font-size:12px;cursor:pointer}
 .btn:hover{background:#366}
 .credit{position:absolute;z-index:1000;bottom:22px;left:62px;background:#111c;color:#9ab;
   font-size:10px;padding:3px 7px;border-radius:6px}
 .credit a{color:#7cf;text-decoration:none}
 /* ? help button bottom-left */
 #helpbtn{position:absolute;z-index:1001;bottom:14px;left:14px;width:38px;height:38px;
   border-radius:50%;background:#1a3a5a;color:#cdf;border:1px solid #6ad;font-size:20px;
   font-weight:700;cursor:pointer;box-shadow:0 2px 8px #0009}
 #helpbtn:hover{background:#2a5a8a}
 /* modal */
 #modal{position:absolute;inset:0;z-index:2000;background:#000a;display:none;
   align-items:center;justify-content:center}
 #modal.show{display:flex}
 .card{background:#0f1622;color:#dde;max-width:620px;max-height:82vh;overflow:auto;
   padding:22px 26px;border-radius:12px;box-shadow:0 6px 30px #000b;line-height:1.55;
   font-size:13.5px}
 .card h2{margin:0 0 8px;color:#8ef;font-size:17px}
 .card h3{margin:16px 0 4px;color:#7fd;font-size:13px;text-transform:uppercase;letter-spacing:.04em}
 .card a{color:#7cf}
 .card .x{float:right;cursor:pointer;color:#9ab;font-size:20px;line-height:1}
 .card ul{padding-left:18px;margin:6px 0}
 .card li{margin:6px 0}
</style></head><body>
<div id='map'></div>

<div class='panel'>
 <h1>PMM High Resolution Precipitation Radar Atlas</h1>
 <div class='sub'>TRMM PR + GPM Ku measurements · annual · 0.05°</div>
 <div class='row'>mission <select id='member'>__MEMBERS__</select></div>
 <div class='row'>quantity <select id='quant'>__QUANTS__</select></div>
 <div style='display:flex;align-items:center;gap:6px;margin-top:4px'>
   <span id='desc' style='color:#9ad'></span>
   <span id='qhelp' title='what does this show?'>?</span>
 </div>
 <div id='qpop'></div>
 <div id='cbar'></div>
 <div id='cbticks'><span id='cmin'></span><span id='cmid'></span><span id='cmax'></span></div>
 <div id='cbunits'></div>
 <div class='row'>color range
   <input id='cmin_b' type='number' step='any' title='colorbar minimum'
     style='width:86px;margin-left:6px'>
   <span style='color:#9ab'>–</span>
   <input id='cmax_b' type='number' step='any' title='colorbar maximum' style='width:86px'>
   <a href='#' id='creset' style='color:#7cf;font-size:11px;margin-left:6px'>reset</a></div>
 <div class='row'>opacity
   <input id='opacity' type='range' min='0' max='1' step='0.05' value='0.9' style='width:120px'>
   <span id='opval' style='font-size:11px;color:#9ab'>0.90</span>
 </div>
 <div id='readout' style='font-size:12px;color:#cfe;border-top:1px solid #2a3a4a;
   padding-top:7px;margin-top:7px;min-height:1.3em'><span style='color:#9ab'>Value at cursor:</span> move cursor over the map</div>
</div>

<button id='helpbtn' title='About this visualization'>?</button>

<div class='credit'>© University of Illinois Board of Trustees ·
 Contact: <a href='https://swnesbitt.github.io' target='_blank' rel='noopener'>Steve Nesbitt</a></div>

<div id='modal'><div class='card'>
 <span class='x' id='closex'>✕</span>
 <h2>About this visualization</h2>
 <p>This map shows a <b>very high resolution (0.05°) precipitation climatology</b>
 built from the radars aboard NASA's <b>Precipitation Measurement Missions</b> —
 the <b>TRMM</b> Precipitation Radar (Ku-band, 1997–2014) and the <b>GPM</b>
 Dual-frequency Precipitation Radar (DPR, Ku band, 2014–present). Every observed radar
 pixel is gridded directly from the orbital swaths, so the <i>views</i> denominator
 counts all sampling and the precipitation fields are consistent with it.</p>
 <p class='src'>Source: the near-surface precipitation rate
 (<code>precipRateNearSurface</code>) from GPM 2A-DPR Ku (<b>V07</b>) and the
 GPM-reprocessed TRMM PR 2A, <code>GPM_2APR</code> (<b>V07</b>).</p>
 <p>The goal is to resolve precipitation structure — related to islands,
 topography, land, lake and river cover, sea surface temperatures, land and sea
 breezes — from the most accurate tool we have from space, precipitation radar.</p>

 <h3>Sensitivity caveat — TRMM vs GPM</h3>
 <p>TRMM's Precipitation Radar has a minimum detectable reflectivity of
 <b>~17–18 dBZ</b>, whereas GPM's Dual-frequency Precipitation Radar (DPR) is more sensitive
 (<b>~12 dBZ</b>) and therefore detects substantially more <b>light
 precipitation</b> that TRMM misses. This difference can introduce
 <b>discontinuities</b> between the TRMM (1997–2015) and GPM (2014–present)
 records — and within the COMBINED member — that are most pronounced in regions
 and seasons where light rain is prevalent. Interpret cross-mission comparisons
 with this sensitivity gap in mind.</p>

 <h3>Key references — 0.05° (or finer) precipitation-radar climatology</h3>
 <ul>
  <li><b>Nesbitt, S. W., and A. M. Anders</b> (2009), Very high resolution
   precipitation climatologies from the Tropical Rainfall Measuring Mission
   precipitation radar, <i>Geophys. Res. Lett.</i>, 36, L15815,
   <a href='https://doi.org/10.1029/2009GL038026' target='_blank' rel='noopener'>doi:10.1029/2009GL038026</a>.</li>
  <li><b>Hirose, M., and K. Nakamura</b> (2005), Spatial and diurnal variation of
   precipitation systems over Asia observed by the TRMM Precipitation Radar,
   <i>J. Geophys. Res.</i>, 110, D05106,
   <a href='https://doi.org/10.1029/2004JD004815' target='_blank' rel='noopener'>doi:10.1029/2004JD004815</a>.</li>
  <li><b>Biasutti, M., S. E. Yuter, C. D. Burleyson, and A. H. Sobel</b> (2012),
   Very high resolution rainfall patterns measured by TRMM precipitation radar:
   seasonal and diurnal cycles, <i>Clim. Dyn.</i>, 39, 239–258,
   <a href='https://doi.org/10.1007/s00382-011-1146-6' target='_blank' rel='noopener'>doi:10.1007/s00382-011-1146-6</a>.</li>
  <li><b>Bookhagen, B., and D. W. Burbank</b> (2006), Topography, relief, and
   TRMM-derived rainfall variations along the Himalaya, <i>Geophys. Res. Lett.</i>,
   33, L08405,
   <a href='https://doi.org/10.1029/2006GL026037' target='_blank' rel='noopener'>doi:10.1029/2006GL026037</a>.</li>
  <li><b>Anders, A. M., and S. W. Nesbitt</b> (2015), Altitudinal precipitation
   gradients in the tropics from the Tropical Rainfall Measuring Mission (TRMM)
   precipitation radar, <i>J. Hydrometeorol.</i>, 16, 441–448,
   <a href='https://doi.org/10.1175/JHM-D-14-0178.1' target='_blank' rel='noopener'>doi:10.1175/JHM-D-14-0178.1</a>.</li>
  <li><b>Kidd, C., J. Kwiatkowski, and S. W. Nesbitt</b> (2010), Investigations
   into high resolution mapping of precipitation features utilizing the TRMM
   precipitation radar, <i>IGARSS 2010</i> (IEEE Int. Geosci. Remote Sens. Symp.),
   2337–2340,
   <a href='https://doi.org/10.1109/IGARSS.2010.5649629' target='_blank' rel='noopener'>doi:10.1109/IGARSS.2010.5649629</a>.</li>
 </ul>
 <h3>Data &amp; citation</h3>
 <ul>
  <li>Underlying gridded dataset (0.05° Zarr):
   <a href='https://huggingface.co/datasets/snesbitt/pf-grid-tiles' target='_blank' rel='noopener'>huggingface.co/datasets/snesbitt/pf-grid-tiles</a>.</li>
  <li>Cite this atlas —
   <a href='https://doi.org/10.57967/hf/9189' target='_blank' rel='noopener'><img
   src='https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9189-blue'
   alt='DOI 10.57967/hf/9189' style='vertical-align:middle'></a></li>
 </ul>
 <p style='color:#9ab;font-size:11px;margin-top:14px'>
  Tiles rendered on the fly with <a href='https://github.com/xpublish-community/xpublish-tiles' target='_blank' rel='noopener'>xpublish-tiles</a>;
  ChaseSpectral colormap from <a href='https://github.com/openradar/cmweather' target='_blank' rel='noopener'>cmweather</a>.
  © University of Illinois Board of Trustees ·
  Contact <a href='https://swnesbitt.github.io' target='_blank' rel='noopener'>Steve Nesbitt</a>.</p>
 <p style='color:#7fd;font-size:11.5px;margin-top:10px;border-top:1px solid #2a3a4a;padding-top:10px'>
  This work was supported by projects from the NASA Precipitation Measurement
  Missions and Weather programs to the University of Illinois.</p>
</div></div>

<script>
const CMAP='__CMAP__', STOPS=__STOPS__, QLABEL=__QLABEL__, LAYMAN=__LAYMAN__, RANGES=__RANGES__;
const map=L.map('map',{worldCopyJump:true,preferCanvas:false}).setView([10,120],3);
// satellite imagery base + political boundaries/places reference overlay (Esri)
const imagery=L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Imagery &copy; Esri, Maxar, Earthstar Geographics',maxZoom:19,
   crossOrigin:true}).addTo(map);
// keep boundaries/labels ABOVE the (semi-transparent) data layer via a dedicated pane
map.createPane('labels');
map.getPane('labels').style.zIndex=450;
map.getPane('labels').style.pointerEvents='none';
const boundaries=L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  {pane:'labels',attribution:'Boundaries &copy; Esri',maxZoom:19,crossOrigin:true}).addTo(map);
// thin white coastlines (Natural Earth 50m), in their own pane above the data layer
map.createPane('coast');
map.getPane('coast').style.zIndex=440;
map.getPane('coast').style.pointerEvents='none';
fetch('https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_coastline.geojson')
 .then(r=>r.json())
 .then(gj=>L.geoJSON(gj,{pane:'coast',interactive:false,
   style:{color:'#ffffff',weight:0.8,opacity:0.9}}).addTo(map))
 .catch(e=>console.warn('coastline load failed',e));
let layer=null;
function fmt(x){const a=Math.abs(x);
  if(a===0)return '0';
  if(a>=1000||a<0.01)return x.toExponential(1);
  if(a>=100)return x.toFixed(0);
  if(a>=10)return x.toFixed(1);
  return x.toFixed(2);}
function curVar(){return document.getElementById('member').value+'_'+document.getElementById('quant').value;}
const cminB=document.getElementById('cmin_b'), cmaxB=document.getElementById('cmax_b');
let curMin=0, curMax=1;
function fmtBox(x){return (x===0)?0:Number(x.toPrecision(4));}
function updateBar(){
 document.getElementById('cbar').style.background='linear-gradient(to right,'+STOPS.join(',')+')';
 document.getElementById('cmin').textContent=fmt(curMin);
 document.getElementById('cmid').textContent=fmt((curMin+curMax)/2);
 document.getElementById('cmax').textContent=fmt(curMax);
}
function buildLayer(){
 const v=curVar();
 const op=parseFloat(document.getElementById('opacity').value);
 const url='/tiles/WebMercatorQuad/{z}/{y}/{x}?variables='+v+
   '&style=raster/'+CMAP+'&colorscalerange='+curMin+','+curMax+'&width=256&height=256';
 if(layer) map.removeLayer(layer);
 layer=L.tileLayer(url,{tileSize:256,opacity:op,minZoom:0,maxZoom:8,crossOrigin:true,
   attribution:'TRMM PR / GPM Ku precipitation-feature grid'}).addTo(map);
}
const qpop=document.getElementById('qpop');
document.getElementById('qhelp').onclick=function(){qpop.classList.toggle('show');};
function setVar(){
 const m=document.getElementById('member').value, q=document.getElementById('quant').value;
 const v=m+'_'+q, r=RANGES[v]||{vmin:0,vmax:1,units:''};
 document.getElementById('desc').textContent=m+' — '+QLABEL[q];
 document.getElementById('cbunits').textContent=QLABEL[q]+(r.units&&r.units!=='1'&&r.units!=='count'?'  ['+r.units+']':'');
 qpop.innerHTML='<b>'+QLABEL[q]+'</b> — '+LAYMAN[q];   // plain-language explainer
 curMin=r.vmin; curMax=r.vmax;                       // 0.001 / 0.999 quantile defaults
 cminB.value=fmtBox(curMin); cmaxB.value=fmtBox(curMax);
 updateBar(); buildLayer();
}
// editable min/max boxes: validate, then recolor + reload tiles on commit
function applyBoxes(){
 let lo=parseFloat(cminB.value), hi=parseFloat(cmaxB.value);
 if(!isFinite(lo)||!isFinite(hi)||hi<=lo){            // revert on invalid input
   cminB.value=fmtBox(curMin); cmaxB.value=fmtBox(curMax); return;}
 curMin=lo; curMax=hi; updateBar(); buildLayer();
}
cminB.onchange=applyBoxes; cmaxB.onchange=applyBoxes;
document.getElementById('creset').onclick=function(e){e.preventDefault();
 const r=RANGES[curVar()]; curMin=r.vmin; curMax=r.vmax;
 cminB.value=fmtBox(curMin); cmaxB.value=fmtBox(curMax); updateBar(); buildLayer();};
document.getElementById('member').onchange=setVar;
document.getElementById('quant').onchange=setVar;
const opIn=document.getElementById('opacity');
opIn.oninput=function(){document.getElementById('opval').textContent=parseFloat(opIn.value).toFixed(2);
  if(layer)layer.setOpacity(parseFloat(opIn.value));};
// mouseover value readout: debounced point query against the open dataset
const ro=document.getElementById('readout');
let roTimer=null, roSeq=0;
map.on('mousemove', function(e){
 const lat=e.latlng.lat, lon=((e.latlng.lng+180)%360+360)%360-180;
 if(roTimer) clearTimeout(roTimer);
 const seq=++roSeq;
 roTimer=setTimeout(function(){
   fetch('/value?var='+curVar()+'&lat='+lat.toFixed(4)+'&lon='+lon.toFixed(4))
    .then(r=>r.json()).then(d=>{
      if(seq!==roSeq) return;                 // ignore stale responses
      const loc=Math.abs(d.lat!=null?d.lat:lat).toFixed(2)+'°'+(lat>=0?'N':'S')+' '+
                Math.abs(d.lon!=null?d.lon:lon).toFixed(2)+'°'+(lon>=0?'E':'W');
      const lbl='<span style="color:#9ab">Value at cursor:</span> ';
      if(d.value==null||!isFinite(d.value)) ro.innerHTML=lbl+'— &nbsp; <span style="color:#9ab">'+loc+'</span>';
      else ro.innerHTML=lbl+'<b style="color:#8ef">'+fmt(d.value)+'</b> '+(d.units&&d.units!=='1'?d.units:'')+
           ' &nbsp; <span style="color:#9ab">'+loc+'</span>';
    }).catch(()=>{});
 }, 110);
});
map.on('mouseout', function(){ if(roTimer)clearTimeout(roTimer); ++roSeq;
 ro.innerHTML='<span style="color:#9ab">Value at cursor:</span> move cursor over the map'; });
// help modal
const modal=document.getElementById('modal');
document.getElementById('helpbtn').onclick=()=>modal.classList.add('show');
document.getElementById('closex').onclick=()=>modal.classList.remove('show');
modal.onclick=(e)=>{if(e.target===modal)modal.classList.remove('show');};
setVar();
</script></body></html>"""

def _opts(xs, default=None, labels=None):
    labels = labels or {}
    return "".join(
        f"<option value='{x}'" + (" selected" if x == default else "")
        + f">{labels.get(x, x)}</option>" for x in xs)


def _grouped_opts(groups, default=None, labels=None):
    """Like _opts but wraps each (label, items) group in an <optgroup>."""
    return "".join(
        f"<optgroup label='{glabel}'>" + _opts(items, default, labels) + "</optgroup>"
        for glabel, items in groups)


INDEX = (INDEX.replace("__MEMBERS__", _opts(MEMBERS, "COMBINED"))
              .replace("__QUANTS__", _grouped_opts(QGROUPS, "freq", QDISPLAY))  # frequency = default
              .replace("__CMAP__", CMAP)
              .replace("__STOPS__", json.dumps(CMAP_STOPS))
              .replace("__QLABEL__", json.dumps(QLABEL))
              .replace("__LAYMAN__", json.dumps(LAYMAN))
              .replace("__RANGES__", json.dumps(RANGES)))


_LAT0, _LAT1 = float(ds["lat"].min()), float(ds["lat"].max())
_DLAT = abs(float(ds["lat"][1] - ds["lat"][0]))


@app.get("/value")
def value(var: str, lat: float, lon: float):
    """Nearest-cell value of `var` at (lat, lon) for the mouseover readout."""
    if var not in ds.data_vars:
        return {"value": None}
    lon = ((lon + 180.0) % 360.0) - 180.0          # wrap to [-180, 180]
    if lat < _LAT0 - _DLAT or lat > _LAT1 + _DLAT:  # outside the ±68° grid
        return {"value": None}
    try:
        da = ds[var].sel(lat=lat, lon=lon, method="nearest")
        v = float(da.values)
        return {"var": var, "lat": float(da["lat"]), "lon": float(da["lon"]),
                "value": (None if not math.isfinite(v) else v),
                "units": ds[var].attrs.get("units", "")}
    except Exception:
        return {"value": None}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX
