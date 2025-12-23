#!/usr/bin/env python3
# +
# #!/usr/bin/env python3
"""
Monthly → Tile-level Indices (Sentinel-2)

• One image per month
• Compute selected indices (NDVI, NDBI, BSI, MNDWI, SAVI, IBI)
• Aggregate monthly rasters → MEAN + MEDIAN per tile
• Skips months already processed
• Memory-safe aggregation using xarray + dask
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import logging

import xarray as xr
import rioxarray as rxr
from dask.diagnostics import ProgressBar

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("monthly-indices")

# --------------------------------------------------
# Constants
# --------------------------------------------------
MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# --------------------------------------------------
# Index math
# --------------------------------------------------
def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a - b) / (a + b)
        r[~np.isfinite(r)] = np.nan
    return r

def NDVI(b08, b04):
    return safe_div(b08, b04)

def NDBI(b11, b08):
    return safe_div(b11, b08)

def BSI(b11, b04, b08, b02):
    return safe_div((b11 + b04) - (b08 + b02),
                    (b11 + b04) + (b08 + b02))

def MNDWI(b03, b11):
    return safe_div(b03, b11)

def SAVI(b08, b04, L=0.5):
    return (1 + L) * (b08 - b04) / (b08 + b04 + L)

def IBI(ndbi, savi, mndwi):
    return (ndbi - (savi + mndwi) / 2) / (ndbi + (savi + mndwi) / 2)

# --------------------------------------------------
# Raster helpers
# --------------------------------------------------
def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile

def resample(src, src_prof, dst_prof):
    dst = np.empty((dst_prof["height"], dst_prof["width"]), dtype="float32")
    reproject(
        src, dst,
        src_transform=src_prof["transform"],
        src_crs=src_prof["crs"],
        dst_transform=dst_prof["transform"],
        dst_crs=dst_prof["crs"],
        resampling=Resampling.bilinear,
    )
    return dst

def write(path, arr, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="DEFLATE")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)

# --------------------------------------------------
# Helper — check if month already processed
# --------------------------------------------------
def monthly_indices_exist(idx_dir: Path, itemid: str, indices):
    return all(
        (idx_dir / f"{itemid}_{idx}.tif").exists()
        for idx in indices
    )

# --------------------------------------------------
# STEP 1 — Monthly indices
# --------------------------------------------------
def compute_monthly_indices(tile_dir: Path, indices, overwrite=False):

    for month in MONTHS:
        m10 = tile_dir / month / "10m"
        m20 = tile_dir / month / "20m"
        if not m10.exists():
            continue

        idx_dir = tile_dir / month / "indices"
        idx_dir.mkdir(exist_ok=True)

        b08_files = list(m10.glob("*_B08.tif"))
        if not b08_files:
            continue

        itemid = b08_files[0].stem.replace("_B08", "")

        if not overwrite and monthly_indices_exist(idx_dir, itemid, indices):
            log.info("[%s %s] indices exist → skipped", tile_dir.name, month)
            continue

        try:
            # --- Read 10m bands ---
            b08, prof = read_band(m10 / f"{itemid}_B08.tif")  # NIR
            b04, _    = read_band(m10 / f"{itemid}_B04.tif")  # Red
            b03, _    = read_band(m10 / f"{itemid}_B03.tif")  # Green
            b02, _    = read_band(m10 / f"{itemid}_B02.tif")  # Blue

            # --- Read & resample SWIR ---
            b11, p11 = read_band(m20 / f"{itemid}_B11.tif")
            if p11["width"] != prof["width"]:
                b11 = resample(b11, p11, prof)

            # --- Core indices ---
            ndbi = NDBI(b11, b08)
            savi = SAVI(b08, b04)
            mndwi = MNDWI(b03, b11)

            if "NDVI" in indices:
                write(idx_dir / f"{itemid}_NDVI.tif", NDVI(b08, b04), prof)

            if "NDBI" in indices:
                write(idx_dir / f"{itemid}_NDBI.tif", ndbi, prof)

            if "BSI" in indices:
                write(idx_dir / f"{itemid}_BSI.tif", BSI(b11, b04, b08, b02), prof)

            if "MNDWI" in indices:
                write(idx_dir / f"{itemid}_MNDWI.tif", mndwi, prof)

            if "SAVI" in indices:
                write(idx_dir / f"{itemid}_SAVI.tif", savi, prof)

            if "IBI" in indices:
                ibi = IBI(ndbi, savi, mndwi)
                write(idx_dir / f"{itemid}_IBI.tif", ibi, prof)

            log.info("[%s %s] monthly indices processed", tile_dir.name, month)

        except Exception as e:
            log.warning("[%s %s] skipped: %s", tile_dir.name, month, e)

# --------------------------------------------------
# STEP 2 — Tile aggregation (xarray + dask)
# --------------------------------------------------
def aggregate_tile(tile_dir: Path, indices, do_mean=True, do_median=True):

    outdir = tile_dir / "indices"
    outdir.mkdir(exist_ok=True)

    for idx in indices:
        rasters = sorted(tile_dir.glob(f"*/indices/*_{idx}.tif"))
        if not rasters:
            continue

        da_list = [
            rxr.open_rasterio(r, chunks={"x": 512, "y": 512}).squeeze()
            for r in rasters
        ]

        stack = xr.concat(da_list, dim="time")

        with ProgressBar():
            if do_mean:
                mean = stack.mean(dim="time", skipna=True)
                mean.rio.to_raster(outdir / f"{tile_dir.name}_MEAN_{idx}.tif",
                                   compress="DEFLATE")

            if do_median:
                median = stack.median(dim="time", skipna=True)
                median.rio.to_raster(outdir / f"{tile_dir.name}_MEDIAN_{idx}.tif",
                                     compress="DEFLATE")

        del stack, da_list
        log.info("[%s] aggregated %s", tile_dir.name, idx)

# --------------------------------------------------
# RUN
# --------------------------------------------------
def run(
    root="data/sentinel",
    tiles=None,
    indices=("NDVI","NDBI","BSI","MNDWI","SAVI","IBI"),
    aggregate_mean=True,
    aggregate_median=True,
    overwrite_monthly=False,
):
    root = Path(root)
    tile_dirs = [p for p in root.iterdir() if p.is_dir()]

    if tiles:
        tile_dirs = [t for t in tile_dirs if t.name in set(tiles)]

    for tile in tile_dirs:
        log.info("Processing tile %s", tile.name)
        compute_monthly_indices(tile, indices, overwrite=overwrite_monthly)
        aggregate_tile(tile, indices, aggregate_mean, aggregate_median)

if __name__ == "__main__":
    run()
# -


