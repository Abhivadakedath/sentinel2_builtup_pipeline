#!/usr/bin/env python3
# +
# #!/usr/bin/env python3
"""
Monthly → Tile Mean Indices (Sentinel-2)

• One image per month
• Compute selected indices
• Aggregate monthly rasters → MEAN per tile
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("monthly-indices")

MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# --------------------------------------------------
# Index functions
# --------------------------------------------------
def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a - b) / (a + b)
        r[~np.isfinite(r)] = np.nan
    return r

def NDVI(b08, b04): return safe_div(b08, b04)
def NDBI(b11, b08): return safe_div(b11, b08)
def BSI(b11, b04, b08, b02):
    return safe_div((b11 + b04) - (b08 + b02), (b11 + b04) + (b08 + b02))
def MNDWI(b03, b11): return safe_div(b03, b11)

# --------------------------------------------------
# Helpers
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
# STEP 1 — MONTHLY INDICES
# --------------------------------------------------
def compute_monthly_indices(tile_dir: Path, indices: list[str]):
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

        b08_path = b08_files[0]
        itemid = b08_path.stem.replace("_B08", "")

        try:
            b08, prof = read_band(b08_path)
            b04, _ = read_band(m10 / f"{itemid}_B04.tif")

            if "NDVI" in indices:
                write(idx_dir / f"{itemid}_NDVI.tif", NDVI(b08, b04), prof)

            if "NDBI" in indices or "BSI" in indices or "MNDWI" in indices:
                b11, p11 = read_band(m20 / f"{itemid}_B11.tif")
                if p11["width"] != prof["width"]:
                    b11 = resample(b11, p11, prof)

            if "NDBI" in indices:
                write(idx_dir / f"{itemid}_NDBI.tif", NDBI(b11, b08), prof)

            if "BSI" in indices:
                b02, _ = read_band(m10 / f"{itemid}_B02.tif")
                write(idx_dir / f"{itemid}_BSI.tif", BSI(b11, b04, b08, b02), prof)

            if "MNDWI" in indices:
                b03, _ = read_band(m10 / f"{itemid}_B03.tif")
                write(idx_dir / f"{itemid}_MNDWI.tif", MNDWI(b03, b11), prof)

            log.info("[%s %s] %s written", tile_dir.name, month, indices)

        except Exception as e:
            log.warning("[%s %s] failed: %s", tile_dir.name, month, e)

# --------------------------------------------------
# STEP 2 — TILE AGGREGATION (MEAN / MEDIAN CONTROL)
# --------------------------------------------------
def aggregate_tile(
    tile_dir: Path,
    indices: list[str],
    do_mean: bool = True,
    do_median: bool = False,
):
    outdir = tile_dir / "indices"
    outdir.mkdir(exist_ok=True)

    for idx in indices:
        rasters = list(tile_dir.glob(f"*/indices/*_{idx}.tif"))
        if not rasters:
            continue

        stack = []
        prof = None
        for r in rasters:
            arr, prof = read_band(r)
            stack.append(arr)

        data = np.stack(stack)

        if do_mean:
            mean = np.nanmean(data, axis=0)
            write(
                outdir / f"{tile_dir.name}_MEAN_{idx}.tif",
                mean,
                prof,
            )
            log.info("[%s] MEAN_%s written", tile_dir.name, idx)

        if do_median:
            median = np.nanmedian(data, axis=0)
            write(
                outdir / f"{tile_dir.name}_MEDIAN_{idx}.tif",
                median,
                prof,
            )
            log.info("[%s] MEDIAN_%s written", tile_dir.name, idx)

# --------------------------------------------------
# RUN
# --------------------------------------------------
def run(
    root="data/sentinel",
    tiles=None,
    indices=("NDVI","NDBI","BSI","MNDWI"),
    aggregate_mean=True,
    aggregate_median=False,
):
    root = Path(root)
    tile_dirs = [p for p in root.iterdir() if p.is_dir()]

    if tiles:
        tile_dirs = [t for t in tile_dirs if t.name in set(tiles)]

    for tile in tile_dirs:
        log.info("Processing tile %s", tile.name)

        # STEP 1: monthly indices
        compute_monthly_indices(tile, list(indices))

        # STEP 2: aggregation
        aggregate_tile(
            tile,
            list(indices),
            do_mean=aggregate_mean,
            do_median=aggregate_median,
        )
# -


