import os
import zipfile
import tempfile
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from tqdm import tqdm   # NEW: for progress bar


# -----------------------
# CONFIG / INPUTS
# -----------------------
INDEX_TXT = "index_clutter.txt"     # whitespace-delimited: filename min_x max_x min_y max_y res
KMZ_DIR = "sq_centroid_kmz"         # directory containing KMZ/KML files
EPSG_CODE = "EPSG:32633"            # from projection.txt (UTM zone 33N)
DTYPE_BIN = np.uint16               # tiles dtype
OUTPUT_BASE = "polygon_data"        # base output directory


# -----------------------
# HELPERS
# -----------------------
def load_polygon_from_kmz_or_kml(path):
    """Return a single shapely polygon in target CRS"""
    if path.lower().endswith(".kmz"):
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmpdir)
        kml = None
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.lower().endswith(".kml"):
                    kml = os.path.join(root, f)
                    break
            if kml:
                break
        if kml is None:
            raise RuntimeError("No .kml found inside .kmz")
        gdf = gpd.read_file(kml, driver="KML")
    else:
        gdf = gpd.read_file(path)  # KML/GeoJSON/Shapefile etc.

    if gdf.empty:
        raise RuntimeError("Polygon layer empty")
    gdf = gdf.to_crs(EPSG_CODE)
    return gdf.geometry.iloc[0]  # single polygon expected


def process_kmz(kmz_path):
    """Process a single KMZ/KML file and save cropped mosaic + metadata"""
    kmz_name = os.path.splitext(os.path.basename(kmz_path))[0]
    save_dir = os.path.join(OUTPUT_BASE, kmz_name)

    # ✅ Skip if already processed
    if os.path.exists(os.path.join(save_dir, "mosaic.npy")) and os.path.exists(os.path.join(save_dir, "metadata.json")):
        print(f"⏩ Skipping {kmz_name}, already processed")
        return

    print(f"\n=== Processing {kmz_path} ===")

    # -----------------------
    # 1) read polygon
    # -----------------------
    poly = load_polygon_from_kmz_or_kml(kmz_path)
    poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds

    # -----------------------
    # 2) read index txt
    # -----------------------
    idx = pd.read_csv(
        INDEX_TXT,
        sep=r"\s+",
        names=["tile_path", "min_x", "max_x", "min_y", "max_y", "res"],
        dtype={"tile_path": str}
    )
    for c in ["min_x", "max_x", "min_y", "max_y", "res"]:
        idx[c] = idx[c].astype(float)

    # -----------------------
    # 3) filter candidate tiles
    # -----------------------
    def bbox_intersects_tile(tile_row, poly_bounds):
        tminx, tminy, tmaxx, tmaxy = (
            tile_row["min_x"], tile_row["min_y"], tile_row["max_x"], tile_row["max_y"]
        )
        pminx, pminy, pmaxx, pmaxy = poly_bounds
        return not (tmaxx <= pminx or tminx >= pmaxx or tmaxy <= pminy or tminy >= pmaxy)

    candidates = idx[idx.apply(
        bbox_intersects_tile,
        axis=1,
        poly_bounds=(poly_minx, poly_miny, poly_maxx, poly_maxy)
    )].reset_index(drop=True)

    if candidates.empty:
        print("⚠️  No tiles intersect polygon bounding box")
        return

    # -----------------------
    # 4–5) build mosaic
    # -----------------------
    pieces = []
    res_global = None

    for _, r in candidates.iterrows():
        tile_path = r["tile_path"]
        min_x, max_x = float(r["min_x"]), float(r["max_x"])
        min_y, max_y = float(r["min_y"]), float(r["max_y"])
        res = float(r["res"])
        if res_global is None:
            res_global = res
        elif abs(res_global - res) > 1e-9:
            raise RuntimeError("Tiles have differing resolutions")

        cols = int(round((max_x - min_x) / res))
        rows = int(round((max_y - min_y) / res))
        if rows <= 0 or cols <= 0:
            continue

        tile_box = box(min_x, min_y, max_x, max_y)
        if not poly.intersects(tile_box):
            continue

        arr = np.fromfile(tile_path, dtype=DTYPE_BIN)
        if arr.size != rows * cols:
            raise RuntimeError(f"Tile size mismatch: {tile_path}")
        arr = arr.reshape((rows, cols))

        transform = from_origin(min_x, max_y, res, res)
        sub_poly = poly.intersection(tile_box)
        if sub_poly.is_empty:
            continue

        geom = json.loads(gpd.GeoSeries([sub_poly], crs=EPSG_CODE).to_json())['features'][0]['geometry']
        mask = geometry_mask([geom], out_shape=(rows, cols), transform=transform, invert=True)

        if not mask.any():
            continue

        arr = arr.astype(np.float32)
        arr_masked = np.where(mask, arr, 0)

        ys, xs = np.where(mask)
        rmin, rmax = ys.min(), ys.max()
        cmin, cmax = xs.min(), xs.max()
        arr_crop = arr_masked[rmin:rmax+1, cmin:cmax+1]

        pieces.append({
            "arr_crop": arr_crop,
            "rmin": rmin, "cmin": cmin,
            "rows": rows, "cols": cols,
            "tile_min_x": min_x,
            "tile_max_y": max_y,
            "res": res
        })

    if not pieces:
        print("⚠️  No overlapping pixels found")
        return

    global_min_x = min(p["tile_min_x"] for p in pieces)
    global_max_y = max(p["tile_max_y"] for p in pieces)
    res = res_global

    rightmost, bottommost = 0.0, 0.0
    for p in pieces:
        col_offset = int(round((p["tile_min_x"] - global_min_x) / res))
        row_offset = int(round((global_max_y - p["tile_max_y"]) / res))
        rightmost = max(rightmost, col_offset + p["cols"])
        bottommost = max(bottommost, row_offset + p["rows"])

    mosaic = np.full((int(bottommost), int(rightmost)), 0, dtype=np.float32)

    for p in pieces:
        arr_crop = p["arr_crop"]
        paste_row = int(round((global_max_y - p["tile_max_y"]) / res)) + p["rmin"]
        paste_col = int(round((p["tile_min_x"] - global_min_x) / res)) + p["cmin"]
        h, w = arr_crop.shape
        mosaic[paste_row:paste_row+h, paste_col:paste_col+w] = arr_crop

    mosaic_transform = from_origin(global_min_x, global_max_y, res, res)

    row_min = int(np.floor((global_max_y - poly_maxy) / res))
    row_max = int(np.ceil((global_max_y - poly_miny) / res))
    col_min = int(np.floor((poly_minx - global_min_x) / res))
    col_max = int(np.ceil((poly_maxx - global_min_x) / res))

    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(mosaic.shape[0], row_max)
    col_max = min(mosaic.shape[1], col_max)

    mosaic_cropped = mosaic[row_min:row_max, col_min:col_max]
    mosaic_transform_cropped = from_origin(
        global_min_x + col_min * res,
        global_max_y - row_min * res,
        res, res
    )

    if len(np.unique(mosaic_cropped)) < 3 :
        print(f"  ⚠️ Clutter map too simple for {kmz_name}")
        return

    # -----------------------
    # SAVE RESULTS
    # -----------------------
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "mosaic.npy"), mosaic_cropped)

    metadata = {
        "transform": mosaic_transform_cropped.to_gdal(),
        "shape": mosaic_cropped.shape
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Saved results for {kmz_name} -> {save_dir}")


# -----------------------
# MAIN LOOP
# -----------------------
if __name__ == "__main__":
    kmz_files = [os.path.join(KMZ_DIR, f) for f in os.listdir(KMZ_DIR) if f.lower().endswith((".kmz", ".kml"))]

    if not kmz_files:
        print("⚠️  No KMZ/KML files found in", KMZ_DIR)
    else:
        print(f"🔍 Found {len(kmz_files)} KMZ/KML files in {KMZ_DIR}")
        for kmz in tqdm(kmz_files, desc="Processing projects", unit="file"):
            try:
                process_kmz(kmz)
            except Exception as e:
                print(f"❌ Failed for {kmz}: {e}")
