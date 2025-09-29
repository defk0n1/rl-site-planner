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





# -----------------------
# CONFIG / INPUTS
# -----------------------
INDEX_TXT = "index_clutter.txt"     # whitespace-delimited: filename min_x max_x min_y max_y res
KMZ_PATH  = "Untitled project (2).kml"
# KMZ_PATH = "sq_centroid_kmz/libya_tile_square_centroid_155_424.kmz"
EPSG_CODE = "EPSG:32633"    # from projection.txt (UTM zone 33N)
DTYPE_BIN = np.uint16       #  tiles dtype

# -----------------------
# HELPERS
# -----------------------
def load_polygon_from_kmz_or_kml(path):
    # return single shapely polygon in target CRS
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
    # single polygon expected
    return gdf.geometry.iloc[0]

# -----------------------
# 1) read polygon
# -----------------------
poly = load_polygon_from_kmz_or_kml(KMZ_PATH)
poly_minx, poly_miny, poly_maxx, poly_maxy = poly.bounds

print(poly.bounds)

# -----------------------
# 2) read index txt
# -----------------------
idx = pd.read_csv(
    INDEX_TXT,
    sep=r"\s+",
    names=["tile_path", "min_x", "max_x", "min_y", "max_y", "res"],
    dtype={"tile_path": str}
)

print(idx)
# ensure numeric columns
for c in ["min_x","max_x","min_y","max_y","res"]:
    idx[c] = idx[c].astype(float)

# -----------------------
# 3) select tiles whose bbox intersects polygon bbox (fast filter)
# -----------------------
def bbox_intersects_tile(tile_row, poly_bounds):
    tminx, tminy, tmaxx, tmaxy = tile_row["min_x"], tile_row["min_y"], tile_row["max_x"], tile_row["max_y"]
    pminx, pminy, pmaxx, pmaxy = poly_bounds
    # No intersection if one is completely to one side of the other
    return not (tmaxx <= pminx or tminx >= pmaxx or tmaxy <= pminy or tminy >= pmaxy)

poly_bounds = (poly_minx, poly_miny, poly_maxx, poly_maxy)
candidates = idx[idx.apply(bbox_intersects_tile, axis=1, poly_bounds=poly_bounds)].reset_index(drop=True)
if candidates.empty:
    raise RuntimeError("No tiles intersect polygon bounding box")

# -----------------------
# 4) iterate candidates, mask & crop, accumulate placement metadata
# -----------------------
# We'll store: arr_crop (float32), rmin, cmin, rows, cols, tile_min_x, tile_max_y, res
pieces = []
res_global = None  # all tiles should have same res; sanity check

for _, r in candidates.iterrows():
    tile_path = r["tile_path"]
    min_x, max_x = float(r["min_x"]), float(r["max_x"])
    min_y, max_y = float(r["min_y"]), float(r["max_y"])
    res = float(r["res"])
    if res_global is None:
        res_global = res
    elif abs(res_global - res) > 1e-9:
        raise RuntimeError("Tiles have differing resolutions; mosaic expects uniform res")

    # compute rows/cols for tile
    cols = int(round((max_x - min_x) / res))
    rows = int(round((max_y - min_y) / res))
    if rows <= 0 or cols <= 0:
        continue

    # quick footprint check with shapely: skip if no real intersection
    tile_box = box(min_x, min_y, max_x, max_y)
    if not poly.intersects(tile_box):
        continue

    # load binary tile (uint16) and reshape
    arr = np.fromfile(tile_path, dtype=DTYPE_BIN)
    if arr.size != rows * cols:
        raise RuntimeError(f"Tile size mismatch: {tile_path} expected {rows*cols} got {arr.size}")
    arr = arr.reshape((rows, cols))

    # build transform and mask polygon-to-tile
    transform = from_origin(min_x, max_y, res, res)
    sub_poly = poly.intersection(tile_box)
    if sub_poly.is_empty:
        del arr
        continue

    # geometry in geojson mapping for rasterio.features.geometry_mask
    geom = json.loads(gpd.GeoSeries([sub_poly], crs=EPSG_CODE).to_json())['features'][0]['geometry']
    mask = geometry_mask([geom], out_shape=(rows, cols), transform=transform, invert=True)

    # If no overlapping pixels, free and continue
    if not mask.any():
        del arr, mask
        continue

    # convert to float32 so we can use NaN for outside values
    arr = arr.astype(np.float32)
    arr_masked = np.where(mask, arr, 0)

    # tight crop to mask bbox
    ys, xs = np.where(mask)
    rmin, rmax = ys.min(), ys.max()
    cmin, cmax = xs.min(), xs.max()
    arr_crop = arr_masked[rmin:rmax+1, cmin:cmax+1]

    # keep metadata needed to place this piece into global mosaic
    piece_meta = {
        "arr_crop": arr_crop,           # float32 array
        "rmin": rmin, "cmin": cmin,     # offsets within tile
        "rows": rows, "cols": cols,
        "tile_min_x": min_x,
        "tile_max_y": max_y,
        "res": res
    }
    pieces.append(piece_meta)

    # free intermediates 
    del arr, arr_masked, mask

if not pieces:
    raise RuntimeError("No overlapping pixels found after precise clipping")

# -----------------------
# 5) build global mosaic canvas (pure NumPy)
# -----------------------
# Compute global mosaic georeference:
# left = min(tile_min_x), top = max(tile_max_y)
global_min_x = min(p["tile_min_x"] for p in pieces)
global_max_y = max(p["tile_max_y"] for p in pieces)
res = res_global

# compute mosaic width/height in pixels: find rightmost and bottommost extents (in meters)
# compute tile-level extents in pixel coordinates relative to global origin
rightmost = 0.0
bottommost = 0.0
for p in pieces:
    t_minx = p["tile_min_x"]
    t_maxy = p["tile_max_y"]
    cols = p["cols"]
    rows = p["rows"]
    # tile origin in pixels:
    col_offset = int(round((t_minx - global_min_x) / res))
    row_offset = int(round((global_max_y - t_maxy) / res))
    rightmost = max(rightmost, col_offset + cols)
    bottommost = max(bottommost, row_offset + rows)

mosaic_width = int(rightmost)
mosaic_height = int(bottommost)

# allocate final canvas as float32 with NaN
mosaic = np.full((mosaic_height, mosaic_width), 0, dtype=np.float32)

# paste each cropped piece into mosaic
for p in pieces:
    arr_crop = p["arr_crop"]
    rmin, cmin = p["rmin"], p["cmin"]
    cols = p["cols"]
    rows = p["rows"]
    t_minx = p["tile_min_x"]
    t_maxy = p["tile_max_y"]

    # tile origin in mosaic pixel coords
    col_offset = int(round((t_minx - global_min_x) / res))
    row_offset = int(round((global_max_y - t_maxy) / res))

    # top-left location in mosaic where arr_crop should be pasted
    paste_row = row_offset + rmin
    paste_col = col_offset + cmin

    h, w = arr_crop.shape
    mosaic[paste_row:paste_row+h, paste_col:paste_col+w] = arr_crop

    # free arr_crop after pasted (optional)
    del p["arr_crop"]

# mosaic transform (affine): top-left = (global_min_x, global_max_y)
mosaic_transform = from_origin(global_min_x, global_max_y, res, res)

# final result: 'mosaic' (2D float32 numpy) and 'mosaic_transform' (rasterio transform)
print("Mosaic shape (rows, cols):", mosaic.shape)
print("Mosaic transform:", mosaic_transform)
print(f"Unique values in mosaic : {np.unique(mosaic)}")

# -----------------------
# CROP MOSAIC TO POLYGON BOUNDS
# -----------------------

# convert bounds to pixel indices
row_min = int(np.floor((global_max_y - poly_maxy) / res))
row_max = int(np.ceil((global_max_y - poly_miny) / res))
col_min = int(np.floor((poly_minx - global_min_x) / res))
col_max = int(np.ceil((poly_maxx - global_min_x) / res))

# clamp indices to mosaic dimensions
row_min = max(0, row_min)
col_min = max(0, col_min)
row_max = min(mosaic.shape[0], row_max)
col_max = min(mosaic.shape[1], col_max)

# crop the mosaic
mosaic_cropped = mosaic[row_min:row_max, col_min:col_max]

# update the transform (top-left corner shifts after cropping)
mosaic_transform_cropped = from_origin(
    global_min_x + col_min * res,
    global_max_y - row_min * res,
    res,
    res
)


print("Cropped mosaic transform:" , mosaic_transform_cropped)

print("Cropped mosaic shape:", mosaic_cropped.shape)

# original_values = np.array([
#     256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
#     2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 5120, 61912
# ])
# new_classes = np.where(original_values == 61912, 0, original_values // 256)
# mapping = dict(zip(original_values, new_classes))
# raster_mapped = np.vectorize(mapping.get)(mosaic)

# print(mosaic_cropped.shape , mosaic_cropped.dtype , mosaic_cropped)

# np.save(f"{KMZ_PATH}.npy", mosaic_cropped)

# print("Raster loaded successfully!")

# # Save transform in the same folder with .pkl extension
# transform_path = os.path.join(OUPUT_PATH, "clutter_map_transform.pkl")
# with open(transform_path, "wb") as f:
#     pickle.dump(mosaic_transform_cropped, f)

# print(f"Saved clutter map to {OUPUT_PATH}")
# print(f"Saved transform to {transform_path}")

output_dir = "polygon_data"
kml_name = os.path.splitext(os.path.basename(KMZ_PATH))[0]  # e.g. "myfile"
save_dir = os.path.join(output_dir, kml_name)
os.makedirs(save_dir, exist_ok=True)


output_dir = os.path.join("polygon_data", os.path.splitext(KMZ_PATH)[0])
os.makedirs(output_dir, exist_ok=True)

# Save numpy array
np.save(os.path.join(output_dir, "mosaic.npy"), mosaic_cropped)

# Save transform + metadata
metadata = {
    "transform": mosaic_transform_cropped.to_gdal(),  # tuple of 6
    "shape" : mosaic_cropped.shape
}

with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)