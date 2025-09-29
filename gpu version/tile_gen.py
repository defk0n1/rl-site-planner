#!/usr/bin/env python3
"""
Generate a tiling grid across Libya and export each tile as its own KML/KMZ.

Updates:
- Fixed Libya extraction with better name matching and error handling
- Added debug output to help diagnose data issues

Dependencies:
  pip install geopandas shapely pyproj simplekml requests
"""

import argparse
import math
import os
import sys
import tempfile
import zipfile
from typing import Iterable, List, Tuple, Union, Optional

import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer
import simplekml
import requests


Geometry = Union[Polygon, MultiPolygon]

# Primary URL (as requested), plus fallback(s)
NE_ADMIN0_PRIMARY_URL = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"
NE_ADMIN0_FALLBACK_URLS = [
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
]
NE_110M_SHP_NAME = "ne_110m_admin_0_countries.shp"


def download_natural_earth_admin0_110m() -> str:
    """
    Download Natural Earth 110m Admin 0 Countries ZIP (primary + fallback) and extract.
    Returns the path to the extracted shapefile (.shp).
    """
    urls = [NE_ADMIN0_PRIMARY_URL] + NE_ADMIN0_FALLBACK_URLS
    last_error: Optional[Exception] = None
    tmpdir = tempfile.mkdtemp(prefix="ne_admin0_")
    zpath = os.path.join(tmpdir, "ne_110m_admin_0_countries.zip")

    headers = {"User-Agent": "libya-grid-kml/1.0 (+https://github.com/)"}
    for url in urls:
        try:
            clean_url = url.rstrip("*")  # tolerate accidental trailing '*'
            print(f"Attempting to download from: {clean_url}")
            with requests.get(clean_url, stream=True, timeout=120, headers=headers) as r:
                r.raise_for_status()
                with open(zpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            # Extract
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(tmpdir)
            # Locate shapefile
            shp_path = os.path.join(tmpdir, NE_110M_SHP_NAME)
            if not os.path.exists(shp_path):
                # Fallback: find any .shp inside
                for root, _, files in os.walk(tmpdir):
                    for fn in files:
                        if fn.lower().endswith(".shp"):
                            shp_path = os.path.join(root, fn)
                            break
                    if os.path.exists(shp_path):
                        break
            if not os.path.exists(shp_path):
                raise RuntimeError("Extracted Natural Earth archive but could not find a .shp file.")
            print(f"Successfully downloaded and extracted to: {shp_path}")
            return shp_path
        except Exception as e:
            print(f"Failed to download from {clean_url}: {e}")
            last_error = e
            continue
    raise RuntimeError(f"Failed to download or extract Natural Earth dataset from all sources. Last error: {last_error}") from last_error


def extract_libya_from_gdf(gdf: gpd.GeoDataFrame) -> Geometry:
    """
    Given a GeoDataFrame of countries or a Libya-only boundary, return Libya geometry in EPSG:4326.
    Tries multiple name variations and field names.
    """
    print(f"Dataset contains {len(gdf)} features")
    print(f"Available columns: {list(gdf.columns)}")
    
    # Print first few country names for debugging
    name_cols = [col for col in gdf.columns if any(name in col.lower() for name in ['name', 'admin', 'sovereign'])]
    if name_cols:
        print(f"Sample country names from column '{name_cols[0]}':")
        sample_names = gdf[name_cols[0]].head(10).tolist()
        for name in sample_names:
            print(f"  - {name}")
    
    # Try multiple Libya name variations
    libya_names = ['libya', 'libyan arab jamahiriya', 'great socialist people\'s libyan arab jamahiriya', 'state of libya']
    
    # Try multiple column names that might contain country names
    potential_name_cols = []
    for col in gdf.columns:
        col_lower = col.lower()
        if any(name_part in col_lower for name_part in ['name', 'admin', 'sovereign', 'country']):
            potential_name_cols.append(col)
    
    libya_gdf = None
    matched_name = None
    matched_col = None
    
    for col in potential_name_cols:
        print(f"Checking column: {col}")
        try:
            for libya_name in libya_names:
                mask = gdf[col].astype(str).str.lower().str.contains(libya_name.split()[0], na=False)
                subset = gdf.loc[mask]
                if not subset.empty:
                    print(f"Found Libya match: '{libya_name}' in column '{col}'")
                    libya_gdf = subset
                    matched_name = libya_name
                    matched_col = col
                    break
            if libya_gdf is not None:
                break
        except Exception as e:
            print(f"Error checking column {col}: {e}")
            continue
    
    if libya_gdf is None:
        print("No Libya found by name matching. Available countries:")
        for col in potential_name_cols[:1]:  # Just show one column to avoid spam
            try:
                unique_names = gdf[col].dropna().unique()
                for name in sorted(unique_names):
                    print(f"  - {name}")
                break
            except Exception:
                continue
        
        # If no name field matched, assume the file is already Libya-only
        print("Assuming the entire dataset represents Libya...")
        libya_gdf = gdf
    else:
        print(f"Successfully matched Libya as '{matched_name}' in column '{matched_col}'")

    # Ensure CRS is set
    if libya_gdf.crs is None:
        print("No CRS found, assuming EPSG:4326")
        libya_gdf = libya_gdf.set_crs(4326, allow_override=True)
    else:
        print(f"Original CRS: {libya_gdf.crs}")
        libya_gdf = libya_gdf.to_crs(4326)
    
    geom = libya_gdf.unary_union
    if geom is None or geom.is_empty:
        raise RuntimeError("Could not obtain Libya geometry from the provided dataset.")
    
    print(f"Libya geometry type: {type(geom)}")
    print(f"Libya bounds: {geom.bounds}")
    return geom


def load_libya_wgs84(boundary: Optional[str]) -> Geometry:
    """
    Load Libya boundary in EPSG:4326.
    - If boundary is provided (path or URL), read it via GeoPandas and extract Libya if needed.
    - Else, download Natural Earth 110m admin0 and extract Libya.
    """
    if boundary:
        print(f"Loading boundary from: {boundary}")
        # If it's a remote ZIP (likely a shapefile), download then read
        if boundary.lower().endswith(".zip") and (boundary.startswith("http://") or boundary.startswith("https://")):
            tmpdir = tempfile.mkdtemp(prefix="boundary_zip_")
            zpath = os.path.join(tmpdir, "boundary.zip")
            try:
                with requests.get(boundary.rstrip("*"), stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(zpath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(tmpdir)
                # Find a shapefile inside
                shp_path = None
                for root, _, files in os.walk(tmpdir):
                    for fn in files:
                        if fn.lower().endswith(".shp"):
                            shp_path = os.path.join(root, fn)
                            break
                    if shp_path:
                        break
                if not shp_path:
                    raise RuntimeError("No .shp found in boundary ZIP.")
                gdf = gpd.read_file(shp_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read boundary ZIP from URL: {e}") from e
        else:
            # Let GeoPandas handle local path or direct GeoJSON/GeoPackage URL
            gdf = gpd.read_file(boundary)
        return extract_libya_from_gdf(gdf)

    # Fallback: download Natural Earth and extract Libya
    print("No boundary provided, downloading Natural Earth data...")
    shp_path = download_natural_earth_admin0_110m()
    gdf = gpd.read_file(shp_path)
    return extract_libya_from_gdf(gdf)


def make_laea_crs_for_geom_center(geom_wgs84: Geometry) -> CRS:
    if geom_wgs84 is None:
        raise RuntimeError("Cannot create CRS for None geometry")
    
    centroid = gpd.GeoSeries([geom_wgs84], crs=4326).centroid.iloc[0]
    lon0, lat0 = float(centroid.x), float(centroid.y)
    print(f"Libya centroid: {lon0:.4f}, {lat0:.4f}")
    proj4 = f"+proj=laea +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    return CRS.from_proj4(proj4)


def square_side_from_area_m2(area_m2: float) -> float:
    return math.sqrt(area_m2)


def hex_side_from_area_m2(area_m2: float) -> float:
    # Regular hexagon area: A = (3*sqrt(3)/2) * a^2
    return math.sqrt((2.0 * area_m2) / (3.0 * math.sqrt(3.0)))


def build_square(x: float, y: float, side: float) -> Polygon:
    return Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


def build_flat_topped_hex(cx: float, cy: float, a: float) -> Polygon:
    # Flat-topped hex: vertices at angles 0, 60, 120, 180, 240, 300 deg
    verts = []
    for k in range(6):
        ang = math.radians(60.0 * k)
        verts.append((cx + a * math.cos(ang), cy + a * math.sin(ang)))
    return Polygon(verts)


def generate_square_grid(cover_geom: Geometry, side: float) -> Iterable[Tuple[Polygon, Tuple[int, int]]]:
    minx, miny, maxx, maxy = cover_geom.bounds
    startx = math.floor((minx - side) / side) * side
    starty = math.floor((miny - side) / side) * side
    ix = 0
    x = startx
    while x <= maxx + side:
        iy = 0
        y = starty
        while y <= maxy + side:
            yield build_square(x, y, side), (ix, iy)
            iy += 1
            y += side
        ix += 1
        x += side


def generate_hex_grid(cover_geom: Geometry, a: float) -> Iterable[Tuple[Polygon, Tuple[int, int]]]:
    width = 2.0 * a
    height = math.sqrt(3.0) * a
    xstep = 1.5 * a
    ystep = height

    minx, miny, maxx, maxy = cover_geom.bounds
    startx = minx - width
    endx = maxx + width

    ix = 0
    x = startx
    while x <= endx:
        y_offset = 0.0 if (ix % 2 == 0) else (height / 2.0)
        starty = miny - height
        endy = maxy + height
        iy = 0
        y = starty + y_offset
        while y <= endy:
            yield build_flat_topped_hex(x, y, a), (ix, iy)
            iy += 1
            y += ystep
        ix += 1
        x += xstep


def filter_and_shape_tiles(
    tiles: Iterable[Tuple[Polygon, Tuple[int, int]]],
    libya_laea: Geometry,
    mode: str,
) -> List[Tuple[Geometry, Tuple[int, int]]]:
    shaped: List[Tuple[Geometry, Tuple[int, int]]] = []
    for tile, idx in tiles:
        if mode == "within":
            if tile.within(libya_laea):
                shaped.append((tile, idx))
        elif mode == "centroid":
            if tile.centroid.within(libya_laea):
                shaped.append((tile, idx))
        else:  # clip (default)
            if tile.intersects(libya_laea):
                inter = tile.intersection(libya_laea)
                if not inter.is_empty and inter.area > 0:
                    shaped.append((inter, idx))
    return shaped


def save_geom_kml(geom_wgs84: Geometry, filepath: str, name: str, description: str):
    kml = simplekml.Kml()
    def add_polygon(poly: Polygon):
        exterior = [(float(x), float(y)) for x, y in poly.exterior.coords]
        inner_boundaries = []
        for interior in poly.interiors:
            inner_boundaries.append([(float(x), float(y)) for x, y in interior.coords])
        pol = kml.newpolygon(name=name, description=description,
                             outerboundaryis=exterior,
                             innerboundaryis=inner_boundaries if inner_boundaries else None)
        pol.style.linestyle.width = 2
        pol.style.linestyle.color = simplekml.Color.red
        pol.style.polystyle.color = simplekml.Color.changealphaint(80, simplekml.Color.yellow)
        pol.extrude = 0
        pol.tessellate = 1
        pol.altitudemode = simplekml.AltitudeMode.clamptoground

    if isinstance(geom_wgs84, Polygon):
        add_polygon(geom_wgs84)
    elif isinstance(geom_wgs84, MultiPolygon):
        for poly in geom_wgs84.geoms:
            add_polygon(poly)
    else:
        try:
            for poly in geom_wgs84.geoms:  # type: ignore[attr-defined]
                if isinstance(poly, Polygon):
                    add_polygon(poly)
        except Exception:
            pass

    kml.save(filepath)


def main():
    parser = argparse.ArgumentParser(description="Generate a tiling grid across Libya and export each tile as its own KML/KMZ.")
    parser.add_argument("--tile-area-km2", type=float, required=True, help="Target area per tile in km^2 (nominal).")
    parser.add_argument("--grid", choices=["square", "hex"], default="square", help="Grid type (default: square).")
    parser.add_argument("--mode", choices=["clip", "within", "centroid"], default="clip",
                        help="Tile selection mode: clip|within|centroid (default: clip).")
    parser.add_argument("--outdir", type=str, default="tiles_kml", help="Output directory for KML/KMZ files.")
    parser.add_argument("--basename", type=str, default="libya_tile", help="Base name for output files.")
    parser.add_argument("--save-kmz", action="store_true", help="Save KMZ instead of KML.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of tiles to export.")
    parser.add_argument("--boundary", type=str, default=None,
                        help="Path or URL to a boundary dataset (GeoJSON/GPKG/Shapefile). If omitted, Natural Earth is used.")
    args = parser.parse_args()

    if args.tile_area_km2 <= 0:
        raise SystemExit("tile-area-km2 must be > 0")

    os.makedirs(args.outdir, exist_ok=True)

    try:
        # Load Libya and build equal-area projection
        libya_wgs84 = load_libya_wgs84(args.boundary)
        laea_crs = make_laea_crs_for_geom_center(libya_wgs84)
        to_laea = Transformer.from_crs(4326, laea_crs, always_xy=True).transform
        to_wgs84 = Transformer.from_crs(laea_crs, 4326, always_xy=True).transform
        libya_laea = shp_transform(to_laea, libya_wgs84)

        area_m2 = args.tile_area_km2 * 1_000_000.0

        # Generate raw grid tiles in LAEA
        if args.grid == "square":
            side = square_side_from_area_m2(area_m2)
            print(f"Square side length: {side:.2f} meters")
            tiles_iter = generate_square_grid(libya_laea, side)
        else:
            a = hex_side_from_area_m2(area_m2)
            print(f"Hexagon side length: {a:.2f} meters")
            tiles_iter = generate_hex_grid(libya_laea, a)

        # Filter/shape according to mode
        print(f"Filtering tiles using mode: {args.mode}")
        shaped_tiles = filter_and_shape_tiles(tiles_iter, libya_laea, args.mode)

        if not shaped_tiles:
            raise SystemExit("No tiles produced. Try increasing tile size, switching mode, or verifying boundary data.")

        print(f"Generated {len(shaped_tiles)} tiles")

        print(shaped_tiles)

        # Export each tile to its own KML/KMZ
        total = 0
        for geom_laea, (ix, iy) in shaped_tiles:
            if args.limit is not None and total >= args.limit:
                break
            geom_wgs84 = shp_transform(to_wgs84, geom_laea)

            actual_area_km2 = geom_laea.area / 1_000_000.0
            name = f"{args.grid} tile ({args.mode}) idx=({ix},{iy})"
            desc = f"Tile area (nominal): {args.tile_area_km2:.6f} km²; actual: {actual_area_km2:.6f} km²"

            fname = f"{args.basename}_{args.grid}_{args.mode}_{ix}_{iy}"
            if args.save_kmz:
                filepath = os.path.join(args.outdir, f"{fname}.kmz")
            else:
                filepath = os.path.join(args.outdir, f"{fname}.kml")

            if args.save_kmz:
                kml = simplekml.Kml()
                def add_polygon(poly: Polygon):
                    # Get exterior coordinates
                    exterior = [(float(x), float(y)) for x, y in poly.exterior.coords]
                    
                    # Get interior coordinates
                    inner_boundaries = []
                    for interior_ring in poly.interiors:
                        interior_coords = [(float(x), float(y)) for x, y in interior_ring.coords]
                        inner_boundaries.append(interior_coords)
                    
                    # Build kwargs - only include innerboundaryis if we have interior boundaries
                    polygon_kwargs = {
                        'name': name,
                        'description': desc,
                        'outerboundaryis': exterior
                    }
                    
                    # Only add innerboundaryis if we actually have inner boundaries
                    if inner_boundaries:
                        polygon_kwargs['innerboundaryis'] = inner_boundaries
                    
                    # Create polygon with the appropriate parameters
                    pol = kml.newpolygon(**polygon_kwargs)
                    pol.style.linestyle.width = 2
                    pol.style.linestyle.color = simplekml.Color.red
                    pol.style.polystyle.color = simplekml.Color.changealphaint(80, simplekml.Color.yellow)
                    pol.extrude = 0
                    pol.tessellate = 1
                    pol.altitudemode = simplekml.AltitudeMode.clamptoground
                if isinstance(geom_wgs84, Polygon):
                    add_polygon(geom_wgs84)
                elif isinstance(geom_wgs84, MultiPolygon):
                    for poly in geom_wgs84.geoms:
                        add_polygon(poly)
                else:
                    try:
                        for poly in geom_wgs84.geoms:  # type: ignore[attr-defined]
                            if isinstance(poly, Polygon):
                                add_polygon(poly)
                    except Exception:
                        pass
                kml.savekmz(filepath)
            else:
                save_geom_kml(geom_wgs84, filepath, name, desc)

            total += 1

        print(f"Exported {total} tiles to: {os.path.abspath(args.outdir)}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()