"""
prepare_data.py - Extract irradiance_train.npy and coords.npy from PVGIS data.

Usage:
    python prepare_data.py --nc path/to/pvgis_data.nc --coords path/to/panels.csv

Expects a CSV with header:   ID, lat, long
The NetCDF location dimension must match the CSV row count (same order).

Writes:
    datasets/irradiance_train.npy   shape (T, N)  float32
    datasets/coords.npy             shape (N, 2)  float32  -- columns: [lat, lon]
    datasets/panel_ids.npy          shape (N,)    string
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_coords_csv(path):
    """Read CSV with header ID, lat, long. Returns (ids, coords (N,2))."""
    ids, lats, lons = [], [], []
    with open(path, newline="") as f:
        header = f.readline().strip()
        cols = [c.strip().lower() for c in header.split(",")]

        try:
            id_idx = cols.index("id")
        except ValueError:
            id_idx = 0

        try:
            lat_idx = cols.index("lat")
        except ValueError:
            raise ValueError(f"No 'lat' column in CSV header: {header}")

        lon_name = "long" if "long" in cols else "lon"
        try:
            lon_idx = cols.index(lon_name)
        except ValueError:
            raise ValueError(f"No 'lon'/'long' column in CSV header: {header}")

        print(f"  CSV columns: id={cols[id_idx]!r}  "
              f"lat={cols[lat_idx]!r}  lon={cols[lon_idx]!r}")

        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                ids.append(parts[id_idx])
                lats.append(float(parts[lat_idx]))
                lons.append(float(parts[lon_idx]))
            except (IndexError, ValueError) as e:
                print(f"  WARNING: skipping malformed line {line_no}: {line!r} ({e})")

    coords = np.column_stack([lats, lons]).astype(np.float32)
    return np.array(ids), coords


# ── NetCDF loader with encoding-aware extraction ──────────────────────────────

def extract_variable(ds, var_name):
    """
    Extract a 2-D variable from an open xarray dataset as a float32 numpy array
    with shape (T, N).
    """
    if var_name not in ds:
        print(f"  ERROR: '{var_name}' not in NetCDF.")
        print(f"  Available: {list(ds.data_vars)}")
        sys.exit(1)

    da = ds[var_name]
    
    # Identify time vs location axes
    TIME_NAMES     = {"time", "t", "datetime", "date", "step"}
    LOCATION_NAMES = {"location", "loc", "station", "site", "panel", "point", "index", "id"}

    dims_lower = [d.lower() for d in da.dims]
    time_axis = next((i for i, d in enumerate(dims_lower) if d in TIME_NAMES), None)
    loc_axis  = next((i for i, d in enumerate(dims_lower) if d in LOCATION_NAMES), None)

    if time_axis is None or loc_axis is None:
        if da.shape[0] >= da.shape[1]:
            time_axis, loc_axis = 0, 1
        else:
            time_axis, loc_axis = 1, 0

    raw = da.values

    # Manual scale/offset if needed
    if np.issubdtype(raw.dtype, np.integer):
        scale  = da.encoding.get("scale_factor",  da.attrs.get("scale_factor",  1.0))
        offset = da.encoding.get("add_offset",    da.attrs.get("add_offset",    0.0))
        fill   = da.encoding.get("_FillValue",    da.attrs.get("_FillValue",    None))
        raw = raw.astype(np.float64)
        if fill is not None:
            raw[raw == fill] = np.nan
        raw = raw * scale + offset

    # Transpose to (T, N)
    if time_axis == 0 and loc_axis == 1:
        arr = raw.astype(np.float32)
    elif time_axis == 1 and loc_axis == 0:
        arr = raw.T.astype(np.float32)
    else:
        print(f"  ERROR: unexpected dims {da.dims} for {var_name}.")
        sys.exit(1)

    return arr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare irradiance_train.npy, sun_height.npy, and coords.npy from PVGIS NetCDF + CSV"
    )
    parser.add_argument("--nc",      required=True, nargs="+",
                        help="Path(s) to PVGIS NetCDF file(s)")
    parser.add_argument("--coords", required=True,
                        help="Path to CSV file with columns: ID, lat, long")
    parser.add_argument("--irrad",   default="solar_irradiance_poa",
                        help="Irradiance variable name (default: solar_irradiance_poa)")
    parser.add_argument("--sun",     default="sun_height",
                        help="Sun height variable name (default: sun_height)")
    parser.add_argument("--out",     default="datasets",
                        help="Output directory (default: datasets/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Coordinates
    print(f"\n[1/3] Reading coordinates: {args.coords}")
    panel_ids, coords = load_coords_csv(args.coords)
    N_csv = len(coords)
    print(f"      Panels: {N_csv}")

    # 2. Extract Data
    nc_files = args.nc
    print(f"\n[2/3] Reading {len(nc_files)} NetCDF file(s)")

    irrad_slices = []
    sun_slices = []

    for file_idx, nc_path in enumerate(nc_files):
        print(f"\n  [{file_idx + 1}/{len(nc_files)}] {nc_path}")
        ds = xr.open_dataset(nc_path)
        
        # Extract both variables
        irr_arr = extract_variable(ds, args.irrad)
        sun_arr = extract_variable(ds, args.sun)
        
        T_i, N_nc = irr_arr.shape
        
        if N_nc != N_csv:
            print(f"\n  ERROR: CSV has {N_csv} panels but '{nc_path}' has {N_nc} locations.")
            sys.exit(1)

        # Optional Coord Check (First file only usually sufficient, but we do it here)
        if "lat" in ds and "lon" in ds:
            nc_lat = ds["lat"].values.astype(np.float32)
            nc_lon = ds["lon"].values.astype(np.float32)
            if float(np.abs(nc_lat - coords[:, 0]).max()) > 0.01:
                print("  WARNING: Coordinate mismatch detected.")

        irrad_slices.append(irr_arr)
        sun_slices.append(sun_arr)
        ds.close()

    # Concatenate
    print(f"\n  Concatenating arrays along time axis...")
    final_irrad = np.concatenate(irrad_slices, axis=0)
    final_sun = np.concatenate(sun_slices, axis=0)

    # Data Quality Cleanup
    def clean_data(arr, name):
        n_nan = int(np.isnan(arr).sum())
        if n_nan > 0:
            print(f"  WARNING: {n_nan} NaNs in {name} - replacing with 0.0")
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    final_irrad = clean_data(final_irrad, "irradiance")
    final_sun = clean_data(final_sun, "sun_height")

    # 3. Save
    print(f"\n[3/3] Saving to {args.out}/")

    np.save(os.path.join(args.out, "irradiance_train.npy"), final_irrad)
    np.save(os.path.join(args.out, "sun_height.npy"),      final_sun)
    np.save(os.path.join(args.out, "coords.npy"),          coords)
    np.save(os.path.join(args.out, "panel_ids.npy"),       panel_ids)

    print(f"      irradiance_train.npy  shape={final_irrad.shape}")
    print(f"      sun_height.npy        shape={final_sun.shape}")
    print(f"      coords.npy            shape={coords.shape}")
    print("\nDone.")

if __name__ == "__main__":
    main()