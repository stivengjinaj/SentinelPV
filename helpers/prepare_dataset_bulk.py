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

def extract_variable(nc_path, var_name):
    """
    Extract a 2-D variable from a NetCDF file as a float32 numpy array
    with shape (T, N), handling all common PVGIS encoding issues:

      1. Packed integers  (scale_factor / add_offset attributes)
      2. Masked fill values converting valid data to NaN/0
      3. decode_times failures (non-CF time axes)

    Returns: (data (T, N), time_dim_name, loc_dim_name)
    """

    # Open exactly the same way a Jupyter notebook does -- no extra kwargs.
    # Using mask_and_scale + decode_times together zeros out values on
    # certain xarray/netCDF4 version combinations, so we keep it plain.
    ds = xr.open_dataset(nc_path)

    # ── Print full structure ───────────────────────────────────────────────────
    print("\n      --- NetCDF structure ---")
    print("      Dimensions:")
    for dim, size in ds.dims.items():
        print(f"        {dim}: {size}")
    print("      Variables:")
    for vname, var in ds.data_vars.items():
        print(f"        {vname}{list(var.dims)}  shape={var.shape}")
    print("      --- end structure ---\n")

    if var_name not in ds:
        print(f"  ERROR: '{var_name}' not in NetCDF.")
        print(f"  Available: {list(ds.data_vars)}")
        print(f"  Re-run with --var <name>")
        sys.exit(1)

    da = ds[var_name]
    print(f"      Extracting '{var_name}'{list(da.dims)}  shape={da.shape}")
    print(f"      Encoding   : {da.encoding}")
    print(f"      Attributes : {da.attrs}")

    # ── Identify time vs location axes ────────────────────────────────────────
    TIME_NAMES     = {"time", "t", "datetime", "date", "step"}
    LOCATION_NAMES = {"location", "loc", "station", "site",
                      "panel", "point", "index", "id"}

    dims_lower = [d.lower() for d in da.dims]

    time_axis = next(
        (i for i, d in enumerate(dims_lower) if d in TIME_NAMES), None)
    loc_axis  = next(
        (i for i, d in enumerate(dims_lower) if d in LOCATION_NAMES), None)

    if time_axis is None or loc_axis is None:
        print(f"  WARNING: cannot identify axes by name (dims={list(da.dims)}).")
        print(f"  Using heuristic: larger axis = time.")
        if da.shape[0] >= da.shape[1]:
            time_axis, loc_axis = 0, 1
        else:
            time_axis, loc_axis = 1, 0

    print(f"      time axis     : '{da.dims[time_axis]}' "
          f"(axis {time_axis}, size {da.shape[time_axis]})")
    print(f"      location axis : '{da.dims[loc_axis]}'  "
          f"(axis {loc_axis}, size {da.shape[loc_axis]})")

    # ── Extract values with masking awareness ─────────────────────────────────
    # .values on a masked DataArray returns a numpy array where masked cells
    # become np.nan. We convert nan -> 0 later in the pipeline.
    raw = da.values

    # If the array came back as integer dtype, scale/offset was NOT applied
    # by xarray (can happen when mask_and_scale is ignored for some backends).
    # Apply manually in that case.
    if np.issubdtype(raw.dtype, np.integer):
        scale  = da.encoding.get("scale_factor",  da.attrs.get("scale_factor",  1.0))
        offset = da.encoding.get("add_offset",    da.attrs.get("add_offset",    0.0))
        fill   = da.encoding.get("_FillValue",    da.attrs.get("_FillValue",    None))
        print(f"      Packed integer detected -- applying manually: "
              f"scale={scale}, offset={offset}, fill={fill}")
        raw = raw.astype(np.float64)
        if fill is not None:
            raw[raw == fill] = np.nan
        raw = raw * scale + offset

    # Transpose to (T, N) if stored as (N, T)
    if time_axis == 0 and loc_axis == 1:
        arr = raw.astype(np.float32)
    elif time_axis == 1 and loc_axis == 0:
        arr = raw.T.astype(np.float32)
    else:
        print(f"  ERROR: unexpected dims {da.dims} -- only 2-D variables supported.")
        sys.exit(1)

    ds.close()
    return arr, da.dims[time_axis], da.dims[loc_axis]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare irradiance_train.npy and coords.npy from PVGIS NetCDF + CSV"
    )
    parser.add_argument("--nc",     required=True, nargs="+",
                        help="Path(s) to PVGIS NetCDF file(s); multiple files are concatenated along the time axis")
    parser.add_argument("--coords", required=True,
                        help="Path to CSV file with columns: ID, lat, long")
    parser.add_argument("--var",    default="solar_irradiance_poa",
                        help="NetCDF variable name  (default: solar_irradiance_poa)")
    parser.add_argument("--out",    default="datasets",
                        help="Output directory  (default: datasets/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── 1. Coordinates ────────────────────────────────────────────────────────
    print(f"\n[1/3] Reading coordinates: {args.coords}")
    panel_ids, coords = load_coords_csv(args.coords)
    N_csv = len(coords)
    print(f"      Panels  : {N_csv}")
    print(f"      Lat     : {coords[:,0].min():.5f} -> {coords[:,0].max():.5f}")
    print(f"      Lon     : {coords[:,1].min():.5f} -> {coords[:,1].max():.5f}")

    # ── 2. Irradiance ─────────────────────────────────────────────────────────
    nc_files = args.nc
    print(f"\n[2/3] Reading {len(nc_files)} NetCDF file(s)")

    slices = []
    for file_idx, nc_path in enumerate(nc_files):
        print(f"\n  [{file_idx + 1}/{len(nc_files)}] {nc_path}")
        arr, time_dim, loc_dim = extract_variable(nc_path, args.var)
        T_i, N_nc = arr.shape
        print(f"      Output shape : (T={T_i}, N={N_nc})")

        if N_nc != N_csv:
            print(
                f"\n  ERROR: CSV has {N_csv} panels but '{nc_path}' has {N_nc} locations."
                f"\n  Both files must list panels in the same order."
            )
            sys.exit(1)

        ds_check = xr.open_dataset(nc_path)
        if "lat" in ds_check and "lon" in ds_check:
            nc_lat = ds_check["lat"].values.astype(np.float32)
            nc_lon = ds_check["lon"].values.astype(np.float32)
            max_lat_err = float(np.abs(nc_lat - coords[:, 0]).max())
            max_lon_err = float(np.abs(nc_lon - coords[:, 1]).max())
            if max_lat_err > 0.01 or max_lon_err > 0.01:
                print(f"\n  WARNING: CSV vs NetCDF coords differ by up to "
                      f"{max_lat_err:.5f} lat / {max_lon_err:.5f} lon. "
                      f"Using CSV as source of truth.")
            else:
                print(f"      Coord cross-check OK  "
                      f"(max diff: {max_lat_err:.6f} lat, {max_lon_err:.6f} lon)")
        ds_check.close()

        slices.append(arr)

    if len(slices) == 1:
        irr = slices[0]
    else:
        print(f"\n  Concatenating {len(slices)} arrays along time axis ...")
        irr = np.concatenate(slices, axis=0)

    T, N_nc = irr.shape
    print(f"\n      Combined shape : (T={T}, N={N_nc})")

    # ── 3. Data quality ───────────────────────────────────────────────────────
    n_nan  = int(np.isnan(irr).sum())
    n_inf  = int(np.isinf(irr).sum())
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: {n_nan} NaN and {n_inf} Inf values -- replacing with 0.0.")
        irr = np.nan_to_num(irr, nan=0.0, posinf=0.0, neginf=0.0)

    n_zero   = int((irr == 0.0).sum())
    total    = irr.size
    zero_pct = 100.0 * n_zero / total
    nonzero  = irr[irr > 0]

    print(f"\n      Irradiance statistics:")
    print(f"        min          : {irr.min():.4f} W/m2")
    print(f"        max          : {irr.max():.4f} W/m2")
    print(f"        mean (all)   : {irr.mean():.4f} W/m2")
    print(f"        zeros        : {n_zero:,} / {total:,}  ({zero_pct:.1f}%)")
    if len(nonzero) > 0:
        print(f"        mean (daytime): {nonzero.mean():.4f} W/m2")

    if irr.max() == 0.0:
        print(
            "\n  ERROR: all values are zero after decoding."
            "\n"
            "\n  Possible causes and fixes:"
            "\n    1. Wrong variable name"
            "\n       -> Re-run with a different --var (see structure above)"
            "\n"
            "\n    2. scale_factor / add_offset not applied by xarray"
            "\n       -> Run the diagnostic script below and share the output:"
            "\n          python inspect_nc.py your_file.nc"
            "\n"
            "\n    3. File uses a non-standard fill value that masked all data"
            "\n       -> Check '_FillValue' in the attributes printed above"
            "\n"
            "\n    4. File is corrupt or empty"
            "\n       -> Try: ncdump -h your_file.nc"
        )
        sys.exit(1)

    if zero_pct > 70.0:
        print(
            f"\n  NOTE: {zero_pct:.1f}% zeros is normal for hourly PVGIS data"
            f"\n  (nighttime + overcast hours are legitimately zero)."
        )

    # ── 4. Save ───────────────────────────────────────────────────────────────
    print(f"\n[3/3] Saving to {args.out}/")

    irr_path    = os.path.join(args.out, "irradiance_train.npy")
    coords_path = os.path.join(args.out, "coords.npy")
    ids_path    = os.path.join(args.out, "panel_ids.npy")

    np.save(irr_path,    irr)
    np.save(coords_path, coords)
    np.save(ids_path,    panel_ids)

    print(f"      irradiance_train.npy  shape={irr.shape}   dtype=float32")
    print(f"      coords.npy            shape={coords.shape}  dtype=float32")
    print(f"      panel_ids.npy         shape={panel_ids.shape}")

    print("\nDone. Next step: python train_stage1.py")


if __name__ == "__main__":
    main()