import numpy as np


def snap_to_regular_grid(coords, grid_h, grid_w):
    """
    Assign each panel to a cell in a grid_h x grid_w regular grid.
    coords : (N, 2) already normalised to [0, 1]
    Returns:
        grid_indices : (N, 2) int    (row, col) cell for each panel
        grid_coords  : (N, 2) float  centre of assigned cell in [0, 1]
    """
    row_step = 1.0 / grid_h
    col_step = 1.0 / grid_w

    rows = np.clip((coords[:, 0] / row_step).astype(int), 0, grid_h - 1)
    cols = np.clip((coords[:, 1] / col_step).astype(int), 0, grid_w - 1)

    grid_coords = np.stack([
        (rows + 0.5) * row_step,
        (cols + 0.5) * col_step,
    ], axis=-1).astype(np.float32)

    return np.stack([rows, cols], axis=-1), grid_coords


# ── Load raw coords and normalise FIRST ───────────────────────────────────────
raw_coords = np.load("datasets/coords.npy")          # (N, 2)  raw lat/lon degrees

coords_min  = raw_coords.min(axis=0)
coords_max  = raw_coords.max(axis=0)
coords_norm = (raw_coords - coords_min) / (coords_max - coords_min + 1e-8)

print(f"Raw  lat range : {raw_coords[:, 0].min():.5f} -> {raw_coords[:, 0].max():.5f}")
print(f"Raw  lon range : {raw_coords[:, 1].min():.5f} -> {raw_coords[:, 1].max():.5f}")
print(f"Norm lat range : {coords_norm[:, 0].min():.5f} -> {coords_norm[:, 0].max():.5f}")
print(f"Norm lon range : {coords_norm[:, 1].min():.5f} -> {coords_norm[:, 1].max():.5f}")

# ── Snap to grid ──────────────────────────────────────────────────────────────
GRID_H, GRID_W = 34, 34

grid_indices, grid_coords = snap_to_regular_grid(coords_norm, GRID_H, GRID_W)

# ── Sanity check before saving ────────────────────────────────────────────────
flat        = grid_indices[:, 0] * GRID_W + grid_indices[:, 1]
unique, counts = np.unique(flat, return_counts=True)
total_cells    = GRID_H * GRID_W
filled_cells   = len(unique)
empty_cells    = total_cells - filled_cells
collision_cells = (counts > 1).sum()

print(f"\nGrid {GRID_H}x{GRID_W} = {total_cells} cells  |  {len(raw_coords)} panels")
print(f"Filled cells    : {filled_cells}  ({100*filled_cells/total_cells:.1f}%)")
print(f"Empty cells     : {empty_cells}   ({100*empty_cells/total_cells:.1f}%)")
print(f"Collision cells : {collision_cells}  (cells with 2+ panels)")
print(f"Max panels/cell : {counts.max()}")
print(f"grid_coords sample (first 5):\n{grid_coords[:5]}")

# ── Save ──────────────────────────────────────────────────────────────────────
np.save("datasets/grid_indices.npy", grid_indices)
np.save("datasets/grid_coords.npy",  grid_coords)
np.save("datasets/coords_norm.npy",  coords_norm)   # save for reuse in dataset/infer

print("\nSaved grid_indices.npy, grid_coords.npy, coords_norm.npy")