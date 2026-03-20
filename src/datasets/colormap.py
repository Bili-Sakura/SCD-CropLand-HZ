"""
Dataset colormap utilities for JL1 cropland change detection.
"""

import numpy as np

# ── JL1 land-cover classes ──────────────────────────────────────────
# 0: background (unannotated / no cropland-change region)
# 1: cropland   (耕地)
# 2: road       (道路)
# 3: forest-grass (林草)
# 4: building   (建筑)
# 5: other      (其他 — catch-all land-cover that is none of the above)
NUM_CLASSES = 6  # 0-5 inclusive

JL1_COLORMAP = [
    [0, 0, 0],        # 0: background (black)
    [255, 255, 0],    # 1: cropland   (yellow)
    [128, 128, 128],  # 2: road       (gray)
    [0, 180, 0],      # 3: forest-grass (green)
    [255, 0, 0],      # 4: building   (red)
    [0, 0, 255],      # 5: other      (blue)
]
JL1_CLASSES = [
    "background",
    "cropland",
    "road",
    "forest-grass",
    "building",
    "other",
]

# ── Change-label → (GT_T1, GT_T2) mapping ──────────────────────────
# From jl1_cropland_competition_2023/readme.txt:
#   1: cropland→road, 2: cropland→forest-grass, 3: cropland→building,
#   4: cropland→other, 5: road→cropland, 6: forest-grass→cropland,
#   7: building→cropland, 8: other→cropland, 0: no cropland change
CHANGE_TO_T1T2 = {
    0: (0, 0),  # background
    1: (1, 2),  # cropland → road
    2: (1, 3),  # cropland → forest-grass
    3: (1, 4),  # cropland → building
    4: (1, 5),  # cropland → other
    5: (2, 1),  # road → cropland
    6: (3, 1),  # forest-grass → cropland
    7: (4, 1),  # building → cropland
    8: (5, 1),  # other → cropland
}


def change_label_to_semantic(change_label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a change-label map (0-8) to GT_T1, GT_T2, GT_CD.

    Args:
        change_label: HxW uint8 array with values 0-8

    Returns:
        gt_t1: HxW uint8 semantic map for T1
        gt_t2: HxW uint8 semantic map for T2
        gt_cd: HxW uint8 binary change map (0/255)
    """
    change_label = np.asarray(change_label, dtype=np.uint8)
    gt_t1 = np.zeros_like(change_label)
    gt_t2 = np.zeros_like(change_label)
    for cv, (c1, c2) in CHANGE_TO_T1T2.items():
        mask = change_label == cv
        gt_t1[mask] = c1
        gt_t2[mask] = c2
    gt_cd = ((change_label > 0).astype(np.uint8)) * 255
    return gt_t1, gt_t2, gt_cd


# ── Color utilities ─────────────────────────────────────────────────
# Build RGB → class index lookup table
_colormap2label = np.zeros(256**3, dtype=np.int64)
for _i, _cm in enumerate(JL1_COLORMAP):
    _colormap2label[(_cm[0] * 256 + _cm[1]) * 256 + _cm[2]] = _i


def color2index(color_label: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert RGB label image to class index map."""
    data = color_label.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    index_map = _colormap2label[idx].copy()
    index_map = index_map * (index_map < num_classes)
    return index_map


def index2color(pred: np.ndarray) -> np.ndarray:
    """Convert class index map to RGB color image for visualization."""
    colormap = np.asarray(JL1_COLORMAP, dtype=np.uint8)
    x = np.asarray(pred, dtype=np.int32)
    x = np.clip(x, 0, len(JL1_COLORMAP) - 1)
    return colormap[x, :]


def color2index_batch(color_labels: list) -> list:
    """Convert a list of RGB label images to index maps."""
    return [color2index(cl) for cl in color_labels]


# Backward-compatible aliases
Color2Index = color2index
Index2Color = index2color
Colorls2Index = color2index_batch

# Legacy names kept for imports that reference the old names
ST_COLORMAP = JL1_COLORMAP
ST_CLASSES = JL1_CLASSES
