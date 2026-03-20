#!/usr/bin/env python3
"""
Fill sparse GT_T1 and GT_T2 to produce full semantic maps.

The JL1_second labels only annotate change regions; unchanged pixels are 0.
This script fills zeros using nearest-neighbor propagation so every pixel
has a semantic class (1-6) or 0 (unchanged).
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy import ndimage


def fill_labels_nn(label: np.ndarray, valid_classes: tuple = (1, 2, 3, 4, 5, 6)) -> np.ndarray:
    """
    Fill zero pixels with the value of the nearest non-zero pixel (nearest-neighbor).
    """
    label = np.asarray(label, dtype=np.int32)
    mask = np.isin(label, valid_classes)
    if mask.sum() == 0:
        return label
    if mask.all():
        return label

    # Distance from each unlabeled pixel to nearest labeled pixel
    dist, indices = ndimage.distance_transform_edt(
        ~mask.astype(bool), return_distances=True, return_indices=True
    )
    fill_mask = label == 0
    if fill_mask.sum() == 0:
        return label
    ri, ci = indices[0], indices[1]
    filled = label.copy()
    filled[fill_mask] = label[ri[fill_mask], ci[fill_mask]]
    return filled.astype(np.uint8)


def fix_sample(gt_t1: np.ndarray, gt_t2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill GT_T1 and GT_T2. At unchanged (0,0) pixels, use t1 filled for both."""
    t1 = np.asarray(gt_t1, dtype=np.int32)
    t2 = np.asarray(gt_t2, dtype=np.int32)
    valid = (1, 2, 3, 4, 5, 6)

    t1_filled = fill_labels_nn(t1, valid)
    t2_filled = fill_labels_nn(t2, valid)

    # At originally (0,0) pixels, enforce same class for both (unchanged = same land-cover)
    unchanged_orig = (t1 == 0) & (t2 == 0)
    if unchanged_orig.any():
        t2_filled[unchanged_orig] = t1_filled[unchanged_orig]

    return t1_filled.astype(np.uint8), t2_filled.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Fill sparse GT_T1/GT_T2 to full semantic maps")
    parser.add_argument(
        "--root",
        type=str,
        default="/root/workspace/sakura/ML4RS-Cropland-Recognition/datasets/JL1_second",
        help="JL1_second dataset root",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report, do not overwrite")
    args = parser.parse_args()
    root = Path(args.root)

    for mode in ["train", "val", "test"]:
        gt_t1_dir = root / mode / "GT_T1"
        gt_t2_dir = root / mode / "GT_T2"
        if not gt_t1_dir.exists() or not gt_t2_dir.exists():
            continue

        files = sorted(f.stem for f in gt_t1_dir.glob("*.png") if (gt_t2_dir / f"{f.stem}.png").exists())
        try:
            from tqdm import tqdm
            files = tqdm(files, desc=f"fix {mode}")
        except ImportError:
            pass

        for stem in files:
            t1_path = gt_t1_dir / f"{stem}.png"
            t2_path = gt_t2_dir / f"{stem}.png"
            t1 = np.asarray(imageio.imread(str(t1_path)))
            t2 = np.asarray(imageio.imread(str(t2_path)))
            if t1.ndim > 2:
                t1 = t1[:, :, 0]
            if t2.ndim > 2:
                t2 = t2[:, :, 0]

            nz1, nz2 = (t1 > 0).sum(), (t2 > 0).sum()
            total = t1.size
            if nz1 == total and nz2 == total:
                continue

            t1_new, t2_new = fix_sample(t1, t2)
            if not args.dry_run:
                imageio.imwrite(str(t1_path), t1_new)
                imageio.imwrite(str(t2_path), t2_new)

    print("Done.")


if __name__ == "__main__":
    main()
