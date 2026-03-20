#!/usr/bin/env python3
"""
Reformat JL1_second dataset to ChangeMamba SCD format.

JL1_second structure:
  train|val|test/
    im1/*.tif, im2/*.tif
    label1_gray/*.png, label2_gray/*.png

ChangeMamba SCD structure:
  train|val|test/
    T1/*.png, T2/*.png
    GT_T1/*.png, GT_T2/*.png, GT_CD/*.png
  train.txt, val.txt, test.txt (run scripts/build_jl1_second_holdout_test.py afterward for competition hold-out test)
"""

import argparse
import os
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def ensure_3ch(img: np.ndarray) -> np.ndarray:
    """Ensure image is HxWx3 for T1/T2."""
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[-1] > 3:
        return img[..., :3].copy()
    return img


def ensure_2d(label: np.ndarray) -> np.ndarray:
    """Ensure label is HxW single-channel."""
    if label.ndim == 3:
        if label.shape[-1] == 1:
            return label[:, :, 0]
        # RGB label - take first channel or convert; assume grayscale stored in R
        return label[:, :, 0]
    return label


def reformat_split(root: Path, mode: str) -> list[str]:
    """Reformat one split (train/val/test). Returns list of new base names for the list file."""
    im1_dir = root / mode / "im1"
    im2_dir = root / mode / "im2"
    label1_dir = root / mode / "label1_gray"
    label2_dir = root / mode / "label2_gray"

    # Create new dirs
    t1_dir = root / mode / "T1"
    t2_dir = root / mode / "T2"
    gt_t1_dir = root / mode / "GT_T1"
    gt_t2_dir = root / mode / "GT_T2"
    gt_cd_dir = root / mode / "GT_CD"
    for d in [t1_dir, t2_dir, gt_t1_dir, gt_t2_dir, gt_cd_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all image names from im1 (match .tif)
    im1_files = sorted(f for f in im1_dir.iterdir() if f.suffix.lower() in (".tif", ".tiff"))
    names = []
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(im1_files), total=len(im1_files), desc=mode)
    except ImportError:
        iterator = enumerate(im1_files)
    for idx, im1_path in iterator:
        base = im1_path.stem  # e.g. train_image_0
        im2_path = im2_dir / (base + im1_path.suffix)
        if not im2_path.exists():
            im2_path = im2_dir / (base + ".tif")
        label1_path = label1_dir / (base + ".png")
        label2_path = label2_dir / (base + ".png")

        if not im2_path.exists() or not label1_path.exists() or not label2_path.exists():
            print(f"  Skip {base}: missing pair")
            continue

        new_name = f"{idx + 1:05d}"
        names.append(new_name)

        # T1, T2: convert tif -> png, ensure 3ch
        for src, dst_dir in [(im1_path, t1_dir), (im2_path, t2_dir)]:
            img = imageio.imread(str(src))
            img = ensure_3ch(np.asarray(img, dtype=np.uint8))
            imageio.imwrite(str(dst_dir / f"{new_name}.png"), img)

        # GT_T1, GT_T2: copy labels, ensure single-channel; GT_CD from same data
        t1_lbl = ensure_2d(np.asarray(imageio.imread(str(label1_path)), dtype=np.uint8))
        t2_lbl = ensure_2d(np.asarray(imageio.imread(str(label2_path)), dtype=np.uint8))
        imageio.imwrite(str(gt_t1_dir / f"{new_name}.png"), t1_lbl)
        imageio.imwrite(str(gt_t2_dir / f"{new_name}.png"), t2_lbl)
        cd = ((t1_lbl != t2_lbl).astype(np.uint8)) * 255
        imageio.imwrite(str(gt_cd_dir / f"{new_name}.png"), cd)

    # Remove old dirs
    for old in [im1_dir, im2_dir, label1_dir, label2_dir]:
        if old.exists():
            shutil.rmtree(old)

    return names


def main():
    parser = argparse.ArgumentParser(description="Reformat JL1_second to ChangeMamba SCD format")
    parser.add_argument(
        "--root",
        type=str,
        default="/root/workspace/sakura/ML4RS-Cropland-Recognition/datasets/JL1_second",
        help="JL1_second dataset root",
    )
    args = parser.parse_args()
    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for mode in ["train", "val", "test"]:
        if (root / mode).exists():
            print(f"Reformatting {mode}...")
            names = reformat_split(root, mode)
            list_path = root / f"{mode}.txt"
            with open(list_path, "w") as f:
                for n in names:
                    # train: no extension (loader adds .png)
                    # val/test: with .png (used as validation/test, loader uses as-is)
                    entry = f"{n}.png" if mode in ("val", "test") else n
                    f.write(f"{entry}\n")
            print(f"  {len(names)} samples -> {list_path}")
        else:
            print(f"  Skip {mode}: no such dir")

    print("Done.")


if __name__ == "__main__":
    main()
