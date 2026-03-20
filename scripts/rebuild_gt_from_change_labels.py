#!/usr/bin/env python3
"""
Rebuild GT_T1, GT_T2, GT_CD in JL1_second from jl1_cropland_competition_2023 change labels.

Matches JL1_second T1 images to competition pre images by pixel-hash,
then converts the 0-8 change labels to per-pixel semantic maps.

Classes:  0=background, 1=cropland, 2=road, 3=forest-grass, 4=building, 5=other
"""

import argparse
import hashlib
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.colormap import change_label_to_semantic


def pixel_hash(arr: np.ndarray) -> str:
    """MD5 hash of raw pixel bytes."""
    a = np.ascontiguousarray(arr)
    return hashlib.md5(a.tobytes()).hexdigest()


def ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[-1] > 3:
        return img[..., :3].copy()
    return img


def build_competition_hash_table(comp_root: Path) -> dict[str, str]:
    """Build hash → competition base-name lookup from train/pre images."""
    table = {}
    pre_dir = comp_root / "train" / "pre"
    files = sorted(pre_dir.glob("*.tif"))
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="hashing competition pre")
    except ImportError:
        pass
    for f in files:
        img = np.asarray(imageio.imread(str(f)), dtype=np.uint8)
        img = ensure_3ch(img)
        h = pixel_hash(img)
        table[h] = f.stem  # e.g. "image_0"
    return table


def rebuild_split(jl1_root: Path, split: str, hash_table: dict, comp_label_dir: Path):
    """Rebuild GT_T1/GT_T2/GT_CD for one split."""
    t1_dir = jl1_root / split / "T1"
    gt_t1_dir = jl1_root / split / "GT_T1"
    gt_t2_dir = jl1_root / split / "GT_T2"
    gt_cd_dir = jl1_root / split / "GT_CD"

    if not t1_dir.exists():
        print(f"  {split}: T1 dir not found, skip")
        return

    for d in [gt_t1_dir, gt_t2_dir, gt_cd_dir]:
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(t1_dir.glob("*.png"))
    matched, unmatched = 0, 0
    try:
        from tqdm import tqdm
        files = tqdm(files, desc=f"rebuild {split}")
    except ImportError:
        pass

    for f in files:
        img = np.asarray(imageio.imread(str(f)), dtype=np.uint8)
        img = ensure_3ch(img)
        h = pixel_hash(img)
        stem = f.stem  # e.g. "00001"

        comp_name = hash_table.get(h)
        if comp_name is None:
            unmatched += 1
            continue

        label_path = comp_label_dir / f"{comp_name}.png"
        if not label_path.exists():
            unmatched += 1
            continue

        change_label = np.asarray(imageio.imread(str(label_path)), dtype=np.uint8)
        if change_label.ndim == 3:
            change_label = change_label[:, :, 0]

        gt_t1, gt_t2, gt_cd = change_label_to_semantic(change_label)
        imageio.imwrite(str(gt_t1_dir / f"{stem}.png"), gt_t1)
        imageio.imwrite(str(gt_t2_dir / f"{stem}.png"), gt_t2)
        imageio.imwrite(str(gt_cd_dir / f"{stem}.png"), gt_cd)
        matched += 1

    print(f"  {split}: {matched} matched, {unmatched} unmatched")


def main():
    parser = argparse.ArgumentParser(description="Rebuild GT from competition change labels")
    parser.add_argument("--jl1-root", type=str,
                        default="/root/workspace/sakura/ML4RS-Cropland-Recognition/datasets/JL1_second")
    parser.add_argument("--comp-root", type=str,
                        default="/root/workspace/sakura/ML4RS-Cropland-Recognition/datasets/jl1_cropland_competition_2023")
    args = parser.parse_args()

    jl1_root = Path(args.jl1_root)
    comp_root = Path(args.comp_root)
    comp_label_dir = comp_root / "train" / "label"

    print("Building hash table from competition pre images...")
    hash_table = build_competition_hash_table(comp_root)
    print(f"  {len(hash_table)} competition images hashed")

    # test/ is competition hold-out without public labels (see build_jl1_second_holdout_test.py)
    for split in ["train", "val"]:
        rebuild_split(jl1_root, split, hash_table, comp_label_dir)

    print("Done.")


if __name__ == "__main__":
    main()
