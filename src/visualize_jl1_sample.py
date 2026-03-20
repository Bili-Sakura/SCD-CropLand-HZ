#!/usr/bin/env python3
"""
Visualize a random sample from JL1_second dataset with color-coded semantic labels.
"""

import argparse
import random
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.colormap import index2color


def collect_samples(root: Path, splits: list[str] | None = None) -> list[tuple[str, str]]:
    """Collect (mode, stem) pairs for samples with T1 (and GT for visualization).
    Skips splits without GT_T1 (e.g. hold-out test has imagery only).
    """
    if splits is None:
        splits = ["train", "val", "test"]
    samples = []
    for mode in splits:
        t1_dir = root / mode / "T1"
        gt_t1_dir = root / mode / "GT_T1"
        if not t1_dir.exists() or not gt_t1_dir.exists():
            continue
        for f in t1_dir.iterdir():
            if f.suffix.lower() in (".tif", ".tiff", ".png"):
                samples.append((mode, f.stem))
    return samples


def load_sample(root: Path, mode: str, stem: str):
    """Load T1, T2, GT_T1, GT_T2, GT_CD for a sample (ChangeMamba SCD format)."""
    t1_dir = root / mode / "T1"
    t2_dir = root / mode / "T2"
    gt_t1_dir = root / mode / "GT_T1"
    gt_t2_dir = root / mode / "GT_T2"
    gt_cd_dir = root / mode / "GT_CD"

    # Determine image extension (JL1_second uses .png)
    candidates = list(t1_dir.glob(f"{stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"No image found for {stem} in {t1_dir}")
    img_ext = candidates[0].suffix

    t1_path = t1_dir / f"{stem}{img_ext}"
    t2_path = t2_dir / f"{stem}{img_ext}"
    gt_t1_path = gt_t1_dir / f"{stem}.png"
    gt_t2_path = gt_t2_dir / f"{stem}.png"
    gt_cd_path = gt_cd_dir / f"{stem}.png"

    for p in [t1_path, t2_path, gt_t1_path, gt_t2_path, gt_cd_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    t1 = np.asarray(imageio.imread(str(t1_path)))
    t2 = np.asarray(imageio.imread(str(t2_path)))
    gt_t1 = np.asarray(imageio.imread(str(gt_t1_path)))
    gt_t2 = np.asarray(imageio.imread(str(gt_t2_path)))
    gt_cd = np.asarray(imageio.imread(str(gt_cd_path)))

    # Ensure T1, T2 are HxWx3
    if t1.ndim == 2:
        t1 = np.stack([t1] * 3, axis=-1)
    elif t1.shape[-1] > 3:
        t1 = t1[..., :3]
    if t2.ndim == 2:
        t2 = np.stack([t2] * 3, axis=-1)
    elif t2.shape[-1] > 3:
        t2 = t2[..., :3]

    # GT_T1, GT_T2: single-channel class index; ensure 2D int
    if gt_t1.ndim == 3:
        gt_t1 = gt_t1[:, :, 0] if gt_t1.shape[-1] >= 1 else gt_t1.squeeze()
    gt_t1 = gt_t1.astype(np.int32)
    if gt_t2.ndim == 3:
        gt_t2 = gt_t2[:, :, 0] if gt_t2.shape[-1] >= 1 else gt_t2.squeeze()
    gt_t2 = gt_t2.astype(np.int32)

    # GT_CD: binary 0/255
    if gt_cd.ndim == 3:
        gt_cd = gt_cd[:, :, 0] if gt_cd.shape[-1] >= 1 else gt_cd.squeeze()
    gt_cd = gt_cd.astype(np.int32)

    return t1, t2, gt_t1, gt_t2, gt_cd


def plot_sample(t1, t2, gt_t1, gt_t2, gt_cd, stem: str, out_path: Path):
    """Create 2x3 visualization: T1, T2, GT_CD | GT_T1, GT_T2, legend."""
    gt_t1_color = index2color(gt_t1).copy()
    gt_t2_color = index2color(gt_t2).copy()
    # Use white background for unlabeled (class 0) pixels
    gt_t1_color[gt_t1 == 0] = [255, 255, 255]
    gt_t2_color[gt_t2 == 0] = [255, 255, 255]
    # GT_CD: 0=no change (black), 255=change (white)
    gt_cd_viz = np.stack([gt_cd] * 3, axis=-1).astype(np.uint8)

    from src.datasets.colormap import JL1_COLORMAP, JL1_CLASSES

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), facecolor="white")
    axes[0, 0].imshow(t1)
    axes[0, 0].set_title("T1 (before)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(t2)
    axes[0, 1].set_title("T2 (after)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gt_cd_viz)
    axes[0, 2].set_title("GT_CD (0=no change, 255=change)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gt_t1_color)
    axes[1, 0].set_title("GT_T1 (semantic)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt_t2_color)
    axes[1, 1].set_title("GT_T2 (semantic)")
    axes[1, 1].axis("off")

    # Legend in last subplot
    legend_patches = [
        plt.matplotlib.patches.Patch(color=np.array(c) / 255, label=name)
        for c, name in zip(JL1_COLORMAP, JL1_CLASSES)
    ]
    axes[1, 2].legend(handles=legend_patches, loc="center", fontsize=8)
    axes[1, 2].axis("off")

    fig.suptitle(f"JL1_second — {stem}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize random JL1_second sample")
    parser.add_argument(
        "--root",
        type=str,
        default="/root/workspace/sakura/ML4RS-Cropland-Recognition/datasets/JL1_second",
        help="Dataset root",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: dataset_root/jl1_vis_sample.png)",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=None,
        help="Splits to sample from (default: train val test)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    splits = args.split if args.split else ["train", "val", "test"]
    samples = collect_samples(root, splits=splits)
    if not samples:
        raise RuntimeError(
            f"No labeled samples found (need T1 + GT_T1) under requested splits of {root}"
        )

    if args.seed is not None:
        random.seed(args.seed)
    mode, stem = random.choice(samples)
    print(f"Random sample: {mode}/{stem}")

    t1, t2, gt_t1, gt_t2, gt_cd = load_sample(root, mode, stem)
    out_path = Path(args.out) if args.out else root / "jl1_vis_sample.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_sample(t1, t2, gt_t1, gt_t2, gt_cd, f"{mode}/{stem}", out_path)


if __name__ == "__main__":
    main()
