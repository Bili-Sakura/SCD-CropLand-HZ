#!/usr/bin/env python3
"""
Populate datasets/JL1_second/test from jl1_cropland_competition_2023/test.

Creates test/T1 and test/T2 as PNGs (same naming as val: 00001.png, …) with no GT folders.
Overwrites datasets/JL1_second/test.txt.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[-1] > 3:
        return img[..., :3].copy()
    return img


def natural_tif_paths(pre_dir: Path) -> list[Path]:
    def sort_key(p: Path) -> tuple[int, str]:
        m = re.match(r"image_(\d+)$", p.stem, re.I)
        return (int(m.group(1)), p.name) if m else (10**12, p.name)

    paths = sorted(
        [p for p in pre_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")],
        key=sort_key,
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jl1-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "JL1_second",
    )
    parser.add_argument(
        "--competition-test",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "jl1_cropland_competition_2023"
        / "test",
    )
    args = parser.parse_args()
    jl1_root: Path = args.jl1_root
    comp_test: Path = args.competition_test
    pre_dir = comp_test / "pre"
    post_dir = comp_test / "post"
    if not pre_dir.is_dir() or not post_dir.is_dir():
        raise FileNotFoundError(f"Expected {pre_dir} and {post_dir}")

    test_root = jl1_root / "test"
    if test_root.exists():
        shutil.rmtree(test_root)
    t1_dir = test_root / "T1"
    t2_dir = test_root / "T2"
    t1_dir.mkdir(parents=True)
    t2_dir.mkdir(parents=True)

    pre_paths = natural_tif_paths(pre_dir)
    try:
        from tqdm import tqdm

        iterator = tqdm(enumerate(pre_paths, start=1), total=len(pre_paths), desc="test")
    except ImportError:
        iterator = enumerate(pre_paths, start=1)

    names_out: list[str] = []
    for idx, pre_path in iterator:
        post_path = post_dir / pre_path.name
        if not post_path.exists():
            post_path = post_dir / (pre_path.stem + ".tif")
        if not post_path.exists():
            raise FileNotFoundError(f"Missing post for {pre_path.name}")

        new_name = f"{idx:05d}"
        for src, dst in [
            (pre_path, t1_dir / f"{new_name}.png"),
            (post_path, t2_dir / f"{new_name}.png"),
        ]:
            img = imageio.imread(str(src))
            img = ensure_3ch(np.asarray(img, dtype=np.uint8))
            imageio.imwrite(str(dst), img)
        names_out.append(new_name)

    list_path = jl1_root / "test.txt"
    with open(list_path, "w") as f:
        for n in names_out:
            f.write(f"{n}.png\n")
    print(f"Wrote {len(names_out)} pairs -> {test_root} and {list_path}")


if __name__ == "__main__":
    main()
