#!/usr/bin/env python3
"""
Reformat cropland datasets into ChangeMamba BCD collections layout.

Output root layout:
  datasets/cropland_bcd_collections/
    clcd/
      train|val|test/{T1,T2,GT} + train.txt/val.txt/test.txt
    jl1_competition/
      train|val/{T1,T2,GT} + train.txt/val.txt
    input_quick/
      train|val/{T1,T2,GT} + train.txt/val.txt
    train_all.txt
    val_all.txt
    test_all.txt

Global *_all.txt format:
  source/split/file.png
Example:
  clcd/train/00001.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1).astype(np.uint8, copy=False)
    if img.shape[-1] > 3:
        return img[..., :3].astype(np.uint8, copy=False)
    return img.astype(np.uint8, copy=False)


def _ensure_single_channel(label: np.ndarray) -> np.ndarray:
    if label.ndim == 3:
        return label[..., 0].astype(np.uint8, copy=False)
    return label.astype(np.uint8, copy=False)


def _label_to_binary_255(label: np.ndarray) -> np.ndarray:
    x = _ensure_single_channel(label)
    # Works for either binary labels or multi-class change codes.
    return (x > 0).astype(np.uint8) * 255


def _write_list(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def _prepare_split_dirs(dst_source_root: Path, split: str) -> dict[str, Path]:
    split_root = dst_source_root / split
    t1 = split_root / "T1"
    t2 = split_root / "T2"
    gt = split_root / "GT"
    t1.mkdir(parents=True, exist_ok=True)
    t2.mkdir(parents=True, exist_ok=True)
    gt.mkdir(parents=True, exist_ok=True)
    return {"split_root": split_root, "T1": t1, "T2": t2, "GT": gt}


def _copy_triplet(
    src_t1: Path,
    src_t2: Path,
    src_label: Path,
    out_dirs: dict[str, Path],
    out_name: str,
) -> None:
    t1 = _ensure_rgb(np.asarray(imageio.imread(str(src_t1))))
    t2 = _ensure_rgb(np.asarray(imageio.imread(str(src_t2))))
    gt = _label_to_binary_255(np.asarray(imageio.imread(str(src_label))))
    imageio.imwrite(str(out_dirs["T1"] / out_name), t1)
    imageio.imwrite(str(out_dirs["T2"] / out_name), t2)
    imageio.imwrite(str(out_dirs["GT"] / out_name), gt)


def convert_clcd(clcd_root: Path, dst_root: Path) -> dict[str, list[str]]:
    source_name = "clcd"
    out_source = dst_root / source_name
    out_source.mkdir(parents=True, exist_ok=True)

    all_lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_map = {"train": "train", "val": "val", "test": "test"}

    for split_in, split_out in split_map.items():
        in_split = clcd_root / split_in
        if not in_split.exists():
            continue
        out_dirs = _prepare_split_dirs(out_source, split_out)

        t1_dir = in_split / "time1"
        t2_dir = in_split / "time2"
        label_dir = in_split / "label"
        names = sorted(p.name for p in t1_dir.glob("*.png"))
        out_names: list[str] = []
        for name in names:
            p1 = t1_dir / name
            p2 = t2_dir / name
            pl = label_dir / name
            if not (p1.exists() and p2.exists() and pl.exists()):
                continue
            _copy_triplet(p1, p2, pl, out_dirs, name)
            out_names.append(name)

        _write_list(out_source / f"{split_out}.txt", out_names)
        all_lists[split_out] = [f"{source_name}/{split_out}/{n}" for n in out_names]
        print(f"[CLCD] {split_out}: {len(out_names)}")

    return all_lists


def convert_jl1_competition(
    jl1_root: Path,
    dst_root: Path,
    train_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    source_name = "jl1_competition"
    out_source = dst_root / source_name
    out_source.mkdir(parents=True, exist_ok=True)
    train_dirs = _prepare_split_dirs(out_source, "train")
    val_dirs = _prepare_split_dirs(out_source, "val")

    t1_dir = jl1_root / "train" / "pre"
    t2_dir = jl1_root / "train" / "post"
    label_dir = jl1_root / "train" / "label"
    names = sorted(p.name for p in t1_dir.glob("*.tif"))

    valid: list[str] = []
    for name in names:
        if (t2_dir / name).exists() and (label_dir / name.replace(".tif", ".png")).exists():
            valid.append(name)

    rnd = random.Random(seed)
    rnd.shuffle(valid)
    split_idx = int(len(valid) * train_ratio)
    train_names = sorted(valid[:split_idx])
    val_names = sorted(valid[split_idx:])

    out_train_names: list[str] = []
    for idx, name in enumerate(train_names):
        out_name = f"{idx:05d}.png"
        _copy_triplet(t1_dir / name, t2_dir / name, label_dir / name.replace(".tif", ".png"), train_dirs, out_name)
        out_train_names.append(out_name)

    out_val_names: list[str] = []
    for idx, name in enumerate(val_names):
        out_name = f"{idx:05d}.png"
        _copy_triplet(t1_dir / name, t2_dir / name, label_dir / name.replace(".tif", ".png"), val_dirs, out_name)
        out_val_names.append(out_name)

    _write_list(out_source / "train.txt", out_train_names)
    _write_list(out_source / "val.txt", out_val_names)
    _write_list(out_source / "test.txt", [])
    print(f"[JL1] train: {len(out_train_names)} val: {len(out_val_names)}")

    return {
        "train": [f"{source_name}/train/{n}" for n in out_train_names],
        "val": [f"{source_name}/val/{n}" for n in out_val_names],
        "test": [],
    }


def _infer_vector_label_field(props: dict[str, Any]) -> str:
    # Prefer explicit "change" style keys.
    preferred = [
        "change",
        "label",
        "class",
        "cls",
        "type",
        "category",
    ]
    lower_map = {str(k).lower(): str(k) for k in props.keys()}
    for key in preferred:
        if key in lower_map:
            return lower_map[key]

    # Fallback to first integer-like field.
    for k, v in props.items():
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            return str(k)
        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
            return str(k)
    raise ValueError("Could not infer integer label field from shapefile properties.")


def convert_input_quick(
    input_root: Path,
    dst_root: Path,
    patch_size: int,
    train_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    try:
        import fiona
        import rasterio
        from rasterio import features as rio_features
        from rasterio.warp import transform_geom
    except Exception as e:
        raise RuntimeError(
            "Converting data/input requires rasterio + fiona. "
            "Install and rerun: pip install rasterio fiona"
        ) from e

    source_name = "input_quick"
    out_source = dst_root / source_name
    out_source.mkdir(parents=True, exist_ok=True)
    train_dirs = _prepare_split_dirs(out_source, "train")
    val_dirs = _prepare_split_dirs(out_source, "val")

    t1_candidates = sorted((input_root / "T1").glob("*.tif"))
    t2_candidates = sorted((input_root / "T2").glob("*.tif"))
    shp_candidates = sorted((input_root / "label").glob("*.shp"))
    if not t1_candidates or not t2_candidates or not shp_candidates:
        raise FileNotFoundError(
            "Expected data/input/T1/*.tif, data/input/T2/*.tif, and data/input/label/*.shp"
        )

    # Use first pair by convention in this quick dataset.
    t1_path = t1_candidates[0]
    t2_path = t2_candidates[0]
    shp_path = shp_candidates[0]

    with rasterio.open(str(t1_path)) as ds1, rasterio.open(str(t2_path)) as ds2:
        t1 = ds1.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8, copy=False)
        # Reproject T2 to T1 grid when needed.
        if (ds1.height, ds1.width) != (ds2.height, ds2.width) or ds1.transform != ds2.transform or ds1.crs != ds2.crs:
            dst = np.zeros((3, ds1.height, ds1.width), dtype=np.float32)
            for ch in range(3):
                rasterio.warp.reproject(
                    source=rasterio.band(ds2, ch + 1),
                    destination=dst[ch],
                    src_transform=ds2.transform,
                    src_crs=ds2.crs,
                    dst_transform=ds1.transform,
                    dst_crs=ds1.crs,
                    resampling=rasterio.enums.Resampling.bilinear,
                    dst_nodata=0.0,
                )
            t2 = np.transpose(np.clip(dst, 0, 255).astype(np.uint8), (1, 2, 0))
        else:
            t2 = ds2.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8, copy=False)

        with fiona.open(str(shp_path)) as src:
            vec_crs = rasterio.crs.CRS.from_user_input(src.crs) if src.crs else None
            features = list(src)
            if not features:
                raise ValueError(f"No features in shapefile: {shp_path}")
            label_field = _infer_vector_label_field(features[0].get("properties") or {})

            shapes = []
            for feat in features:
                geom = feat.get("geometry")
                if geom is None:
                    continue
                props = feat.get("properties") or {}
                raw = props.get(label_field, 0)
                if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
                    raw = int(raw.strip())
                value = 255 if int(raw) > 0 else 0
                if vec_crs is not None and ds1.crs is not None and vec_crs != ds1.crs:
                    geom = transform_geom(vec_crs, ds1.crs, geom)
                shapes.append((geom, value))

            if not shapes:
                raise ValueError(f"No valid geometries in shapefile: {shp_path}")
            mask = rio_features.rasterize(
                shapes,
                out_shape=(ds1.height, ds1.width),
                transform=ds1.transform,
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            )

    h, w = mask.shape
    nh = int(np.ceil(h / patch_size))
    nw = int(np.ceil(w / patch_size))
    ph = nh * patch_size - h
    pw = nw * patch_size - w
    if ph > 0 or pw > 0:
        t1_pad = np.zeros((h + ph, w + pw, 3), dtype=np.uint8)
        t2_pad = np.zeros((h + ph, w + pw, 3), dtype=np.uint8)
        m_pad = np.zeros((h + ph, w + pw), dtype=np.uint8)
        t1_pad[:h, :w] = t1
        t2_pad[:h, :w] = t2
        m_pad[:h, :w] = mask
        t1, t2, mask = t1_pad, t2_pad, m_pad

    tile_names: list[str] = [f"{i:05d}.png" for i in range(nh * nw)]

    # Write tiles after deterministic split assignment.
    rnd = random.Random(seed)
    rnd.shuffle(tile_names)
    split_idx = int(len(tile_names) * train_ratio)
    train_set = set(tile_names[:split_idx])
    val_set = set(tile_names[split_idx:])
    train_names = sorted(train_set)
    val_names = sorted(val_set)

    idx = 0
    for r in range(nh):
        for c in range(nw):
            y0, x0 = r * patch_size, c * patch_size
            y1, x1 = y0 + patch_size, x0 + patch_size
            name = f"{idx:05d}.png"
            idx += 1
            out_dirs = train_dirs if name in train_set else val_dirs
            imageio.imwrite(str(out_dirs["T1"] / name), t1[y0:y1, x0:x1])
            imageio.imwrite(str(out_dirs["T2"] / name), t2[y0:y1, x0:x1])
            gt = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
            imageio.imwrite(str(out_dirs["GT"] / name), gt)

    _write_list(out_source / "train.txt", train_names)
    _write_list(out_source / "val.txt", val_names)
    _write_list(out_source / "test.txt", [])
    print(f"[INPUT_QUICK] train: {len(train_names)} val: {len(val_names)}")

    return {
        "train": [f"{source_name}/train/{n}" for n in train_names],
        "val": [f"{source_name}/val/{n}" for n in val_names],
        "test": [],
    }


def merge_global_lists(dst_root: Path, all_entries: list[dict[str, list[str]]]) -> None:
    train_all: list[str] = []
    val_all: list[str] = []
    test_all: list[str] = []
    for entries in all_entries:
        train_all.extend(entries.get("train", []))
        val_all.extend(entries.get("val", []))
        test_all.extend(entries.get("test", []))
    train_all = sorted(train_all)
    val_all = sorted(val_all)
    test_all = sorted(test_all)
    _write_list(dst_root / "train_all.txt", train_all)
    _write_list(dst_root / "val_all.txt", val_all)
    _write_list(dst_root / "test_all.txt", test_all)
    print(f"[ALL] train: {len(train_all)} val: {len(val_all)} test: {len(test_all)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reformat cropland datasets into BCD collections format.")
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Output collections root. Default: <project_root>/datasets/cropland_bcd_collections",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio for sources without val split.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for data/input tiling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (project_root / "datasets" / "cropland_bcd_collections").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    clcd_root = project_root / "datasets" / "CLCD"
    jl1_root = project_root / "datasets" / "jl1_cropland_competition_2023"
    input_root = project_root / "data" / "input"

    all_entries: list[dict[str, list[str]]] = []
    all_entries.append(convert_clcd(clcd_root, output_root))
    all_entries.append(convert_jl1_competition(jl1_root, output_root, train_ratio=args.train_ratio, seed=args.seed))
    all_entries.append(
        convert_input_quick(
            input_root,
            output_root,
            patch_size=args.patch_size,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    )
    merge_global_lists(output_root, all_entries)
    print(f"Done. Output root: {output_root}")


if __name__ == "__main__":
    main()
