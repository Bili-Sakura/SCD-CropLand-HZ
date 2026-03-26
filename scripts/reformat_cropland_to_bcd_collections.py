#!/usr/bin/env python3
"""
Reformat cropland datasets into ChangeMamba BCD collections layout.

Output root layout:
  datasets/cropland_bcd_collections/
    clcd/
      train|val|test/{T1,T2,GT} + train.txt/val.txt/test.txt
    hrscd/
      train/{T1,T2,GT} + train.txt
    fpcd/
      train/{T1,T2,GT} + train.txt
    hi_cna/
      train/{T1,T2,GT} + train.txt
    jl1_competition/
      train|val/{T1,T2,GT} + train.txt/val.txt
    input_quick/
      train|val/{T1,T2,GT} + train.txt/val.txt
    train_stage2_joint.txt
    train_all.txt
    val_all.txt
    test_all.txt

Manifest format:
  source/split/file.png
Example:
  clcd/train/00001.png
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

_T = TypeVar("_T")
_R = TypeVar("_R")

# Set False via --no-progress or when embedding this module without bars.
_TQDM_DISABLE = False
_PNG_COMPRESS_LEVEL = 1


def _pbar(iterable: Iterable[_T], desc: str, **kwargs: Any) -> Iterable[_T]:
    if _TQDM_DISABLE:
        return iterable
    return tqdm(iterable, desc=desc, **kwargs)


def _write_png(
    path: Path,
    arr: np.ndarray,
    delete_existing: bool = False,
    compress_level: int | None = None,
) -> None:
    level = _PNG_COMPRESS_LEVEL if compress_level is None else int(compress_level)
    if delete_existing and path.exists():
        path.unlink()
    try:
        imageio.imwrite(str(path), arr, compress_level=level)
    except TypeError:
        # Some imageio backends ignore/deny compress_level kwargs.
        imageio.imwrite(str(path), arr)


def _recompress_png_job(path_str: str, level: int) -> tuple[int, int]:
    path = Path(path_str)
    if not path.exists():
        return (0, 0)
    before = path.stat().st_size
    arr = np.asarray(imageio.imread(str(path)))
    _write_png(path, arr, delete_existing=True, compress_level=level)
    after = path.stat().st_size if path.exists() else 0
    return (before, after)


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1).astype(np.uint8, copy=False)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32, copy=False)
    lo = np.min(img_f, axis=(0, 1), keepdims=True)
    hi = np.max(img_f, axis=(0, 1), keepdims=True)
    den = np.where((hi - lo) > 0, (hi - lo), 1.0)
    scaled = (img_f - lo) * (255.0 / den)
    return np.clip(scaled, 0, 255).astype(np.uint8)


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
    _write_png(out_dirs["T1"] / out_name, t1)
    _write_png(out_dirs["T2"] / out_name, t2)
    _write_png(out_dirs["GT"] / out_name, gt)


def _triplet_exists(out_t1: str, out_t2: str, out_gt: str, out_name: str) -> bool:
    return (
        (Path(out_t1) / out_name).exists()
        and (Path(out_t2) / out_name).exists()
        and (Path(out_gt) / out_name).exists()
    )


def _expected_tile_names(h: int, w: int, patch_size: int, out_name_prefix: str) -> list[str]:
    ys = _axis_starts(h, patch_size)
    xs = _axis_starts(w, patch_size)
    total = len(ys) * len(xs)
    return [f"{out_name_prefix}_{i:03d}.png" for i in range(total)]


def _all_triplets_exist(out_t1: str, out_t2: str, out_gt: str, names: list[str]) -> bool:
    if not names:
        return False
    for n in names:
        if not _triplet_exists(out_t1, out_t2, out_gt, n):
            return False
    return True


def _resolve_workers(workers: int) -> int:
    if workers > 0:
        return workers
    cpu_count = os.cpu_count() or 1
    return max(1, min(16, cpu_count))


def _run_jobs(
    jobs: list[tuple[Any, ...]],
    fn: Callable[..., _R],
    desc: str,
    unit: str,
    workers: int,
) -> list[_R]:
    if not jobs:
        return []

    if workers <= 1:
        out: list[_R] = []
        for job in _pbar(jobs, desc=desc, unit=unit):
            out.append(fn(*job))
        return out

    out: list[_R] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, *job) for job in jobs]
        for fut in _pbar(as_completed(futures), desc=desc, total=len(futures), unit=unit):
            out.append(fut.result())
    return out


def _copy_triplet_job(
    src_t1: str,
    src_t2: str,
    src_label: str,
    out_t1: str,
    out_t2: str,
    out_gt: str,
    out_name: str,
) -> str:
    if _triplet_exists(out_t1, out_t2, out_gt, out_name):
        return out_name
    out_dirs = {"T1": Path(out_t1), "T2": Path(out_t2), "GT": Path(out_gt)}
    _copy_triplet(Path(src_t1), Path(src_t2), Path(src_label), out_dirs, out_name)
    return out_name


def _tile_triplet_raster_label_job(
    src_t1: str,
    src_t2: str,
    src_label: str,
    out_t1: str,
    out_t2: str,
    out_gt: str,
    out_name_prefix: str,
    patch_size: int,
) -> list[str]:
    import rasterio

    with rasterio.open(src_label) as ds:
        expected = _expected_tile_names(ds.height, ds.width, patch_size, out_name_prefix)
    if _all_triplets_exist(out_t1, out_t2, out_gt, expected):
        return expected

    with rasterio.open(src_t1) as ds:
        t1 = ds.read()
    with rasterio.open(src_t2) as ds:
        t2 = ds.read()
    with rasterio.open(src_label) as ds:
        gt = ds.read(1)

    out_dirs = {"T1": Path(out_t1), "T2": Path(out_t2), "GT": Path(out_gt)}
    return _tile_and_write_triplet(t1, t2, gt, out_dirs, out_name_prefix, patch_size=patch_size)


def _tile_triplet_imageio_label_job(
    src_t1: str,
    src_t2: str,
    src_label: str,
    out_t1: str,
    out_t2: str,
    out_gt: str,
    out_name_prefix: str,
    patch_size: int,
) -> list[str]:
    gt = np.asarray(imageio.imread(src_label))
    expected = _expected_tile_names(gt.shape[0], gt.shape[1], patch_size, out_name_prefix)
    if _all_triplets_exist(out_t1, out_t2, out_gt, expected):
        return expected

    t1 = np.asarray(imageio.imread(src_t1))
    t2 = np.asarray(imageio.imread(src_t2))
    out_dirs = {"T1": Path(out_t1), "T2": Path(out_t2), "GT": Path(out_gt)}
    return _tile_and_write_triplet(t1, t2, gt, out_dirs, out_name_prefix, patch_size=patch_size)


def _tile_triplet_raster_png_label_job(
    src_t1: str,
    src_t2: str,
    src_label_png: str,
    out_t1: str,
    out_t2: str,
    out_gt: str,
    out_name_prefix: str,
    patch_size: int,
) -> list[str]:
    import rasterio

    gt = np.asarray(imageio.imread(src_label_png))
    expected = _expected_tile_names(gt.shape[0], gt.shape[1], patch_size, out_name_prefix)
    if _all_triplets_exist(out_t1, out_t2, out_gt, expected):
        return expected

    with rasterio.open(src_t1) as ds:
        t1 = ds.read()
    with rasterio.open(src_t2) as ds:
        t2 = ds.read()
    out_dirs = {"T1": Path(out_t1), "T2": Path(out_t2), "GT": Path(out_gt)}
    return _tile_and_write_triplet(t1, t2, gt, out_dirs, out_name_prefix, patch_size=patch_size)


def _axis_starts(length: int, patch_size: int) -> list[int]:
    if length < patch_size:
        raise ValueError(f"Axis length {length} is smaller than patch size {patch_size}.")
    n = int(np.ceil(length / patch_size))
    if n <= 1:
        return [0]
    starts = np.linspace(0, length - patch_size, n, dtype=int).tolist()
    starts[0] = 0
    starts[-1] = length - patch_size
    return starts


def _tile_and_write_triplet(
    t1: np.ndarray,
    t2: np.ndarray,
    label: np.ndarray,
    out_dirs: dict[str, Path],
    out_name_prefix: str,
    patch_size: int,
) -> list[str]:
    t1_u8 = _ensure_rgb(t1)
    t2_u8 = _ensure_rgb(t2)
    gt_u8 = _label_to_binary_255(label)
    # Some datasets (e.g. FPCD) contain occasional off-by-few-pixels pairs.
    # Align by cropping to common overlap, then pad to patch_size if needed.
    h = min(t1_u8.shape[0], t2_u8.shape[0], gt_u8.shape[0])
    w = min(t1_u8.shape[1], t2_u8.shape[1], gt_u8.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid shapes: t1={t1_u8.shape}, t2={t2_u8.shape}, label={gt_u8.shape}")
    if t1_u8.shape[:2] != (h, w):
        t1_u8 = t1_u8[:h, :w]
    if t2_u8.shape[:2] != (h, w):
        t2_u8 = t2_u8[:h, :w]
    if gt_u8.shape[:2] != (h, w):
        gt_u8 = gt_u8[:h, :w]

    if h < patch_size or w < patch_size:
        h2 = max(h, patch_size)
        w2 = max(w, patch_size)
        t1_pad = np.zeros((h2, w2, 3), dtype=np.uint8)
        t2_pad = np.zeros((h2, w2, 3), dtype=np.uint8)
        gt_pad = np.zeros((h2, w2), dtype=np.uint8)
        t1_pad[:h, :w] = t1_u8
        t2_pad[:h, :w] = t2_u8
        gt_pad[:h, :w] = gt_u8
        t1_u8, t2_u8, gt_u8 = t1_pad, t2_pad, gt_pad

    h, w = gt_u8.shape
    ys = _axis_starts(h, patch_size)
    xs = _axis_starts(w, patch_size)

    out_names: list[str] = []
    tile_idx = 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + patch_size, x0 + patch_size
            out_name = f"{out_name_prefix}_{tile_idx:03d}.png"
            tile_idx += 1
            out_t1 = out_dirs["T1"] / out_name
            out_t2 = out_dirs["T2"] / out_name
            out_gt = out_dirs["GT"] / out_name
            if out_t1.exists() and out_t2.exists() and out_gt.exists():
                out_names.append(out_name)
                continue
            _write_png(out_t1, t1_u8[y0:y1, x0:x1])
            _write_png(out_t2, t2_u8[y0:y1, x0:x1])
            _write_png(out_gt, gt_u8[y0:y1, x0:x1])
            out_names.append(out_name)
    return out_names


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def convert_clcd(clcd_root: Path, dst_root: Path, workers: int = 1) -> dict[str, list[str]]:
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
        jobs: list[tuple[Any, ...]] = []
        for name in names:
            p1 = t1_dir / name
            p2 = t2_dir / name
            pl = label_dir / name
            if not (p1.exists() and p2.exists() and pl.exists()):
                continue
            jobs.append(
                (
                    str(p1),
                    str(p2),
                    str(pl),
                    str(out_dirs["T1"]),
                    str(out_dirs["T2"]),
                    str(out_dirs["GT"]),
                    name,
                )
            )
        out_names = sorted(_run_jobs(jobs, _copy_triplet_job, desc=f"CLCD/{split_out}", unit="img", workers=workers))

        _write_list(out_source / f"{split_out}.txt", out_names)
        all_lists[split_out] = [f"{source_name}/{split_out}/{n}" for n in out_names]
        print(f"[CLCD] {split_out}: {len(out_names)}")

    return all_lists


def convert_jl1_competition(
    jl1_root: Path,
    dst_root: Path,
    train_ratio: float,
    seed: int,
    workers: int = 1,
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

    train_jobs: list[tuple[Any, ...]] = []
    for idx, name in enumerate(train_names):
        out_name = f"{idx:05d}.png"
        train_jobs.append(
            (
                str(t1_dir / name),
                str(t2_dir / name),
                str(label_dir / name.replace(".tif", ".png")),
                str(train_dirs["T1"]),
                str(train_dirs["T2"]),
                str(train_dirs["GT"]),
                out_name,
            )
        )
    out_train_names = sorted(_run_jobs(train_jobs, _copy_triplet_job, desc="JL1/train", unit="img", workers=workers))

    val_jobs: list[tuple[Any, ...]] = []
    for idx, name in enumerate(val_names):
        out_name = f"{idx:05d}.png"
        val_jobs.append(
            (
                str(t1_dir / name),
                str(t2_dir / name),
                str(label_dir / name.replace(".tif", ".png")),
                str(val_dirs["T1"]),
                str(val_dirs["T2"]),
                str(val_dirs["GT"]),
                out_name,
            )
        )
    out_val_names = sorted(_run_jobs(val_jobs, _copy_triplet_job, desc="JL1/val", unit="img", workers=workers))

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
    tile_total = nh * nw
    rc_iter = ((r, c) for r in range(nh) for c in range(nw))
    for r, c in _pbar(rc_iter, desc="input_quick/tiles", total=tile_total, unit="tile"):
        y0, x0 = r * patch_size, c * patch_size
        y1, x1 = y0 + patch_size, x0 + patch_size
        name = f"{idx:05d}.png"
        idx += 1
        out_dirs = train_dirs if name in train_set else val_dirs
        _write_png(out_dirs["T1"] / name, t1[y0:y1, x0:x1])
        _write_png(out_dirs["T2"] / name, t2[y0:y1, x0:x1])
        gt = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        _write_png(out_dirs["GT"] / name, gt)

    _write_list(out_source / "train.txt", train_names)
    _write_list(out_source / "val.txt", val_names)
    _write_list(out_source / "test.txt", [])
    print(f"[INPUT_QUICK] train: {len(train_names)} val: {len(val_names)}")

    return {
        "train": [f"{source_name}/train/{n}" for n in train_names],
        "val": [f"{source_name}/val/{n}" for n in val_names],
        "test": [],
    }


def convert_hrscd(hrscd_root: Path, dst_root: Path, patch_size: int, workers: int = 1) -> dict[str, list[str]]:
    try:
        import rasterio
    except Exception as e:
        raise RuntimeError("Converting HRSCD requires rasterio. Install and rerun: pip install rasterio") from e

    source_name = "hrscd"
    out_source = dst_root / source_name
    out_source.mkdir(parents=True, exist_ok=True)
    out_dirs = _prepare_split_dirs(out_source, "train")

    d2006 = hrscd_root / "2006"
    d2012 = hrscd_root / "2012"
    dchg = hrscd_root / "change"
    if not (d2006.exists() and d2012.exists() and dchg.exists()):
        print("[HRSCD] skipped (missing 2006/2012/change folders).")
        _write_list(out_source / "train.txt", [])
        _write_list(out_source / "val.txt", [])
        _write_list(out_source / "test.txt", [])
        return {"train": [], "val": [], "test": []}

    names_train: list[str] = []
    matched_pairs = 0
    skipped_pairs = 0
    jobs: list[tuple[Any, ...]] = []

    p2012_list = sorted(d2012.rglob("*.tif"))
    for p2012 in _pbar(p2012_list, desc="HRSCD/pairs", unit="pair"):
        rel = p2012.relative_to(d2012)
        pchg = dchg / rel
        if not pchg.exists():
            skipped_pairs += 1
            continue
        parts = p2012.stem.split("-")
        if len(parts) < 4:
            skipped_pairs += 1
            continue
        zone, gx, gy = parts[0], parts[2], parts[3]
        cands_local = sorted((d2006 / rel.parent).glob(f"{zone}-*-{gx}-{gy}-LA93.tif"))
        cands_global = cands_local if cands_local else sorted(d2006.rglob(f"{zone}-*-{gx}-{gy}-LA93.tif"))
        if not cands_global:
            skipped_pairs += 1
            continue
        p2006 = cands_global[0]

        pair_base = _safe_stem(p2012.stem)
        jobs.append(
            (
                str(p2006),
                str(p2012),
                str(pchg),
                str(out_dirs["T1"]),
                str(out_dirs["T2"]),
                str(out_dirs["GT"]),
                pair_base,
                patch_size,
            )
        )

    for out_names in _run_jobs(jobs, _tile_triplet_raster_label_job, desc="HRSCD/write", unit="pair", workers=workers):
        names_train.extend(out_names)
    matched_pairs = len(jobs)

    names_train = sorted(names_train)
    _write_list(out_source / "train.txt", names_train)
    _write_list(out_source / "val.txt", [])
    _write_list(out_source / "test.txt", [])
    print(f"[HRSCD] pairs: {matched_pairs}, skipped: {skipped_pairs}, patches: {len(names_train)}")
    return {
        "train": [f"{source_name}/train/{n}" for n in names_train],
        "val": [],
        "test": [],
    }


def _fpcd_pair_key_from_temporal(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 3:
        return stem
    idx = parts[-1]
    village = "_".join(parts[:-2])
    return f"{village}_{idx}"


def convert_fpcd(fpcd_root: Path, dst_root: Path, patch_size: int, workers: int = 1) -> dict[str, list[str]]:
    source_name = "fpcd"
    out_source = dst_root / source_name
    out_source.mkdir(parents=True, exist_ok=True)
    out_dirs = _prepare_split_dirs(out_source, "train")

    t0_dir = fpcd_root / "T0"
    t1_dir = fpcd_root / "T1"
    mask_dir = fpcd_root / "task_3_masks"
    if not (t0_dir.exists() and t1_dir.exists() and mask_dir.exists()):
        print("[FPCD] skipped (missing T0/T1/task_3_masks folders).")
        _write_list(out_source / "train.txt", [])
        _write_list(out_source / "val.txt", [])
        _write_list(out_source / "test.txt", [])
        return {"train": [], "val": [], "test": []}

    t0_map = {_fpcd_pair_key_from_temporal(p.stem): p for p in t0_dir.glob("*.jpg")}
    t1_map = {_fpcd_pair_key_from_temporal(p.stem): p for p in t1_dir.glob("*.jpg")}
    m_map = {p.stem: p for p in mask_dir.glob("*.jpg")}
    keys = sorted(set(t0_map.keys()) & set(t1_map.keys()) & set(m_map.keys()))

    names_train: list[str] = []
    jobs: list[tuple[Any, ...]] = []
    for key in keys:
        p0 = t0_map[key]
        p1 = t1_map[key]
        pm = m_map[key]
        jobs.append(
            (
                str(p0),
                str(p1),
                str(pm),
                str(out_dirs["T1"]),
                str(out_dirs["T2"]),
                str(out_dirs["GT"]),
                _safe_stem(key),
                patch_size,
            )
        )
    for out_names in _run_jobs(jobs, _tile_triplet_imageio_label_job, desc="FPCD/write", unit="pair", workers=workers):
        names_train.extend(out_names)

    names_train = sorted(names_train)
    _write_list(out_source / "train.txt", names_train)
    _write_list(out_source / "val.txt", [])
    _write_list(out_source / "test.txt", [])
    print(
        f"[FPCD] matched pairs: {len(keys)}, patches: {len(names_train)}, "
        f"missing keys: {len(set(m_map.keys()) - set(keys))}"
    )
    return {
        "train": [f"{source_name}/train/{n}" for n in names_train],
        "val": [],
        "test": [],
    }


def convert_hi_cna(hi_root: Path, dst_root: Path, patch_size: int, workers: int = 1) -> dict[str, list[str]]:
    try:
        import rasterio
    except Exception as e:
        raise RuntimeError("Converting Hi-CNA requires rasterio. Install and rerun: pip install rasterio") from e

    source_name = "hi_cna"
    out_source = dst_root / source_name
    out_source.mkdir(parents=True, exist_ok=True)
    out_dirs = _prepare_split_dirs(out_source, "train")

    if not hi_root.exists():
        print("[Hi-CNA] skipped (dataset root missing).")
        _write_list(out_source / "train.txt", [])
        _write_list(out_source / "val.txt", [])
        _write_list(out_source / "test.txt", [])
        return {"train": [], "val": [], "test": []}

    names_train: list[str] = []
    used = 0
    skipped = 0
    hi_jobs: list[tuple[Any, ...]] = []
    for split in ("train", "val"):
        split_root = hi_root / split
        d1 = split_root / "image1"
        d2 = split_root / "image2"
        dc = split_root / "change"
        if not (d1.exists() and d2.exists() and dc.exists()):
            continue
        for p1 in sorted(d1.glob("*.tif")):
            p2 = d2 / p1.name
            pc = dc / f"{p1.stem}.png"
            if not (p2.exists() and pc.exists()):
                skipped += 1
                continue
            base = _safe_stem(f"{split}_{p1.stem}")
            hi_jobs.append(
                (
                    str(p1),
                    str(p2),
                    str(pc),
                    str(out_dirs["T1"]),
                    str(out_dirs["T2"]),
                    str(out_dirs["GT"]),
                    base,
                    patch_size,
                )
            )

    for out_names in _run_jobs(hi_jobs, _tile_triplet_raster_png_label_job, desc="Hi-CNA/write", unit="pair", workers=workers):
        names_train.extend(out_names)
    used = len(hi_jobs)

    names_train = sorted(names_train)
    _write_list(out_source / "train.txt", names_train)
    _write_list(out_source / "val.txt", [])
    _write_list(out_source / "test.txt", [])
    print(f"[Hi-CNA] pairs: {used}, skipped: {skipped}, patches: {len(names_train)}")
    return {
        "train": [f"{source_name}/train/{n}" for n in names_train],
        "val": [],
        "test": [],
    }


def write_stage2_joint_manifest(dst_root: Path, by_source: dict[str, dict[str, list[str]]]) -> None:
    stage2_entries: list[str] = []
    for source_name in ("clcd", "fpcd", "hi_cna", "hrscd"):
        entries = by_source.get(source_name)
        if not entries:
            continue
        stage2_entries.extend(entries.get("train", []))
        stage2_entries.extend(entries.get("val", []))
        stage2_entries.extend(entries.get("test", []))
    stage2_entries = sorted(stage2_entries)
    _write_list(dst_root / "train_stage2_joint.txt", stage2_entries)
    print(f"[STAGE2] train_stage2_joint.txt: {len(stage2_entries)}")


def write_stage1_jl1_manifest(dst_root: Path, by_source: dict[str, dict[str, list[str]]]) -> None:
    jl1 = by_source.get("jl1_competition", {})
    # Stage 1 uses the full JL1 labeled pool (train+val) as one training list.
    stage1_entries = sorted(jl1.get("train", []) + jl1.get("val", []) + jl1.get("test", []))

    # Robustness: if the JL1 source root is missing, convert_jl1_competition may produce empty
    # lists, but previously-converted tiles may still exist under output_root/jl1_competition.
    # Recover in that case so we don't silently write an empty training manifest.
    if not stage1_entries:
        recovered: list[str] = []
        src_root = dst_root / "jl1_competition"
        for split in ("train", "val"):
            t1 = src_root / split / "T1"
            t2 = src_root / split / "T2"
            gt = src_root / split / "GT"
            if not (t1.is_dir() and t2.is_dir() and gt.is_dir()):
                continue
            for p in sorted(t1.glob("*.png")):
                name = p.name
                if (t2 / name).is_file() and (gt / name).is_file():
                    recovered.append(f"jl1_competition/{split}/{name}")
        stage1_entries = sorted(set(recovered))
        if stage1_entries:
            print(f"[STAGE1] recovered jl1_competition entries from output_root: {len(stage1_entries)}")

    if not stage1_entries:
        raise RuntimeError(
            "Stage-1 manifest would be empty: no jl1_competition entries found. "
            "Either provide raw JL1 at datasets/jl1_cropland_competition_2023, "
            "or ensure converted tiles exist under datasets/cropland_bcd_collections/jl1_competition."
        )

    _write_list(dst_root / "train_stage1_jl1.txt", stage1_entries)
    print(f"[STAGE1] train_stage1_jl1.txt: {len(stage1_entries)}")


def write_stage3_input_quick_manifests(dst_root: Path, by_source: dict[str, dict[str, list[str]]]) -> None:
    iq = by_source.get("input_quick", {})
    train_entries = sorted(iq.get("train", []))
    val_entries = sorted(iq.get("val", []))
    trainval_entries = sorted(train_entries + val_entries)
    _write_list(dst_root / "train_stage3_input_quick_trainval.txt", trainval_entries)
    _write_list(dst_root / "val_stage3_input_quick_val.txt", val_entries)
    print(
        f"[STAGE3] train_stage3_input_quick_trainval.txt: {len(trainval_entries)} "
        f"val_stage3_input_quick_val.txt: {len(val_entries)}"
    )


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


def _folder_png_stats(path: Path) -> tuple[int, int]:
    files = list(path.glob("*.png"))
    return len(files), sum(f.stat().st_size for f in files)


def _estimate_total_hrscd_tiles(hrscd_root: Path, patch_size: int) -> tuple[int, int]:
    import rasterio

    d2006 = hrscd_root / "2006"
    d2012 = hrscd_root / "2012"
    dchg = hrscd_root / "change"
    if not (d2006.exists() and d2012.exists() and dchg.exists()):
        return (0, 0)

    pairs = 0
    total_tiles = 0
    for p2012 in sorted(d2012.rglob("*.tif")):
        rel = p2012.relative_to(d2012)
        pchg = dchg / rel
        if not pchg.exists():
            continue
        parts = p2012.stem.split("-")
        if len(parts) < 4:
            continue
        zone, gx, gy = parts[0], parts[2], parts[3]
        cands_local = sorted((d2006 / rel.parent).glob(f"{zone}-*-{gx}-{gy}-LA93.tif"))
        cands = cands_local if cands_local else sorted(d2006.rglob(f"{zone}-*-{gx}-{gy}-LA93.tif"))
        if not cands:
            continue
        with rasterio.open(pchg) as ds:
            h, w = ds.height, ds.width
        total_tiles += int(np.ceil(h / patch_size) * np.ceil(w / patch_size))
        pairs += 1
    return pairs, total_tiles


def _estimate_total_fpcd_tiles(fpcd_root: Path, patch_size: int) -> tuple[int, int]:
    t0_dir = fpcd_root / "T0"
    t1_dir = fpcd_root / "T1"
    mask_dir = fpcd_root / "task_3_masks"
    if not (t0_dir.exists() and t1_dir.exists() and mask_dir.exists()):
        return (0, 0)

    t0_map = {_fpcd_pair_key_from_temporal(p.stem): p for p in t0_dir.glob("*.jpg")}
    t1_map = {_fpcd_pair_key_from_temporal(p.stem): p for p in t1_dir.glob("*.jpg")}
    m_map = {p.stem: p for p in mask_dir.glob("*.jpg")}
    keys = sorted(set(t0_map.keys()) & set(t1_map.keys()) & set(m_map.keys()))
    total_tiles = 0
    for key in keys:
        gt = np.asarray(imageio.imread(str(m_map[key])))
        h, w = gt.shape[:2]
        total_tiles += int(np.ceil(h / patch_size) * np.ceil(w / patch_size))
    return len(keys), total_tiles


def _estimate_total_hicna_tiles(hi_root: Path, patch_size: int) -> tuple[int, int]:
    if not hi_root.exists():
        return (0, 0)
    pairs = 0
    total_tiles = 0
    for split in ("train", "val"):
        d1 = hi_root / split / "image1"
        d2 = hi_root / split / "image2"
        dc = hi_root / split / "change"
        if not (d1.exists() and d2.exists() and dc.exists()):
            continue
        for p1 in sorted(d1.glob("*.tif")):
            p2 = d2 / p1.name
            pc = dc / f"{p1.stem}.png"
            if not (p2.exists() and pc.exists()):
                continue
            gt = np.asarray(imageio.imread(str(pc)))
            h, w = gt.shape[:2]
            total_tiles += int(np.ceil(h / patch_size) * np.ceil(w / patch_size))
            pairs += 1
    return pairs, total_tiles


def _dry_run_estimate(project_root: Path, output_root: Path, patch_size: int, include_hrscd: bool) -> None:
    # Use HRSCD-generated tile sizes as practical priors for remaining datasets.
    hr_out = output_root / "hrscd" / "train"
    n1, s1 = _folder_png_stats(hr_out / "T1")
    n2, s2 = _folder_png_stats(hr_out / "T2")
    ng, sg = _folder_png_stats(hr_out / "GT")
    if n1 > 0 and n2 > 0 and ng > 0:
        avg_rgb = (s1 / n1 + s2 / n2) / 2.0
        avg_gt = sg / ng
    else:
        avg_rgb = 700 * 1024.0
        avg_gt = 200 * 1024.0
    per_tile_bytes = 2.0 * avg_rgb + avg_gt

    if include_hrscd:
        hr_pairs, hr_tiles = _estimate_total_hrscd_tiles(project_root / "datasets" / "HRSCD", patch_size)
    else:
        hr_pairs, hr_tiles = 0, 0
    fp_pairs, fp_tiles = _estimate_total_fpcd_tiles(project_root / "datasets" / "FPCD", patch_size)
    hi_pairs, hi_tiles = _estimate_total_hicna_tiles(project_root / "datasets" / "Hi-CNA_dataset", patch_size)

    hr_done_tiles = min(n1, n2, ng)
    hr_remaining_tiles = max(hr_tiles - hr_done_tiles, 0)
    est_hr_rem = hr_remaining_tiles * per_tile_bytes
    est_fp = fp_tiles * per_tile_bytes
    est_hi = hi_tiles * per_tile_bytes
    est_total_rem = est_hr_rem + est_fp + est_hi
    free = shutil.disk_usage(project_root).free

    print("========== Dry-Run Storage Estimate ==========")
    print(f"patch_size={patch_size}")
    print(f"tile_size_prior from existing HRSCD: rgb~{avg_rgb/1024:.1f}KB gt~{avg_gt/1024:.1f}KB")
    if include_hrscd:
        print(f"HRSCD pairs={hr_pairs} tiles_total={hr_tiles} tiles_done~{hr_done_tiles} tiles_remaining~{hr_remaining_tiles}")
    else:
        print("HRSCD skipped (not included in this run)")
    print(f"FPCD pairs={fp_pairs} tiles_total={fp_tiles}")
    print(f"Hi-CNA pairs={hi_pairs} tiles_total={hi_tiles}")
    print(f"estimated_remaining_hrscd_gb={est_hr_rem/1024**3:.1f}")
    print(f"estimated_fpcd_gb={est_fp/1024**3:.1f}")
    print(f"estimated_hicna_gb={est_hi/1024**3:.1f}")
    print(f"estimated_total_remaining_gb={est_total_rem/1024**3:.1f}")
    print(f"free_space_gb={free/1024**3:.1f}")
    print(f"free_minus_estimate_gb={(free-est_total_rem)/1024**3:.1f}")


def _recompress_existing_pngs(output_root: Path, workers: int, level: int) -> None:
    png_files = sorted(output_root.rglob("*.png"))
    if not png_files:
        print(f"No PNG files under {output_root}")
        return

    print(f"Recompressing {len(png_files)} PNG files at level={level} with workers={workers}")
    jobs = [(str(p), level) for p in png_files]
    out = _run_jobs(jobs, _recompress_png_job, desc="recompress/png", unit="img", workers=workers)
    before = sum(x[0] for x in out)
    after = sum(x[1] for x in out)
    print(f"recompress done: before={before/1024**3:.2f}GB after={after/1024**3:.2f}GB delta={(after-before)/1024**3:.2f}GB")


def main() -> None:
    global _TQDM_DISABLE, _PNG_COMPRESS_LEVEL

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
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size for FPCD/Hi-CNA/input_quick tiling (and HRSCD only when --include_hrscd is set).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Process workers for CLCD/JL1/FPCD/Hi-CNA conversion (plus HRSCD only when --include_hrscd is set).",
    )
    parser.add_argument(
        "--include_hrscd",
        action="store_true",
        help="Include HRSCD conversion in Stage-2 joint collection build (disabled by default).",
    )
    parser.add_argument(
        "--png_compress_level",
        type=int,
        default=1,
        help="PNG compression level [0..9]. Lower is faster and larger (recommended 0-1 for faster preprocessing).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (e.g. when redirecting logs to a file).",
    )
    parser.add_argument(
        "--dry_run_estimate",
        action="store_true",
        help="Estimate remaining storage requirement and exit without writing files.",
    )
    parser.add_argument(
        "--recompress_existing",
        action="store_true",
        help="Recompress existing PNG files under output_root using --png_compress_level and exit.",
    )
    args = parser.parse_args()
    _TQDM_DISABLE = bool(args.no_progress)
    _PNG_COMPRESS_LEVEL = max(0, min(9, int(args.png_compress_level)))

    workers = _resolve_workers(int(args.workers))
    print(f"[PERF] worker_processes={workers} png_compress_level={_PNG_COMPRESS_LEVEL}")

    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (project_root / "datasets" / "cropland_bcd_collections").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    ps = int(args.patch_size)
    if args.dry_run_estimate:
        _dry_run_estimate(project_root, output_root, ps, include_hrscd=bool(args.include_hrscd))
        return

    if args.recompress_existing:
        _recompress_existing_pngs(output_root, workers=workers, level=_PNG_COMPRESS_LEVEL)
        return

    clcd_root = project_root / "datasets" / "CLCD"
    fpcd_root = project_root / "datasets" / "FPCD"
    hi_cna_root = project_root / "datasets" / "Hi-CNA_dataset"
    jl1_root = project_root / "datasets" / "jl1_cropland_competition_2023"
    input_root = project_root / "data" / "input"

    all_entries: list[dict[str, list[str]]] = []
    entries_by_source: dict[str, dict[str, list[str]]] = {}

    _Run = Callable[[], dict[str, list[str]]]
    pipeline: list[tuple[str, _Run]] = [
        ("clcd", lambda: convert_clcd(clcd_root, output_root, workers=workers)),
        ("fpcd", lambda: convert_fpcd(fpcd_root, output_root, patch_size=ps, workers=workers)),
        ("hi_cna", lambda: convert_hi_cna(hi_cna_root, output_root, patch_size=ps, workers=workers)),
        (
            "jl1_competition",
            lambda: convert_jl1_competition(
                jl1_root, output_root, train_ratio=args.train_ratio, seed=args.seed, workers=workers
            ),
        ),
        (
            "input_quick",
            lambda: convert_input_quick(
                input_root,
                output_root,
                patch_size=ps,
                train_ratio=args.train_ratio,
                seed=args.seed,
            ),
        ),
    ]
    if args.include_hrscd:
        hrscd_root = project_root / "datasets" / "HRSCD"
        pipeline.insert(
            1,
            ("hrscd", lambda: convert_hrscd(hrscd_root, output_root, patch_size=ps, workers=workers)),
        )

    for key, run in _pbar(pipeline, desc="BCD collections", unit="src"):
        ent = run()
        entries_by_source[key] = ent
        all_entries.append(ent)

    write_stage1_jl1_manifest(output_root, entries_by_source)
    write_stage2_joint_manifest(output_root, entries_by_source)
    write_stage3_input_quick_manifests(output_root, entries_by_source)
    merge_global_lists(output_root, all_entries)
    print(f"Done. Output root: {output_root}")


if __name__ == "__main__":
    main()
