#!/usr/bin/env python3
"""
Gradio UI: load ChangeMambaSCD, run tiled inference on large T1/T2 pairs, save outputs,
then optionally run a separate evaluation step against GT (two buttons).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo root: .../src/gradio_large_image_infer.py -> parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import imageio.v2 as imageio
import numpy as np
import torch

import gradio as gr
from PIL import Image, ImageDraw

try:
    import fiona
    import rasterio
    from rasterio import features as rio_features
    from rasterio.warp import transform_geom

    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False
    fiona = None  # type: ignore
    rasterio = None  # type: ignore
    rio_features = None  # type: ignore
    transform_geom = None  # type: ignore

from ChangeMamba.changedetection.configs.config import get_config
from ChangeMamba.changedetection.datasets import imutils
from ChangeMamba.changedetection.models.ChangeMambaSCD import ChangeMambaSCD
from ChangeMamba.changedetection.utils_func.mcd_utils import SCDD_metrics_from_hist, accuracy, get_hist
from datasets.colormap import JL1_CLASSES, NUM_CLASSES, index2color


NUM_SCD_CLASSES = 37
MAX_SCENE_PREVIEW_SIDE = 1400
MAX_PATCH_CARD_SIDE = 320
MAX_PATCH_SAMPLES = 12


def _load_rgb_f32(path: str) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    img = np.asarray(imageio.imread(str(p)), dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]
    return img


def _to_chw_normalized(img_hwc: np.ndarray) -> np.ndarray:
    x = imutils.normalize_img(img_hwc)
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))


def _pad_pair_to_multiple(
    pre: np.ndarray, post: np.ndarray, multiple: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Pad bottom/right with zeros so H,W are multiples of `multiple`. Returns padded pair and (pad_h, pad_w)."""
    h, w = pre.shape[:2]
    nh = int(np.ceil(h / multiple) * multiple)
    nw = int(np.ceil(w / multiple) * multiple)
    ph, pw = nh - h, nw - w
    if ph == 0 and pw == 0:
        return pre, post, (0, 0)
    out_pre = np.zeros((nh, nw, 3), dtype=pre.dtype)
    out_post = np.zeros((nh, nw, 3), dtype=post.dtype)
    out_pre[:h, :w] = pre
    out_post[:h, :w] = post
    return out_pre, out_post, (ph, pw)


def _is_vector_gt_path(p: Path) -> bool:
    if p.suffix.lower() == ".shp":
        return True
    if p.is_dir():
        return any(p.glob("*.shp"))
    return False


def _resolve_shapefile_path(p: Path) -> Path:
    if p.suffix.lower() == ".shp":
        return p
    if p.is_dir():
        shps = sorted(p.glob("*.shp"))
        if len(shps) == 1:
            return shps[0]
        if len(shps) == 0:
            raise FileNotFoundError(f"No .shp file in directory: {p}")
        raise ValueError(f"Multiple .shp files in {p}; pass the path to one .shp file.")
    raise FileNotFoundError(f"Not a shapefile path: {p}")


def _vector_label_from_props(props: dict[str, Any], explicit_field: str | None) -> int | float:
    if explicit_field:
        if explicit_field not in props:
            raise KeyError(f"Attribute {explicit_field!r} not found on vector features.")
        v = props[explicit_field]
    else:
        skip = {
            "id",
            "objectid",
            "fid",
            "shape_area",
            "shape_len",
            "shape_length",
            "area",
            "perimeter",
            "length",
        }
        v = None
        for k, raw in props.items():
            lk = str(k).lower()
            if lk in skip or lk.endswith("_id") or lk == "id":
                continue
            if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool):
                v = int(raw)
                break
            if isinstance(raw, float) and not isinstance(raw, bool):
                if raw == int(raw):
                    v = int(raw)
                    break
                v = raw
                break
        if v is None:
            for _k, raw in props.items():
                if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
                    v = int(raw.strip())
                    break
        if v is None:
            raise ValueError(
                "Could not infer a class attribute on the shapefile. "
                "Add an integer field (0–5 for JL1 semantics) or install/configure attributes."
            )
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        v = int(v.strip())
    if isinstance(v, float) and v == int(v):
        v = int(v)
    return v


def _rasterize_vector_gt(
    gt_path: Path,
    ref_raster_path: Path,
    target_hw: tuple[int, int],
    *,
    kind: str,
    label_field: str | None,
) -> np.ndarray:
    """Burn vector polygons onto the same grid as the unpadded T1 image (via ref_raster_path)."""
    if not _HAS_RASTERIO:
        raise RuntimeError(
            "Vector ground truth requires rasterio and fiona. Install with: pip install rasterio fiona"
        )
    shp = _resolve_shapefile_path(gt_path)
    h0, w0 = target_hw
    pairs: list[tuple[dict, int | float]] = []
    vec_crs = None
    with fiona.open(str(shp)) as src:
        if src.crs is not None:
            vec_crs = rasterio.crs.CRS.from_user_input(src.crs)
        for feat in src:
            geom = feat.get("geometry")
            if geom is None:
                continue
            props = feat.get("properties") or {}
            if kind == "cd":
                val: int | float = 255
            else:
                val = _vector_label_from_props(props, label_field)
                if isinstance(val, float) and val != int(val):
                    raise ValueError(f"Semantic label must be integer-like, got {val!r} from vector.")
                val = int(val)
            pairs.append((geom, val))

    if not pairs:
        raise ValueError(f"No features with geometry in vector: {shp}")

    with rasterio.open(str(ref_raster_path)) as ref:
        ref_crs = ref.crs
        if ref_crs is None:
            raise ValueError(f"Reference raster has no CRS; cannot align vector GT: {ref_raster_path}")
        window = rasterio.windows.Window(0, 0, w0, h0)
        transform = ref.window_transform(window)

        def _iter_shapes():
            for geom, val in pairs:
                g = geom
                if vec_crs is not None and vec_crs != ref_crs:
                    g = transform_geom(vec_crs, ref_crs, g)
                yield (g, val)

        if kind == "cd":
            dtype = np.float32
            fill = 0.0
        else:
            dtype = np.int32
            fill = 0
        out = rio_features.rasterize(
            _iter_shapes(),
            out_shape=(h0, w0),
            transform=transform,
            fill=fill,
            dtype=dtype,
            all_touched=False,
        )
    return np.asarray(out)


def _load_gt_maps(
    gt_cd_path: str | None,
    gt_t1_path: str | None,
    gt_t2_path: str | None,
    target_hw: tuple[int, int],
    *,
    ref_raster_path: str | None,
    vector_label_field: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load GT_CD (0/1), GT_T1, GT_T2 class indices at target H×W (raster or vector)."""
    th, tw = target_hw
    field = (vector_label_field or "").strip() or None

    def _load_cd_raster(p: str) -> np.ndarray:
        x = np.asarray(imageio.imread(p))
        if x.ndim == 3:
            x = x[..., 0]
        x = (x > 127).astype(np.float32)
        return x

    def _load_sem_raster(p: str) -> np.ndarray:
        x = np.asarray(imageio.imread(p))
        if x.ndim == 3:
            x = x[..., 0]
        return x.astype(np.int32)

    def _load_one(path_str: str, name: str, kind: str) -> np.ndarray:
        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")
        if _is_vector_gt_path(p):
            if not ref_raster_path:
                raise ValueError(
                    f"{name} is vector data (.shp). Run inference with the T1 raster path first "
                    "so labels can be rasterized to the same grid (or export GT to GeoTIFF/PNG)."
                )
            ref_p = Path(ref_raster_path).expanduser().resolve()
            if not ref_p.is_file():
                raise FileNotFoundError(f"Reference raster for vector GT not found: {ref_p}")
            return _rasterize_vector_gt(p, ref_p, (th, tw), kind=kind, label_field=field)
        if kind == "cd":
            return _load_cd_raster(str(p))
        return _load_sem_raster(str(p))

    cd = _load_one(gt_cd_path, "GT_CD", "cd") if gt_cd_path else None
    t1 = _load_one(gt_t1_path, "GT_T1", "sem") if gt_t1_path else None
    t2 = _load_one(gt_t2_path, "GT_T2", "sem") if gt_t2_path else None

    for name, arr in (("GT_CD", cd), ("GT_T1", t1), ("GT_T2", t2)):
        if arr is None:
            continue
        if arr.shape[0] != th or arr.shape[1] != tw:
            raise ValueError(
                f"{name} size {arr.shape[:2]} does not match image size {(th, tw)}. "
                "For rasters, match T1/T2 resolution; for shapefiles, check CRS alignment with the T1 reference."
            )
    return cd, t1, t2


def _semantic_rgb(pred_cls: np.ndarray, change_mask: np.ndarray) -> np.ndarray:
    """Colorize class map; no-change (mask 0) and class 0 shown as white."""
    rgb = index2color(pred_cls).copy()
    white = np.array([255, 255, 255], dtype=np.uint8)
    rgb[pred_cls == 0] = white
    rgb[change_mask == 0] = white
    return rgb


def _downsample_to_max_side(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return arr
    step = int(np.ceil(long_side / max_side))
    return arr[::step, ::step]


def _fit_long_side(arr: np.ndarray, max_side: int, *, nearest: bool = False) -> np.ndarray:
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return arr
    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(arr)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(pil.resize((new_w, new_h), resample=resample))


def _to_display_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    elif x.shape[-1] > 3:
        x = x[..., :3]
    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)

    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return np.zeros((*x.shape[:2], 3), dtype=np.uint8)

    if float(np.max(x)) <= 1.0 and float(np.min(x)) >= 0.0:
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    out = np.zeros_like(x, dtype=np.uint8)
    for c in range(min(3, x.shape[-1])):
        chan = x[..., c]
        lo = float(np.percentile(chan, 2.0))
        hi = float(np.percentile(chan, 98.0))
        if hi <= lo:
            lo = float(np.min(chan))
            hi = float(np.max(chan))
        if hi <= lo:
            out[..., c] = np.clip(chan, 0, 255).astype(np.uint8)
        else:
            out[..., c] = np.clip((chan - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def _alpha_blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    mask3 = np.asarray(mask > 0, dtype=bool)[..., None]
    overlay_f = overlay.astype(np.float32)
    base_f = base.astype(np.float32)
    blended = (1.0 - alpha) * base_f + alpha * overlay_f
    out = np.where(mask3, blended, base_f)
    return np.clip(out, 0, 255).astype(np.uint8)


def _change_overlay(base: np.ndarray, change_mask: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    red = np.zeros_like(base, dtype=np.uint8)
    red[..., 0] = 255
    return _alpha_blend(base, red, change_mask, alpha=alpha)


def _semantic_overlay(base: np.ndarray, pred_cls: np.ndarray, change_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    sem_rgb = _semantic_rgb(pred_cls, change_mask)
    return _alpha_blend(base, sem_rgb, change_mask, alpha=alpha)


def _make_strip(images: list[np.ndarray], *, pad: int = 6, bg: int = 255) -> np.ndarray:
    valid = [img for img in images if img is not None]
    if not valid:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    max_h = max(img.shape[0] for img in valid)
    total_w = sum(img.shape[1] for img in valid) + pad * (len(valid) - 1)
    canvas = np.full((max_h, total_w, 3), bg, dtype=np.uint8)
    x0 = 0
    for img in valid:
        h, w = img.shape[:2]
        y0 = (max_h - h) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = img
        x0 += w + pad
    return canvas


def _make_grid(images: list[np.ndarray], *, cols: int = 3, pad: int = 6, bg: int = 255) -> np.ndarray:
    valid = [img for img in images if img is not None]
    if not valid:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    rows = int(np.ceil(len(valid) / cols))
    row_heights: list[int] = []
    col_widths: list[int] = []
    for r in range(rows):
        row_imgs = valid[r * cols : (r + 1) * cols]
        row_heights.append(max(img.shape[0] for img in row_imgs))
    for c in range(cols):
        col_imgs = valid[c::cols]
        if col_imgs:
            col_widths.append(max(img.shape[1] for img in col_imgs))
    total_h = sum(row_heights) + pad * (rows - 1)
    total_w = sum(col_widths) + pad * (len(col_widths) - 1)
    canvas = np.full((total_h, total_w, 3), bg, dtype=np.uint8)
    y0 = 0
    for r in range(rows):
        x0 = 0
        row_imgs = valid[r * cols : (r + 1) * cols]
        for c, img in enumerate(row_imgs):
            h, w = img.shape[:2]
            yy = y0 + (row_heights[r] - h) // 2
            xx = x0 + (col_widths[c] - w) // 2
            canvas[yy : yy + h, xx : xx + w] = img
            x0 += col_widths[c] + pad
        y0 += row_heights[r] + pad
    return canvas


def _draw_patch_box(scene_img: np.ndarray, original_hw: tuple[int, int], bounds: tuple[int, int, int, int]) -> np.ndarray:
    h0, w0 = original_hw
    y0, x0, y1, x1 = bounds
    scene_h, scene_w = scene_img.shape[:2]
    x0s = int(round(x0 * scene_w / w0))
    x1s = max(x0s + 1, int(round(x1 * scene_w / w0)))
    y0s = int(round(y0 * scene_h / h0))
    y1s = max(y0s + 1, int(round(y1 * scene_h / h0)))
    pil = Image.fromarray(scene_img)
    draw = ImageDraw.Draw(pil)
    for offset in range(3):
        draw.rectangle(
            [(x0s - offset, y0s - offset), (x1s + offset, y1s + offset)],
            outline=(255, 215, 0),
            width=1,
        )
    return np.asarray(pil)


def _empty_patch_outputs():
    return (
        gr.update(choices=[], value=None),
        None,
        None,
        None,
        [],
        "Patch explorer is empty. Run tiled inference to populate representative tiles.",
    )


def _empty_inference_outputs(status: str):
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        *_empty_patch_outputs(),
        "",
        status,
        "",
        "",
    )


@dataclass
class PatchSample:
    label: str
    row: int
    col: int
    bounds: tuple[int, int, int, int]
    change_ratio: float
    scene_view: np.ndarray
    input_strip: np.ndarray
    pred_strip: np.ndarray
    gallery_card: np.ndarray


def _build_model_and_load(
    cfg_path: str,
    pretrained_backbone: str | None,
    checkpoint_path: str | None,
    device: torch.device,
) -> ChangeMambaSCD:
    class _Args:
        def __init__(self):
            self.cfg = cfg_path
            self.opts = None

    config = get_config(_Args())
    model = ChangeMambaSCD(
        output_cd=2,
        output_clf=7,
        pretrained=pretrained_backbone,
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )
    model = model.to(device)
    model.eval()

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
    return model


@dataclass
class Session:
    model: ChangeMambaSCD | None = None
    device: torch.device | None = None
    cfg_path: str = ""
    pretrained_backbone: str | None = None
    checkpoint_path: str | None = None
    # T1 path from last inference — georeference for rasterizing vector GT (shapefile) in Evaluate
    ref_t1_path: str | None = None
    # Cropped to original image size (h0, w0), same as saved previews — for Evaluate
    pred_change_mask: np.ndarray | None = None
    pred_sem_t1: np.ndarray | None = None
    pred_sem_t2: np.ndarray | None = None
    image_hw: tuple[int, int] | None = None
    patch_grid: tuple[int, int] | None = None
    scene_t1_preview: np.ndarray | None = None
    scene_t2_preview: np.ndarray | None = None
    scene_change_overlay: np.ndarray | None = None
    scene_sem_t1_overlay: np.ndarray | None = None
    scene_sem_t2_overlay: np.ndarray | None = None
    patch_samples: dict[str, PatchSample] = field(default_factory=dict)
    patch_sample_labels: list[str] = field(default_factory=list)


_SESSION = Session()


def _reset_cached_outputs() -> None:
    _SESSION.ref_t1_path = None
    _SESSION.pred_change_mask = None
    _SESSION.pred_sem_t1 = None
    _SESSION.pred_sem_t2 = None
    _SESSION.image_hw = None
    _SESSION.patch_grid = None
    _SESSION.scene_t1_preview = None
    _SESSION.scene_t2_preview = None
    _SESSION.scene_change_overlay = None
    _SESSION.scene_sem_t1_overlay = None
    _SESSION.scene_sem_t2_overlay = None
    _SESSION.patch_samples.clear()
    _SESSION.patch_sample_labels.clear()


def _select_patch_samples(
    patch_stats: list[tuple[float, int, int, int]],
    max_samples: int,
) -> list[tuple[float, int, int, int]]:
    if not patch_stats:
        return []
    ranked = sorted(patch_stats, key=lambda item: (-item[0], item[1]))
    chosen: list[tuple[float, int, int, int]] = []
    seen: set[int] = set()

    for item in ranked:
        if item[0] <= 0:
            continue
        idx = item[1]
        if idx in seen:
            continue
        chosen.append(item)
        seen.add(idx)
        if len(chosen) >= max_samples:
            return chosen

    if len(chosen) < max_samples:
        if len(patch_stats) <= max_samples:
            fill_indices = range(len(patch_stats))
        else:
            fill_indices = np.linspace(0, len(patch_stats) - 1, num=max_samples, dtype=int).tolist()
        for pos in fill_indices:
            item = patch_stats[int(pos)]
            idx = item[1]
            if idx in seen:
                continue
            chosen.append(item)
            seen.add(idx)
            if len(chosen) >= max_samples:
                break

    return chosen


def _build_patch_visuals(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    change_mask: np.ndarray,
    t1_pred: np.ndarray,
    t2_pred: np.ndarray,
    patch_size: int,
    nh: int,
    nw: int,
    scene_change_overlay: np.ndarray,
) -> tuple[dict[str, PatchSample], list[tuple[np.ndarray, str]]]:
    h0, w0 = change_mask.shape[:2]
    patch_stats: list[tuple[float, int, int, int]] = []
    for idx, (y0, x0) in enumerate((i, j) for i in range(0, h0, patch_size) for j in range(0, w0, patch_size)):
        y1 = min(y0 + patch_size, h0)
        x1 = min(x0 + patch_size, w0)
        patch_ratio = float(np.mean(change_mask[y0:y1, x0:x1] > 0))
        row = idx // nw
        col = idx % nw
        patch_stats.append((patch_ratio, idx, row, col))

    selected = _select_patch_samples(patch_stats, MAX_PATCH_SAMPLES)
    patch_samples: dict[str, PatchSample] = {}
    gallery_items: list[tuple[np.ndarray, str]] = []

    for change_ratio, idx, row, col in selected:
        y0 = row * patch_size
        x0 = col * patch_size
        y1 = min(y0 + patch_size, h0)
        x1 = min(x0 + patch_size, w0)

        pre_crop = _to_display_rgb(pre_img[y0:y1, x0:x1])
        post_crop = _to_display_rgb(post_img[y0:y1, x0:x1])
        change_crop = change_mask[y0:y1, x0:x1]
        t1_crop = t1_pred[y0:y1, x0:x1]
        t2_crop = t2_pred[y0:y1, x0:x1]
        change_vis = _change_overlay(post_crop, change_crop)
        t1_overlay = _semantic_overlay(pre_crop, t1_crop, change_crop)
        t2_overlay = _semantic_overlay(post_crop, t2_crop, change_crop)

        pre_crop = _fit_long_side(pre_crop, MAX_PATCH_CARD_SIDE)
        post_crop = _fit_long_side(post_crop, MAX_PATCH_CARD_SIDE)
        change_vis = _fit_long_side(change_vis, MAX_PATCH_CARD_SIDE)
        t1_overlay = _fit_long_side(t1_overlay, MAX_PATCH_CARD_SIDE)
        t2_overlay = _fit_long_side(t2_overlay, MAX_PATCH_CARD_SIDE)

        input_strip = _make_strip([pre_crop, post_crop])
        pred_strip = _make_strip([change_vis, t1_overlay, t2_overlay])
        gallery_card = _make_grid([pre_crop, post_crop, change_vis, t1_overlay, t2_overlay], cols=3)
        label = f"Patch {idx + 1} - row {row + 1}/{nh}, col {col + 1}/{nw} - changed {change_ratio:.1%}"
        sample = PatchSample(
            label=label,
            row=row,
            col=col,
            bounds=(y0, x0, y1, x1),
            change_ratio=change_ratio,
            scene_view=_draw_patch_box(scene_change_overlay, (h0, w0), (y0, x0, y1, x1)),
            input_strip=input_strip,
            pred_strip=pred_strip,
            gallery_card=gallery_card,
        )
        patch_samples[label] = sample
        gallery_items.append((gallery_card, label))

    return patch_samples, gallery_items


def show_patch_details(patch_label: str):
    label = (patch_label or "").strip()
    if not label or label not in _SESSION.patch_samples:
        return None, None, None, "Select one representative patch after inference finishes."

    sample = _SESSION.patch_samples[label]
    h0, w0 = _SESSION.image_hw or (0, 0)
    nh, nw = _SESSION.patch_grid or (0, 0)
    y0, x0, y1, x1 = sample.bounds
    info = (
        f"Viewing {sample.label}\n"
        f"- pixel bounds: y={y0}:{y1}, x={x0}:{x1}\n"
        f"- patch size on disk: {y1 - y0} x {x1 - x0}\n"
        f"- scene size: {h0} x {w0}\n"
        f"- patch grid: {nh} rows x {nw} cols"
    )
    return sample.scene_view, sample.input_strip, sample.pred_strip, info


def load_model_fn(
    cfg_path: str,
    pretrained_backbone: str,
    checkpoint_path: str,
    use_cuda: bool,
) -> str:
    global _SESSION
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    if use_cuda and not torch.cuda.is_available():
        return "CUDA requested but not available; falling back would happen at inference — uncheck GPU or install CUDA PyTorch."

    cfg = (cfg_path or "").strip()
    if not cfg:
        cfg = str(
            Path(__file__).resolve().parent
            / "ChangeMamba"
            / "changedetection"
            / "configs"
            / "vssm1"
            / "vssm_base_224.yaml"
        )
    pb = pretrained_backbone.strip() or None
    ck = checkpoint_path.strip() or None

    try:
        model = _build_model_and_load(cfg, pb, ck, device)
    except Exception as e:
        return f"Failed to load model: {e}"

    _SESSION.model = model
    _SESSION.device = device
    _SESSION.cfg_path = cfg
    _SESSION.pretrained_backbone = pb
    _SESSION.checkpoint_path = ck
    _reset_cached_outputs()
    return f"Model ready on {device} (cfg={Path(cfg).name}, checkpoint={'yes' if ck else 'no'})."


def run_tiled_inference(
    t1_path: str,
    t2_path: str,
    patch_size: int,
    micro_batch: int,
    out_dir: str,
    progress: gr.Progress,
):
    if _SESSION.model is None or _SESSION.device is None:
        yield _empty_inference_outputs("Load the model first (Model tab).")
        return

    t1_path = (t1_path or "").strip()
    t2_path = (t2_path or "").strip()
    if not t1_path or not t2_path:
        yield _empty_inference_outputs("Provide absolute paths to T1 and T2 images.")
        return

    patch_size = int(patch_size)
    if patch_size <= 0:
        yield _empty_inference_outputs("patch_size must be positive.")
        return
    micro_batch = max(1, int(micro_batch))

    device = _SESSION.device
    model = _SESSION.model
    _reset_cached_outputs()

    try:
        pre_img = _load_rgb_f32(t1_path)
        post_img = _load_rgb_f32(t2_path)
    except Exception as e:
        yield _empty_inference_outputs(f"Failed to read images: {e}")
        return

    if pre_img.shape[:2] != post_img.shape[:2]:
        yield _empty_inference_outputs(f"T1 shape {pre_img.shape[:2]} != T2 shape {post_img.shape[:2]}.")
        return

    pre_pad, post_pad, _ = _pad_pair_to_multiple(pre_img, post_img, patch_size)
    H, W = pre_pad.shape[:2]
    nh, nw = H // patch_size, W // patch_size
    n_patches = nh * nw

    coords = [(i, j) for i in range(0, H, patch_size) for j in range(0, W, patch_size)]
    change_full = np.zeros((H, W), dtype=np.int32)
    t1_full = np.zeros((H, W), dtype=np.int32)
    t2_full = np.zeros((H, W), dtype=np.int32)

    progress(0.0, desc="Running tiled inference…")
    with torch.no_grad():
        for start in range(0, len(coords), micro_batch):
            batch_coords = coords[start : start + micro_batch]
            tensors_pre = []
            tensors_post = []
            for (y0, x0) in batch_coords:
                crop_pre = pre_pad[y0 : y0 + patch_size, x0 : x0 + patch_size]
                crop_post = post_pad[y0 : y0 + patch_size, x0 : x0 + patch_size]
                tensors_pre.append(_to_chw_normalized(crop_pre))
                tensors_post.append(_to_chw_normalized(crop_post))
            b_pre = torch.from_numpy(np.stack(tensors_pre, axis=0)).to(device)
            b_post = torch.from_numpy(np.stack(tensors_post, axis=0)).to(device)
            out_cd, out_t1, out_t2 = model(b_pre, b_post)
            change_mask = torch.argmax(out_cd, dim=1).cpu().numpy()
            pred_t1 = torch.argmax(out_t1, dim=1).cpu().numpy()
            pred_t2 = torch.argmax(out_t2, dim=1).cpu().numpy()

            for k, (y0, x0) in enumerate(batch_coords):
                change_full[y0 : y0 + patch_size, x0 : x0 + patch_size] = change_mask[k]
                t1_full[y0 : y0 + patch_size, x0 : x0 + patch_size] = pred_t1[k] * change_mask[k]
                t2_full[y0 : y0 + patch_size, x0 : x0 + patch_size] = pred_t2[k] * change_mask[k]

            frac = min(1.0, (start + len(batch_coords)) / max(1, n_patches))
            progress(frac, desc=f"Patches {min(start + len(batch_coords), n_patches)}/{n_patches}")

    h0, w0 = pre_img.shape[0], pre_img.shape[1]
    change_vis = change_full[:h0, :w0].astype(np.uint8) * 255
    cm = change_full[:h0, :w0]
    t1_crop = t1_full[:h0, :w0]
    t2_crop = t2_full[:h0, :w0]

    rgb_t1 = _semantic_rgb(t1_crop, cm)
    rgb_t2 = _semantic_rgb(t2_crop, cm)

    pre_scene_small = _to_display_rgb(_downsample_to_max_side(pre_img, MAX_SCENE_PREVIEW_SIDE))
    post_scene_small = _to_display_rgb(_downsample_to_max_side(post_img, MAX_SCENE_PREVIEW_SIDE))
    cm_scene = _downsample_to_max_side(cm, MAX_SCENE_PREVIEW_SIDE)
    t1_scene = _downsample_to_max_side(t1_crop, MAX_SCENE_PREVIEW_SIDE)
    t2_scene = _downsample_to_max_side(t2_crop, MAX_SCENE_PREVIEW_SIDE)
    scene_change_overlay = _change_overlay(post_scene_small, cm_scene)
    scene_sem_t1_overlay = _semantic_overlay(pre_scene_small, t1_scene, cm_scene)
    scene_sem_t2_overlay = _semantic_overlay(post_scene_small, t2_scene, cm_scene)
    patch_samples, patch_gallery = _build_patch_visuals(
        pre_img,
        post_img,
        cm,
        t1_crop,
        t2_crop,
        patch_size,
        nh,
        nw,
        scene_change_overlay,
    )
    patch_labels = list(patch_samples.keys())
    if patch_labels:
        patch_dropdown = gr.update(choices=patch_labels, value=patch_labels[0])
        patch_scene, patch_inputs, patch_preds, patch_info = show_patch_details(patch_labels[0])
    else:
        patch_dropdown, patch_scene, patch_inputs, patch_preds, patch_gallery, patch_info = _empty_patch_outputs()

    out_root = Path((out_dir or "").strip() or str(_PROJECT_ROOT / "outputs" / "gradio_scd"))
    out_root.mkdir(parents=True, exist_ok=True)
    stem = Path(t1_path).stem
    p_change = out_root / f"{stem}_pred_GT_CD.png"
    p_t1 = out_root / f"{stem}_pred_semantic_T1.png"
    p_t2 = out_root / f"{stem}_pred_semantic_T2.png"
    imageio.imwrite(str(p_change), change_vis)
    imageio.imwrite(str(p_t1), rgb_t1)
    imageio.imwrite(str(p_t2), rgb_t2)

    _SESSION.ref_t1_path = t1_path
    _SESSION.pred_change_mask = cm.copy()
    _SESSION.pred_sem_t1 = t1_crop.copy()
    _SESSION.pred_sem_t2 = t2_crop.copy()
    _SESSION.image_hw = (h0, w0)
    _SESSION.patch_grid = (nh, nw)
    _SESSION.scene_t1_preview = pre_scene_small
    _SESSION.scene_t2_preview = post_scene_small
    _SESSION.scene_change_overlay = scene_change_overlay
    _SESSION.scene_sem_t1_overlay = scene_sem_t1_overlay
    _SESSION.scene_sem_t2_overlay = scene_sem_t2_overlay
    _SESSION.patch_samples = patch_samples
    _SESSION.patch_sample_labels = patch_labels

    status_tail = (
        f"Input scene: {h0}x{w0}px, patch grid: {nh} x {nw} ({n_patches} tiles), patch size: {patch_size}px, micro-batch: {micro_batch}\n"
        "UI view: downsampled scene previews + representative patch explorer for large-image inspection.\n\n"
        f"Saved:\n- {p_change}\n- {p_t1}\n- {p_t2}\n\n"
        "Predictions are cached for **Evaluate** (same resolution as T1/T2, unpadded)."
    )

    yield (
        pre_scene_small,
        post_scene_small,
        scene_change_overlay,
        scene_sem_t1_overlay,
        scene_sem_t2_overlay,
        np.stack([change_vis] * 3, axis=-1),
        rgb_t1,
        rgb_t2,
        patch_dropdown,
        patch_scene,
        patch_inputs,
        patch_preds,
        patch_gallery,
        patch_info,
        str(out_root),
        status_tail,
        "",
        "",
    )


def run_evaluation(
    gt_cd_path: str,
    gt_t1_path: str,
    gt_t2_path: str,
    vector_label_field: str,
):
    if _SESSION.pred_change_mask is None:
        return "", "Run **Run tiled inference** first so predictions are available."

    cm = _SESSION.pred_change_mask
    t1_crop = _SESSION.pred_sem_t1
    t2_crop = _SESSION.pred_sem_t2
    assert t1_crop is not None and t2_crop is not None

    gt_cd_s = (gt_cd_path or "").strip()
    gt_t1_s = (gt_t1_path or "").strip()
    gt_t2_s = (gt_t2_path or "").strip()
    if not (gt_cd_s and gt_t1_s and gt_t2_s):
        return "", "Provide all three paths: GT_CD, GT_T1, and GT_T2."

    h0, w0 = cm.shape[:2]
    ref_t1 = _SESSION.ref_t1_path
    vfield = (vector_label_field or "").strip() or None
    try:
        gt_cd, gt_t1, gt_t2 = _load_gt_maps(
            gt_cd_s,
            gt_t1_s,
            gt_t2_s,
            (h0, w0),
            ref_raster_path=ref_t1,
            vector_label_field=vfield,
        )
    except Exception as e:
        return "", str(e)

    if gt_cd is None or gt_t1 is None or gt_t2 is None:
        return "", "Failed to load ground truth."

    labels_cd_np = (gt_cd > 0.5).astype(np.int32)
    labels_A = gt_t1
    labels_B = gt_t2
    preds_scd = (t1_crop - 1) * 6 + t2_crop
    preds_scd[cm == 0] = 0
    labels_scd = (labels_A - 1) * 6 + labels_B
    labels_scd[labels_cd_np == 0] = 0
    hist = np.zeros((NUM_SCD_CLASSES, NUM_SCD_CLASSES), dtype=np.float64)
    oa, _ = accuracy(preds_scd, labels_scd)
    hist += get_hist(preds_scd, labels_scd, NUM_SCD_CLASSES)
    kappa_n0, Fscd, IoU_mean, Sek = SCDD_metrics_from_hist(hist)
    metrics_md = (
        "| Metric | Value |\n|:---|---:|\n"
        f"| OA | {oa:.4f} |\n"
        f"| Kappa (no n00) | {kappa_n0:.4f} |\n"
        f"| Fscd | {Fscd:.4f} |\n"
        f"| mIoU (binary change) | {IoU_mean:.4f} |\n"
        f"| SeK | {Sek:.4f} |\n"
    )
    status = (
        f"Evaluated on {h0}×{w0} (SCD 37-class encoding, same as training validation)."
    )
    return metrics_md, status


def build_app():
    default_cfg = str(
        Path(__file__).resolve().parent
        / "ChangeMamba"
        / "changedetection"
        / "configs"
        / "vssm1"
        / "vssm_base_224.yaml"
    )

    with gr.Blocks(title="ChangeMamba SCD — large-image tiled inference") as demo:
        gr.Markdown(
            "## Large-image semantic change detection\n"
            "1. **Load model** → 2. **Run tiled inference** (default 256×256 patches) → saves maps and caches preds → "
            "3. **Evaluate vs GT** (optional, second button).\n\n"
            "The viewer is optimized for very large rasters: it shows **downsampled full-scene context** for T1/T2, "
            "**prediction overlays** that are easier to read than raw stitched masks, and a **representative patch explorer** "
            "that surfaces the most changed tiles so users can inspect local evidence without opening the saved files manually.\n\n"
            f"Class legend uses **{NUM_CLASSES}** JL1 semantic indices (0–5). "
            "Metrics use the **37-class SCD** encoding from training (`train_MambaSCD` validation). "
            "GT rasters must match **T1/T2 resolution** (unpadded image size). "
            "Vector GT (**`.shp`** or a folder containing one shapefile) is rasterized with **rasterio** "
            "onto the **T1 image grid** from the last inference (same CRS as the T1 GeoTIFF/raster). "
            "Install **`pip install rasterio fiona`** if you use shapefiles. "
            "Semantic shapefiles need an integer class field (JL1 indices 0–5); set **Vector label field** "
            "if the attribute name is not detected automatically."
        )
        with gr.Tabs():
            with gr.Tab("Model"):
                cfg_in = gr.Textbox(label="Config YAML", value=default_cfg)
                pretrain_in = gr.Textbox(label="Backbone pretrained checkpoint (optional)", placeholder="path to ImageNet backbone .pth")
                ckpt_in = gr.Textbox(label="Trained SCD checkpoint (optional)", placeholder="best_model.pth or step checkpoint")
                cuda_chk = gr.Checkbox(label="Use CUDA", value=True)
                load_btn = gr.Button("Load model", variant="primary")
                load_status = gr.Textbox(label="Status", lines=3)
                load_btn.click(load_model_fn, [cfg_in, pretrain_in, ckpt_in, cuda_chk], load_status)

            with gr.Tab("Inference"):
                t1 = gr.Textbox(label="T1 image path (before)", placeholder="/data/.../T1/large.tif")
                t2 = gr.Textbox(label="T2 image path (after)", placeholder="/data/.../T2/large.tif")
                psz = gr.Number(label="Patch size (px)", value=256, precision=0)
                mb = gr.Number(label="Micro-batch (patches per forward)", value=4, precision=0)
                odir = gr.Textbox(label="Output directory", placeholder="default: <repo>/outputs/gradio_scd")
                run_btn = gr.Button("Run tiled inference", variant="primary")
                class_legend = ", ".join(f"{idx}: {name}" for idx, name in enumerate(JL1_CLASSES))
                gr.Markdown(f"**JL1 legend** — {class_legend}")
                with gr.Accordion("Large-scene viewer", open=True):
                    with gr.Row():
                        scene_t1_prev = gr.Image(label="T1 overview (downsampled)", type="numpy")
                        scene_t2_prev = gr.Image(label="T2 overview (downsampled)", type="numpy")
                    with gr.Row():
                        scene_change_prev = gr.Image(label="Change overlay on T2", type="numpy")
                        scene_sem1_prev = gr.Image(label="Semantic overlay on T1", type="numpy")
                        scene_sem2_prev = gr.Image(label="Semantic overlay on T2", type="numpy")
                with gr.Accordion("Representative patch explorer", open=True):
                    patch_choice = gr.Dropdown(label="Representative patch", choices=[], value=None, interactive=True)
                    patch_info = gr.Textbox(label="Patch details", lines=5)
                    patch_scene = gr.Image(label="Patch location in scene", type="numpy")
                    patch_inputs = gr.Image(label="Patch inputs (T1 | T2)", type="numpy")
                    patch_preds = gr.Image(label="Patch predictions (change | semantic T1 | semantic T2)", type="numpy")
                    patch_gallery = gr.Gallery(label="Representative patch cards", columns=3, rows=2, object_fit="contain", height="auto")
                with gr.Accordion("Saved stitched outputs", open=False):
                    change_prev = gr.Image(label="Predicted change map (stitched)", type="numpy")
                    sem1_prev = gr.Image(label="Predicted semantic T1 (colored)", type="numpy")
                    sem2_prev = gr.Image(label="Predicted semantic T2 (colored)", type="numpy")
                out_path = gr.Textbox(label="Output folder used")
                run_status = gr.Textbox(label="Inference log", lines=6)
                gr.Markdown("### Evaluation (after inference)")
                gt_cd = gr.Textbox(
                    label="GT_CD path",
                    placeholder="…/GT_CD/large.png or …/change.shp (polygons → binary mask)",
                )
                gt_t1 = gr.Textbox(
                    label="GT_T1 path",
                    placeholder="…/GT_T1/large.tif/.png or …/sem_t1.shp",
                )
                gt_t2 = gr.Textbox(
                    label="GT_T2 path",
                    placeholder="…/GT_T2/large.tif/.png or …/sem_t2.shp",
                )
                vec_lbl = gr.Textbox(
                    label="Vector label field (optional)",
                    placeholder="e.g. class_id — for .shp semantic layers; leave empty to auto-pick",
                    lines=1,
                )
                eval_btn = gr.Button("Evaluate vs ground truth", variant="primary")
                metrics = gr.Markdown()
                eval_status = gr.Textbox(label="Evaluation log", lines=3)
                run_btn.click(
                    run_tiled_inference,
                    [t1, t2, psz, mb, odir],
                    [
                        scene_t1_prev,
                        scene_t2_prev,
                        scene_change_prev,
                        scene_sem1_prev,
                        scene_sem2_prev,
                        change_prev,
                        sem1_prev,
                        sem2_prev,
                        patch_choice,
                        patch_scene,
                        patch_inputs,
                        patch_preds,
                        patch_gallery,
                        patch_info,
                        out_path,
                        run_status,
                        metrics,
                        eval_status,
                    ],
                )
                patch_choice.change(show_patch_details, patch_choice, [patch_scene, patch_inputs, patch_preds, patch_info])
                eval_btn.click(run_evaluation, [gt_cd, gt_t1, gt_t2, vec_lbl], [metrics, eval_status])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_app()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
