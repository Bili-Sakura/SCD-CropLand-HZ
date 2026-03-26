#!/usr/bin/env python3
"""
Gradio UI: load ChangeMambaSCD or ChangeMambaBCD, run tiled inference on large T1/T2 pairs,
save outputs, then optionally evaluate against GT (SCD: CD + semantics; BCD: binary CD only).
"""

from __future__ import annotations

import argparse
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Repo root: .../src/gradio_large_image_infer.py -> parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import imageio.v2 as imageio
import numpy as np
import torch

import gradio as gr
from PIL import Image, ImageDraw

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

try:
    import fiona
    import rasterio
    from rasterio.enums import Resampling
    from rasterio import features as rio_features
    from rasterio.warp import reproject, transform_geom

    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False
    fiona = None  # type: ignore
    rasterio = None  # type: ignore
    Resampling = None  # type: ignore
    rio_features = None  # type: ignore
    reproject = None  # type: ignore
    transform_geom = None  # type: ignore

from ChangeMamba.changedetection.configs.config import get_config
from ChangeMamba.changedetection.datasets import imutils
from ChangeMamba.changedetection.models.ChangeMambaBCD import ChangeMambaBCD
from ChangeMamba.changedetection.models.ChangeMambaSCD import ChangeMambaSCD
from ChangeMamba.changedetection.utils_func.metrics import Evaluator
from ChangeMamba.changedetection.utils_func.mcd_utils import SCDD_metrics_from_hist, accuracy, get_hist
from datasets.colormap import JL1_CLASSES, NUM_CLASSES, index2color


NUM_SCD_CLASSES = 37
# Evaluation-oriented vector export: drop polygonized fragments smaller than this (real-world m²).
SHAPEFILE_EXPORT_MIN_AREA_M2 = 200.0
MAX_SCENE_PREVIEW_SIDE = 1400
MAX_PATCH_CARD_SIDE = 320
MAX_PATCH_SAMPLES = 12
# Same convention as scripts/infer_val_jl1_visualize.py (JL1 ChangeMambaSCD base weights).
_DEFAULT_SCD_CHECKPOINT = str(
    _PROJECT_ROOT / "models" / "BiliSakura" / "JL1-ChangeMambaSCD" / "Base" / "best_model.pth"
)

TaskMode = Literal["scd", "bcd"]

_DATA_ROOT = _PROJECT_ROOT / "data"
_DATA_SELECT_EXTS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".shp"})


def _scan_data_file_choices() -> list[str]:
    """Paths relative to repo root under data/, suitable for dropdown selection."""
    root = _DATA_ROOT
    if not root.is_dir():
        return []
    found: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _DATA_SELECT_EXTS:
            found.append(p)
    return sorted(str(x.relative_to(_PROJECT_ROOT)) for x in found)


def _resolve_project_relative(rel: str | None) -> str:
    s = (rel or "").strip()
    if not s:
        return ""
    p = (_PROJECT_ROOT / s).resolve()
    if not p.is_file():
        return ""
    return str(p)


def _pick_default_t1_t2(choices: list[str]) -> tuple[str | None, str | None]:
    def in_part(path_str: str, part: str) -> bool:
        return part in Path(path_str).parts

    t1s = [c for c in choices if in_part(c, "T1")]
    t2s = [c for c in choices if in_part(c, "T2")]
    return (t1s[0] if t1s else None, t2s[0] if t2s else None)


def _find_free_port(start: int, attempts: int = 32) -> int:
    """First bindable TCP port in [start, start + attempts). Raises if none free."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise OSError(
        f"No free port in {start}–{start + attempts - 1}. "
        "Set GRADIO_SERVER_PORT or pass --port."
    )


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


def _resample_rgb_to_ref_grid(
    ref_raster_path: str,
    src_raster_path: str,
    *,
    ref_label: str = "reference",
    src_label: str = "source",
) -> tuple[np.ndarray, str]:
    """
    Resample source raster onto reference raster grid (H/W + transform + CRS).
    Returns HxWx3 float32 and a short alignment note.
    """
    if not _HAS_RASTERIO:
        raise RuntimeError("rasterio is required for geospatial resampling.")

    with rasterio.open(ref_raster_path) as ref_ds, rasterio.open(src_raster_path) as src_ds:  # type: ignore[attr-defined]
        if ref_ds.crs is None or src_ds.crs is None:
            raise ValueError(
                "Both rasters must have CRS metadata for auto-alignment. "
                "Provide pre-aligned rasters or add CRS to the files."
            )

        dst_h, dst_w = ref_ds.height, ref_ds.width
        dst = np.zeros((3, dst_h, dst_w), dtype=np.float32)
        if src_ds.count >= 3:
            src_indexes = [1, 2, 3]
        elif src_ds.count == 2:
            src_indexes = [1, 2, 2]
        else:
            src_indexes = [1, 1, 1]

        for out_ch, src_band_idx in enumerate(src_indexes):
            reproject(  # type: ignore[misc]
                source=rasterio.band(src_ds, src_band_idx),  # type: ignore[attr-defined]
                destination=dst[out_ch],
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                dst_transform=ref_ds.transform,
                dst_crs=ref_ds.crs,
                resampling=Resampling.bilinear,  # type: ignore[union-attr]
                dst_nodata=0.0,
            )

        note = (
            f"Resampled **{src_label}** onto **{ref_label}** grid (rasterio reproject): "
            f"{src_ds.height}×{src_ds.width} → {dst_h}×{dst_w} px."
        )
        return np.transpose(dst, (1, 2, 0)), note


def _raster_native_grid_note(path: str, label: str) -> str:
    """Best-effort human-readable raster grid/pixel-size note for logs."""
    if not _HAS_RASTERIO:
        return ""
    try:
        with rasterio.open(path) as ds:  # type: ignore[attr-defined]
            x_res = float(abs(ds.transform.a))
            y_res = float(abs(ds.transform.e))
            unit = "units"
            if ds.crs is not None:
                if bool(getattr(ds.crs, "is_geographic", False)):
                    unit = "deg"
                else:
                    linear_units = str(getattr(ds.crs, "linear_units", "") or "").lower()
                    if "meter" in linear_units or "metre" in linear_units or linear_units == "m":
                        unit = "m"
                    elif linear_units:
                        unit = linear_units
                    else:
                        unit = "map-units"
            return (
                f"{label} native grid: {ds.height}x{ds.width}px, "
                f"pixel size: {x_res:.6g} x {y_res:.6g} {unit}/px."
            )
    except Exception:
        return ""


def _to_chw_normalized(img_hwc: np.ndarray) -> np.ndarray:
    x = imutils.normalize_img(img_hwc)
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))


def _stack_to_torch_batch(arrays: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    Convert list of CHW numpy arrays to a float32 tensor on `device`.
    Uses robust fallbacks for environments where torch.from_numpy fails with NumPy ABI mismatches.
    """
    batch_np = np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.float32)
    try:
        return torch.from_numpy(batch_np).to(device)
    except Exception:
        try:
            return torch.tensor(batch_np, dtype=torch.float32, device=device)
        except Exception:
            # Last-resort path avoids NumPy C-API bridge entirely.
            return torch.tensor(batch_np.tolist(), dtype=torch.float32, device=device)


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
                    f"{name} is vector data (.shp). Run inference first so labels can be rasterized "
                    "to the same grid as predictions (or export GT to GeoTIFF/PNG at that resolution)."
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


def _cleanup_shapefile_sidecars(shp_path: Path) -> None:
    """Remove existing shapefile sidecars to avoid stale artifacts on rewrite."""
    stem = shp_path.with_suffix("")
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        p = stem.with_suffix(ext)
        if p.exists():
            p.unlink()


def _ensure_closed_xy_ring(ring: list) -> list:
    if not ring:
        return ring
    if ring[0] != ring[-1]:
        return ring + [ring[0]]
    return ring


def _signed_planar_ring_area(ring: list) -> float:
    ring = _ensure_closed_xy_ring(ring)
    if len(ring) < 4:
        return 0.0
    s = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = float(ring[i][0]), float(ring[i][1])
        x1, y1 = float(ring[i + 1][0]), float(ring[i + 1][1])
        s += x0 * y1 - x1 * y0
    return s / 2.0


def _planar_polygon_area_sq_units(geom: dict) -> float:
    t = geom.get("type")
    coords = geom.get("coordinates")
    if t == "Polygon" and coords:
        ext = coords[0]
        holes = coords[1:]
        a = abs(_signed_planar_ring_area(ext))
        a -= sum(abs(_signed_planar_ring_area(h)) for h in holes)
        return max(a, 0.0)
    if t == "MultiPolygon" and coords:
        total = 0.0
        for poly in coords:
            if not poly:
                continue
            ext = poly[0]
            holes = poly[1:]
            a = abs(_signed_planar_ring_area(ext))
            a -= sum(abs(_signed_planar_ring_area(h)) for h in holes)
            total += max(a, 0.0)
        return total
    return 0.0


def _geod_polygon_area_m2(geom: dict, geod: Any) -> float:
    t = geom.get("type")
    coords = geom.get("coordinates")

    def ring_m2(ring: list) -> float:
        ring = _ensure_closed_xy_ring(ring)
        if len(ring) < 4:
            return 0.0
        lons = [float(p[0]) for p in ring]
        lats = [float(p[1]) for p in ring]
        if lons[0] != lons[-1]:
            lons.append(lons[0])
            lats.append(lats[0])
        a, _ = geod.polygon_area_perimeter(lons, lats)
        return abs(float(a))

    if t == "Polygon" and coords:
        ext = coords[0]
        holes = coords[1:]
        a = ring_m2(ext) - sum(ring_m2(h) for h in holes)
        return max(a, 0.0)
    if t == "MultiPolygon" and coords:
        total = 0.0
        for poly in coords:
            if not poly:
                continue
            ext = poly[0]
            holes = poly[1:]
            total += ring_m2(ext) - sum(ring_m2(h) for h in holes)
        return max(total, 0.0)
    return 0.0


def _geojson_geom_area_square_meters(geom: dict, crs: Any) -> float | None:
    """
    Area of a GeoJSON-like geometry dict in square metres, or None if CRS/geometry is unusable.
    Uses geodesic area for geographic CRS and planar × axis unit conversion for projected CRS.
    """
    if crs is None:
        return None
    try:
        from pyproj import CRS as PyCRS
        from pyproj import Geod
    except ImportError:
        return None
    try:
        pc = PyCRS.from_user_input(crs)
    except Exception:
        return None
    if pc.is_geographic:
        geod = Geod(ellps="WGS84")
        return _geod_polygon_area_m2(geom, geod)
    if pc.is_projected:
        sq_m_per_sq_unit: float | None = None
        try:
            axis = pc.axis_info
            if axis and axis[0].unit is not None:
                u = axis[0].unit
                cf = getattr(u, "conversion_factor", None)
                if cf is not None:
                    cf = float(cf)
                    sq_m_per_sq_unit = cf * cf
                else:
                    uname = (getattr(u, "name", "") or "").lower()
                    if "metre" in uname or "meter" in uname:
                        sq_m_per_sq_unit = 1.0
        except Exception:
            sq_m_per_sq_unit = None
        if sq_m_per_sq_unit is None:
            return None
        return _planar_polygon_area_sq_units(geom) * sq_m_per_sq_unit
    return None


def _export_pred_unified_shapefile(
    change_mask: np.ndarray,
    pred_t1: np.ndarray,
    pred_t2: np.ndarray,
    *,
    ref_raster_path: str,
    out_dir: str,
    stem: str,
    min_area_m2: float = SHAPEFILE_EXPORT_MIN_AREA_M2,
) -> tuple[Path, int, int]:
    """
    Export one unified shapefile from pixel predictions.
    Each polygon feature has: change=1, scd_cls, t1_cls, t2_cls.
    Polygons smaller than min_area_m2 (square metres) are omitted when area can be computed;
    set min_area_m2 <= 0 to disable.
    Returns (shapefile_path, feature_count_kept, feature_count_dropped_small).
    """
    if not _HAS_RASTERIO or fiona is None or rio_features is None:
        raise RuntimeError("Shapefile export requires rasterio and fiona.")

    ref = Path(ref_raster_path).expanduser().resolve()
    if not ref.is_file():
        raise FileNotFoundError(f"Reference raster not found for vector export: {ref}")

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    shp_path = out_root / f"{stem}_pred_unified.shp"
    _cleanup_shapefile_sidecars(shp_path)

    cm = (change_mask > 0).astype(np.uint8)
    if cm.shape != pred_t1.shape or cm.shape != pred_t2.shape:
        raise ValueError(
            f"Prediction map size mismatch for vector export: "
            f"CM={cm.shape}, T1={pred_t1.shape}, T2={pred_t2.shape}"
        )

    scd = ((pred_t1.astype(np.int32) - 1) * 6 + pred_t2.astype(np.int32)).astype(np.int32)
    scd[cm == 0] = 0
    valid_mask = scd > 0

    with rasterio.open(str(ref)) as ref_ds:  # type: ignore[attr-defined]
        h0, w0 = scd.shape
        if (ref_ds.height, ref_ds.width) != (h0, w0):
            raise ValueError(
                "Reference raster grid does not match prediction size for vector export: "
                f"ref={(ref_ds.height, ref_ds.width)}, pred={(h0, w0)}"
            )
        transform = ref_ds.transform
        crs_wkt = ref_ds.crs.to_wkt() if ref_ds.crs is not None else None
        ref_crs = ref_ds.crs
        apply_min_area = min_area_m2 > 0.0

        schema = {
            "geometry": "Polygon",
            "properties": {
                "change": "int",
                "scd_cls": "int",
                "t1_cls": "int",
                "t2_cls": "int",
            },
        }
        with fiona.open(  # type: ignore[misc]
            str(shp_path),
            mode="w",
            driver="ESRI Shapefile",
            schema=schema,
            crs_wkt=crs_wkt,
            encoding="UTF-8",
        ) as dst:
            kept = 0
            dropped_small = 0
            if np.any(valid_mask):
                for geom, value in rio_features.shapes(scd, mask=valid_mask, transform=transform):
                    scd_cls = int(value)
                    if scd_cls <= 0:
                        continue
                    if apply_min_area:
                        area_m2 = _geojson_geom_area_square_meters(geom, ref_crs)
                        if area_m2 is not None and area_m2 < min_area_m2:
                            dropped_small += 1
                            continue
                    t1_cls = (scd_cls - 1) // 6 + 1
                    t2_cls = (scd_cls - 1) % 6 + 1
                    dst.write(
                        {
                            "geometry": geom,
                            "properties": {
                                "change": 1,
                                "scd_cls": scd_cls,
                                "t1_cls": t1_cls,
                                "t2_cls": t2_cls,
                            },
                        }
                    )
                    kept += 1

    return shp_path, kept, dropped_small


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


def _patch_pred_label() -> str:
    return (
        "Patch predictions (change | semantic T1 | semantic T2)"
        if _SESSION.task_mode == "scd"
        else "Patch prediction (binary change on T2)"
    )


def _empty_patch_outputs():
    return (
        gr.update(choices=[], value=None),
        None,
        None,
        None,
        gr.update(value=None, label=_patch_pred_label()),
        [],
        "Patch explorer is empty. Run tiled inference to populate representative tiles.",
    )


def _empty_inference_outputs(status: str):
    sem_vis = _SESSION.task_mode == "scd"
    return (
        None,
        None,
        None,
        gr.update(value=None, visible=sem_vis),
        gr.update(value=None, visible=sem_vis),
        None,
        gr.update(value=None, visible=sem_vis),
        gr.update(value=None, visible=sem_vis),
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


def _build_bcd_model_and_load(
    cfg_path: str,
    pretrained_backbone: str | None,
    checkpoint_path: str | None,
    device: torch.device,
) -> ChangeMambaBCD:
    class _Args:
        def __init__(self):
            self.cfg = cfg_path
            self.opts = None

    config = get_config(_Args())
    model = ChangeMambaBCD(
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
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in state.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


@dataclass
class Session:
    model: ChangeMambaSCD | ChangeMambaBCD | None = None
    task_mode: TaskMode = "scd"
    device: torch.device | None = None
    cfg_path: str = ""
    pretrained_backbone: str | None = None
    checkpoint_path: str | None = None
    # Georeference raster whose grid matches predictions (T1 file if T2→T1 alignment, else T2)
    ref_raster_path: str | None = None
    # Cropped to original image size (h0, w0), same as saved previews — for Evaluate
    pred_change_mask: np.ndarray | None = None
    pred_sem_t1: np.ndarray | None = None  # None in BCD mode (no semantic heads)
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
    last_out_dir: str | None = None
    last_stem: str | None = None


_SESSION = Session()


def _reset_cached_outputs() -> None:
    _SESSION.ref_raster_path = None
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
    _SESSION.last_out_dir = None
    _SESSION.last_stem = None


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


def _build_patch_visuals_bcd(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    change_mask: np.ndarray,
    patch_size: int,
    nh: int,
    nw: int,
    scene_change_overlay: np.ndarray,
) -> tuple[dict[str, PatchSample], list[tuple[np.ndarray, str]]]:
    """Patch cards for binary change only (no semantic T1/T2 panels)."""
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
        change_vis = _change_overlay(post_crop, change_crop)

        pre_crop = _fit_long_side(pre_crop, MAX_PATCH_CARD_SIDE)
        post_crop = _fit_long_side(post_crop, MAX_PATCH_CARD_SIDE)
        change_vis = _fit_long_side(change_vis, MAX_PATCH_CARD_SIDE)

        input_strip = _make_strip([pre_crop, post_crop])
        pred_strip = change_vis
        gallery_card = _make_grid([pre_crop, post_crop, change_vis], cols=3)
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
        return None, None, gr.update(value=None, label=_patch_pred_label()), "Select one representative patch after inference finishes."

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
    return sample.scene_view, sample.input_strip, gr.update(value=sample.pred_strip, label=_patch_pred_label()), info


def load_model_fn(
    task_mode: str,
    cfg_path: str,
    pretrained_backbone: str,
    checkpoint_path: str,
    use_cuda: bool,
) -> str:
    global _SESSION
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    if use_cuda and not torch.cuda.is_available():
        return "CUDA requested but not available; falling back would happen at inference — uncheck GPU or install CUDA PyTorch."

    mode: TaskMode = "bcd" if (task_mode or "").strip().lower() == "bcd" else "scd"

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
        if mode == "bcd":
            model = _build_bcd_model_and_load(cfg, pb, ck, device)
        else:
            model = _build_model_and_load(cfg, pb, ck, device)
    except Exception as e:
        return f"Failed to load model: {e}"

    _SESSION.model = model
    _SESSION.task_mode = mode
    _SESSION.device = device
    _SESSION.cfg_path = cfg
    _SESSION.pretrained_backbone = pb
    _SESSION.checkpoint_path = ck
    _reset_cached_outputs()
    kind = "ChangeMambaBCD" if mode == "bcd" else "ChangeMambaSCD"
    return (
        f"Model ready on {device}: **{kind}** (cfg={Path(cfg).name}, checkpoint={'yes' if ck else 'no'}). "
        f"Task mode: **{mode.upper()}**."
    )


def run_tiled_inference(
    t1_path: str,
    t2_path: str,
    patch_size: int,
    micro_batch: int,
    out_dir: str,
    grid_align: str,
):
    if _SESSION.model is None or _SESSION.device is None:
        yield _empty_inference_outputs("Load the model first (Inference tab → Model).")
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
    alignment_note = ""
    grid_notes = [
        _raster_native_grid_note(t1_path, "T1"),
        _raster_native_grid_note(t2_path, "T2"),
    ]
    grid_notes = [n for n in grid_notes if n]

    try:
        pre_img = _load_rgb_f32(t1_path)
        post_img = _load_rgb_f32(t2_path)
    except Exception as e:
        yield _empty_inference_outputs(f"Failed to read images: {e}")
        return

    ref_raster_path = t1_path
    if pre_img.shape[:2] != post_img.shape[:2]:
        mode = (grid_align or "t2_to_t1").strip().lower()
        if mode not in ("t2_to_t1", "t1_to_t2"):
            yield _empty_inference_outputs(
                "When T1 and T2 sizes differ, set **Grid alignment** to "
                "`t2_to_t1` (resample T2 to T1) or `t1_to_t2` (resample T1 to T2)."
            )
            return
        try:
            if mode == "t2_to_t1":
                post_img, alignment_note = _resample_rgb_to_ref_grid(
                    t1_path, t2_path, ref_label="T1", src_label="T2"
                )
                ref_raster_path = t1_path
            else:
                pre_img, alignment_note = _resample_rgb_to_ref_grid(
                    t2_path, t1_path, ref_label="T2", src_label="T1"
                )
                ref_raster_path = t2_path
        except Exception as e:
            grid_block = "\n".join(grid_notes)
            extra = f"\n{grid_block}" if grid_block else ""
            yield _empty_inference_outputs(
                f"T1 shape {pre_img.shape[:2]} != T2 shape {post_img.shape[:2]}. "
                "If these look aligned in GIS, they likely have different native pixel grids. "
                f"Resampling failed ({mode}): {e}{extra}"
            )
            return

    pre_pad, post_pad, _ = _pad_pair_to_multiple(pre_img, post_img, patch_size)
    H, W = pre_pad.shape[:2]
    nh, nw = H // patch_size, W // patch_size
    n_patches = nh * nw

    coords = [(i, j) for i in range(0, H, patch_size) for j in range(0, W, patch_size)]
    change_full = np.zeros((H, W), dtype=np.int32)
    t1_full = np.zeros((H, W), dtype=np.int32)
    t2_full = np.zeros((H, W), dtype=np.int32)
    is_bcd = _SESSION.task_mode == "bcd"

    n_batches = (len(coords) + micro_batch - 1) // micro_batch
    with torch.no_grad():
        pbar = tqdm(
            range(0, len(coords), micro_batch),
            total=n_batches,
            desc="Tiled inference",
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        )
        for start in pbar:
            batch_coords = coords[start : start + micro_batch]
            tensors_pre = []
            tensors_post = []
            for (y0, x0) in batch_coords:
                crop_pre = pre_pad[y0 : y0 + patch_size, x0 : x0 + patch_size]
                crop_post = post_pad[y0 : y0 + patch_size, x0 : x0 + patch_size]
                tensors_pre.append(_to_chw_normalized(crop_pre))
                tensors_post.append(_to_chw_normalized(crop_post))
            b_pre = _stack_to_torch_batch(tensors_pre, device)
            b_post = _stack_to_torch_batch(tensors_post, device)
            if is_bcd:
                out = model(b_pre, b_post)
                change_mask = torch.argmax(out, dim=1).cpu().numpy()
                for k, (y0, x0) in enumerate(batch_coords):
                    change_full[y0 : y0 + patch_size, x0 : x0 + patch_size] = change_mask[k]
            else:
                out_cd, out_t1, out_t2 = model(b_pre, b_post)
                change_mask = torch.argmax(out_cd, dim=1).cpu().numpy()
                pred_t1 = torch.argmax(out_t1, dim=1).cpu().numpy()
                pred_t2 = torch.argmax(out_t2, dim=1).cpu().numpy()
                for k, (y0, x0) in enumerate(batch_coords):
                    change_full[y0 : y0 + patch_size, x0 : x0 + patch_size] = change_mask[k]
                    t1_full[y0 : y0 + patch_size, x0 : x0 + patch_size] = pred_t1[k] * change_mask[k]
                    t2_full[y0 : y0 + patch_size, x0 : x0 + patch_size] = pred_t2[k] * change_mask[k]
            # Force terminal/log update so progress is visible during execution.
            pbar.set_postfix_str(f"tiles {min(start + micro_batch, n_patches)}/{n_patches}", refresh=True)
            sys.stdout.flush()

    h0, w0 = pre_img.shape[0], pre_img.shape[1]
    change_vis = change_full[:h0, :w0].astype(np.uint8) * 255
    cm = change_full[:h0, :w0]
    t1_crop = t1_full[:h0, :w0]
    t2_crop = t2_full[:h0, :w0]

    pre_scene_small = _to_display_rgb(_downsample_to_max_side(pre_img, MAX_SCENE_PREVIEW_SIDE))
    post_scene_small = _to_display_rgb(_downsample_to_max_side(post_img, MAX_SCENE_PREVIEW_SIDE))
    cm_scene = _downsample_to_max_side(cm, MAX_SCENE_PREVIEW_SIDE)
    scene_change_overlay = _change_overlay(post_scene_small, cm_scene)

    if is_bcd:
        scene_sem_t1_overlay = None
        scene_sem_t2_overlay = None
        rgb_t1 = None
        rgb_t2 = None
        patch_samples, patch_gallery = _build_patch_visuals_bcd(
            pre_img, post_img, cm, patch_size, nh, nw, scene_change_overlay
        )
    else:
        rgb_t1 = _semantic_rgb(t1_crop, cm)
        rgb_t2 = _semantic_rgb(t2_crop, cm)
        t1_scene = _downsample_to_max_side(t1_crop, MAX_SCENE_PREVIEW_SIDE)
        t2_scene = _downsample_to_max_side(t2_crop, MAX_SCENE_PREVIEW_SIDE)
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

    default_out = "gradio_bcd" if is_bcd else "gradio_scd"
    out_root = Path((out_dir or "").strip() or str(_PROJECT_ROOT / "outputs" / default_out))
    out_root.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = out_root / run_ts
    out_run.mkdir(parents=True, exist_ok=True)
    stem = Path(t1_path).stem
    p_change = out_run / f"{stem}_pred_GT_CD.png"
    p_t1 = out_run / f"{stem}_pred_semantic_T1.png"
    p_t2 = out_run / f"{stem}_pred_semantic_T2.png"
    imageio.imwrite(str(p_change), change_vis)
    saved_lines = [f"- {p_change}"]
    if not is_bcd:
        assert rgb_t1 is not None and rgb_t2 is not None
        imageio.imwrite(str(p_t1), rgb_t1)
        imageio.imwrite(str(p_t2), rgb_t2)
        saved_lines.append(f"- {p_t1}")
        saved_lines.append(f"- {p_t2}")
    infer_shp_msg = ""
    if is_bcd:
        infer_shp_msg = (
            "\n\n[Vector export]\n"
            "Status: SKIPPED (BCD mode)\n"
            "Unified SCD shapefile export needs semantic T1/T2 maps; use SCD mode for that export."
        )
    else:
        try:
            shp_path, n_kept, n_drop_small = _export_pred_unified_shapefile(
                cm,
                t1_crop,
                t2_crop,
                ref_raster_path=ref_raster_path,
                out_dir=str(out_run),
                stem=stem,
            )
            drop_line = (
                f"Omitted (< {SHAPEFILE_EXPORT_MIN_AREA_M2:.0f} m²): {n_drop_small}\n"
                if n_drop_small
                else ""
            )
            infer_shp_msg = (
                "\n\n[Vector export]\n"
                "Status: OK\n"
                f"Shapefile: {shp_path}\n"
                f"Features: {n_kept}\n"
                f"{drop_line}"
                "Fields: change, scd_cls, t1_cls, t2_cls"
            )
        except Exception as e:
            infer_shp_msg = (
                "\n\n[Vector export]\n"
                "Status: SKIPPED\n"
                f"Reason: {e}"
            )

    _SESSION.ref_raster_path = ref_raster_path
    _SESSION.pred_change_mask = cm.copy()
    if is_bcd:
        _SESSION.pred_sem_t1 = None
        _SESSION.pred_sem_t2 = None
    else:
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
    _SESSION.last_out_dir = str(out_run)
    _SESSION.last_stem = stem

    mode_note = "BCD (binary change only)" if is_bcd else "SCD (change + semantics)"
    status_tail = (
        f"Mode: **{mode_note}**. Input scene: {h0}x{w0}px, patch grid: {nh} x {nw} ({n_patches} tiles), "
        f"patch size: {patch_size}px, micro-batch: {micro_batch}\n"
        f"Outputs folder (timestamped): {out_run}\n"
        "UI view: downsampled full-scene context + representative patch explorer.\n\n"
        f"Saved:\n" + "\n".join(saved_lines) + f"{infer_shp_msg}\n\n"
        "Predictions are cached for **Evaluate** (same pixel grid as this run, unpadded)."
    )
    if grid_notes:
        status_tail = "\n".join(grid_notes) + "\n\n" + status_tail
    if alignment_note:
        status_tail = f"{alignment_note}\n\n{status_tail}"

    sem1_vis = gr.update(value=scene_sem_t1_overlay, visible=not is_bcd)
    sem2_vis = gr.update(value=scene_sem_t2_overlay, visible=not is_bcd)
    sem1_saved = gr.update(value=rgb_t1, visible=not is_bcd)
    sem2_saved = gr.update(value=rgb_t2, visible=not is_bcd)

    yield (
        pre_scene_small,
        post_scene_small,
        scene_change_overlay,
        sem1_vis,
        sem2_vis,
        np.stack([change_vis] * 3, axis=-1),
        sem1_saved,
        sem2_saved,
        patch_dropdown,
        patch_scene,
        patch_inputs,
        patch_preds,
        patch_gallery,
        patch_info,
        str(out_run),
        status_tail,
        "",
        "",
    )


def _run_tiled_inference_from_data(
    t1_rel: str,
    t2_rel: str,
    patch_size: int,
    micro_batch: int,
    out_dir: str,
    grid_align: str,
):
    t1 = _resolve_project_relative(t1_rel)
    t2 = _resolve_project_relative(t2_rel)
    if not t1 or not t2:
        yield _empty_inference_outputs(
            "Pick **T1** and **T2** from the data folder list (click **Refresh data files** if the list is empty)."
        )
        return
    yield from run_tiled_inference(t1, t2, patch_size, micro_batch, out_dir, grid_align)


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
    is_bcd = _SESSION.task_mode == "bcd"

    gt_cd_s = (gt_cd_path or "").strip()
    gt_t1_s = (gt_t1_path or "").strip()
    gt_t2_s = (gt_t2_path or "").strip()
    if is_bcd:
        if not gt_cd_s:
            return "", "Provide **GT_CD** (binary change ground truth)."
    else:
        if not (gt_cd_s and gt_t1_s and gt_t2_s):
            return "", "Provide all three paths: GT_CD, GT_T1, and GT_T2."
        assert t1_crop is not None and t2_crop is not None

    h0, w0 = cm.shape[:2]
    ref_grid = _SESSION.ref_raster_path
    vfield = (vector_label_field or "").strip() or None

    if is_bcd:
        try:
            gt_cd, _gt_t1, _gt_t2 = _load_gt_maps(
                gt_cd_s,
                None,
                None,
                (h0, w0),
                ref_raster_path=ref_grid,
                vector_label_field=vfield,
            )
        except Exception as e:
            return "", str(e)
        if gt_cd is None:
            return "", "Failed to load GT_CD."
        labels_cd = (gt_cd > 0.5).astype(np.int32)
        preds_cd = (cm > 0).astype(np.int32)
        ev = Evaluator(num_class=2)
        ev.add_batch(labels_cd, preds_cd)
        oa = ev.Pixel_Accuracy()
        f1 = ev.Pixel_F1_score()
        iou = ev.Intersection_over_Union()
        pre = ev.Pixel_Precision_Rate()
        rec = ev.Pixel_Recall_Rate()
        kc = ev.Kappa_coefficient()
        metrics_md = (
            "| Metric (binary change) | Value |\n|:---|---:|\n"
            f"| OA | {oa:.4f} |\n"
            f"| Precision | {pre:.4f} |\n"
            f"| Recall | {rec:.4f} |\n"
            f"| F1 | {f1:.4f} |\n"
            f"| IoU (change) | {iou:.4f} |\n"
            f"| Kappa | {kc:.4f} |\n"
        )
        shp_msg = (
            "\n\n[Vector export]\n"
            "Status: SKIPPED (BCD evaluation — no unified SCD shapefile)\n"
            "Use SCD mode if you need the combined vector export."
        )
        status = f"Evaluated binary change on {h0}×{w0}px.{shp_msg}"
        return metrics_md, status

    try:
        gt_cd, gt_t1, gt_t2 = _load_gt_maps(
            gt_cd_s,
            gt_t1_s,
            gt_t2_s,
            (h0, w0),
            ref_raster_path=ref_grid,
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
    shp_msg = ""
    try:
        if not ref_grid:
            raise ValueError("Missing reference raster path from inference cache.")
        out_dir = _SESSION.last_out_dir or str(_PROJECT_ROOT / "outputs" / "gradio_scd")
        stem = _SESSION.last_stem or Path(ref_grid).stem
        shp_path, n_kept, n_drop_small = _export_pred_unified_shapefile(
            cm,
            t1_crop,
            t2_crop,
            ref_raster_path=ref_grid,
            out_dir=out_dir,
            stem=stem,
        )
        drop_line = (
            f"Omitted (< {SHAPEFILE_EXPORT_MIN_AREA_M2:.0f} m²): {n_drop_small}\n"
            if n_drop_small
            else ""
        )
        shp_msg = (
            "\n\n[Vector export]\n"
            f"Status: OK\n"
            f"Shapefile: {shp_path}\n"
            f"Features: {n_kept}\n"
            f"{drop_line}"
            "Fields: change, scd_cls, t1_cls, t2_cls"
        )
    except Exception as e:
        shp_msg = (
            "\n\n[Vector export]\n"
            "Status: SKIPPED\n"
            f"Reason: {e}"
        )

    status = f"Evaluated on {h0}×{w0} (SCD 37-class encoding, same as training validation).{shp_msg}"
    return metrics_md, status


def _run_evaluation_from_data(
    gt_cd_rel: str,
    gt_t1_rel: str,
    gt_t2_rel: str,
    vector_label_field: str,
):
    gt_cd = _resolve_project_relative(gt_cd_rel)
    gt_t1 = _resolve_project_relative(gt_t1_rel)
    gt_t2 = _resolve_project_relative(gt_t2_rel)
    missing: list[str] = []
    if not gt_cd_rel or not gt_cd:
        missing.append("GT_CD")
    if _SESSION.task_mode != "bcd":
        if not gt_t1_rel or not gt_t1:
            missing.append("GT_T1")
        if not gt_t2_rel or not gt_t2:
            missing.append("GT_T2")
    if missing:
        return "", f"Choose valid files for: {', '.join(missing)} (paths must exist under data/)."
    return run_evaluation(gt_cd, gt_t1, gt_t2, vector_label_field)


def _refresh_data_dropdowns():
    ch = _scan_data_file_choices()
    d1, d2 = _pick_default_t1_t2(ch)
    return (
        gr.update(choices=ch, value=d1),
        gr.update(choices=ch, value=d2),
        gr.update(choices=ch, value=None),
        gr.update(choices=ch, value=None),
        gr.update(choices=ch, value=None),
        f"Scanned **{len(ch)}** selectable file(s) under `data/` (geo rasters and `.shp`).",
    )


def _on_task_mode_change(mode: str):
    """Toggle labels and visibility when user switches SCD vs BCD before loading a model."""
    scd = (mode or "").strip().lower() != "bcd"
    ck_lbl = "Trained SCD checkpoint (optional)" if scd else "Trained BCD checkpoint (optional)"
    ck_ph = (
        "ChangeMambaSCD .pth — clear to load backbone only"
        if scd
        else "ChangeMambaBCD .pth (e.g. MambaBCD_Tiny_*.pth) — clear for backbone only"
    )
    eval_intro_scd = (
        "Use this tab after **Inference** finishes. Predictions from the last run are cached for metrics and shapefile export.\n\n"
        "**SCD mode:** provide GT_CD, GT_T1, and GT_T2. Metrics follow the **37-class SCD** encoding from training.\n\n"
        "GT rasters must match **prediction resolution** (same grid as inference). "
        "Vector GT is rasterized onto the **reference raster** from that run (T1 path if T2→T1, else T2).\n\n"
        "**Refresh data files** on the Inference tab updates the GT dropdowns here too."
    )
    eval_intro_bcd = (
        "Use this tab after **Inference** with a **BCD** model. Only **GT_CD** (binary change) is required; "
        "GT_T1 / GT_T2 and the vector label field are hidden and ignored.\n\n"
        "Metrics: OA, precision, recall, F1, IoU, and Kappa on the change class (same style as `infer_MambaBCD`).\n\n"
        "**Refresh data files** on the Inference tab updates the GT dropdowns here too."
    )
    patch_lbl = (
        "Patch predictions (change | semantic T1 | semantic T2)"
        if scd
        else "Patch prediction (binary change on T2)"
    )
    return (
        gr.update(label=ck_lbl, placeholder=ck_ph),
        gr.update(visible=scd),
        gr.update(visible=scd),
        gr.update(visible=scd),
        gr.update(visible=scd),
        gr.update(visible=scd),
        gr.update(label=patch_lbl),
        gr.update(value=eval_intro_scd if scd else eval_intro_bcd),
    )


def build_app():
    default_cfg = str(
        Path(__file__).resolve().parent
        / "ChangeMamba"
        / "changedetection"
        / "configs"
        / "vssm1"
        / "vssm_base_224.yaml"
    )
    data_choices = _scan_data_file_choices()
    default_t1, default_t2 = _pick_default_t1_t2(data_choices)

    with gr.Blocks(title="ChangeMamba — large-image change detection (SCD / BCD)", theme=gr.themes.Default()) as demo:
        gr.Markdown(
            "## Large-image change detection (SCD or BCD)\n"
            "Pick **task mode** below, then **Load model** with the matching **ChangeMambaSCD** or **ChangeMambaBCD** weights. "
            "1. **Load model** → 2. **Run tiled inference** (default 256×256 patches) → 3. **Evaluate vs GT** (optional).\n\n"
            f"T1, T2, and GT inputs are **dropdowns** over files under **`data/`** (see `{_DATA_ROOT}`). "
            "Supported: common image/geo rasters and `.shp`. Use **Refresh data files** after adding assets.\n\n"
            "**SCD** shows semantic overlays on T1/T2 and saves colored semantic maps plus optional unified shapefile export. "
            "**BCD** shows only the **binary change** overlay and patch cards (no semantic prediction panels); "
            "evaluation needs **GT_CD** only.\n\n"
            f"In SCD mode, the JL1 class legend uses **{NUM_CLASSES}** semantic indices (0–5). "
            "SCD metrics use the **37-class** encoding from training. "
            "GT rasters must match **prediction resolution** (unpadded image size; same grid as the chosen alignment). "
            "Vector GT is rasterized with **rasterio** onto the **reference image grid** from the last inference. "
            "Install **`pip install rasterio fiona`** if you use shapefiles."
        )
        task_mode = gr.Radio(
            label="Task mode (before Load model)",
            choices=[
                ("Semantic change detection (ChangeMambaSCD)", "scd"),
                ("Binary change detection (ChangeMambaBCD)", "bcd"),
            ],
            value="scd",
        )
        with gr.Tabs():
            with gr.Tab("Inference"):
                gr.Markdown("### Model")
                cfg_in = gr.Textbox(label="Config YAML", value=default_cfg)
                pretrain_in = gr.Textbox(
                    label="Backbone pretrained checkpoint (optional)",
                    placeholder="path to ImageNet backbone .pth",
                )
                ckpt_in = gr.Textbox(
                    label="Trained SCD checkpoint (optional)",
                    value=_DEFAULT_SCD_CHECKPOINT,
                    placeholder="ChangeMambaSCD .pth — clear to load backbone only",
                )
                cuda_chk = gr.Checkbox(label="Use CUDA", value=True)
                load_btn = gr.Button("Load model", variant="primary")
                load_status = gr.Textbox(label="Model status", lines=3)

                gr.Markdown("### Data & tiled inference")
                with gr.Row():
                    data_refresh = gr.Button("Refresh data files")
                    data_scan_status = gr.Markdown(
                        value=(
                            f"**{len(data_choices)}** selectable file(s) under `data/`. "
                            "Defaults prefer paths whose folders are named `T1` / `T2`."
                        )
                    )
                t1 = gr.Dropdown(
                    label="T1 image (before) — from data/",
                    choices=data_choices,
                    value=default_t1,
                    allow_custom_value=False,
                )
                t2 = gr.Dropdown(
                    label="T2 image (after) — from data/",
                    choices=data_choices,
                    value=default_t2,
                    allow_custom_value=False,
                )
                grid_align = gr.Radio(
                    label="When T1 and T2 pixel grids differ (height×width)",
                    choices=[
                        ("Resample T2 → T1 grid (keep T1 pixels, warp T2)", "t2_to_t1"),
                        ("Resample T1 → T2 grid (keep T2 pixels, warp T1)", "t1_to_t2"),
                    ],
                    value="t2_to_t1",
                )
                psz = gr.Number(label="Patch size (px)", value=256, precision=0)
                mb = gr.Number(label="Micro-batch (patches per forward)", value=4, precision=0)
                odir = gr.Textbox(
                    label="Output directory",
                    placeholder="default: <repo>/outputs/gradio_scd or gradio_bcd from task mode",
                )
                run_btn = gr.Button("Run tiled inference", variant="primary")
                class_legend = ", ".join(f"{idx}: {name}" for idx, name in enumerate(JL1_CLASSES))
                jl1_legend_md = gr.Markdown(f"**JL1 legend (SCD only)** — {class_legend}")
                with gr.Accordion("Large-scene viewer", open=True):
                    with gr.Row():
                        scene_t1_prev = gr.Image(label="T1 overview (downsampled)", type="numpy")
                        scene_t2_prev = gr.Image(label="T2 overview (downsampled)", type="numpy")
                    with gr.Row():
                        scene_change_prev = gr.Image(label="Change overlay on T2", type="numpy")
                        scene_sem1_prev = gr.Image(label="Semantic overlay on T1 (SCD)", type="numpy")
                        scene_sem2_prev = gr.Image(label="Semantic overlay on T2 (SCD)", type="numpy")
                with gr.Accordion("Representative patch explorer", open=True):
                    patch_choice = gr.Dropdown(label="Representative patch", choices=[], value=None, interactive=True)
                    patch_info = gr.Textbox(label="Patch details", lines=5)
                    patch_scene = gr.Image(label="Patch location in scene", type="numpy")
                    patch_inputs = gr.Image(label="Patch inputs (T1 | T2)", type="numpy")
                    patch_preds = gr.Image(label="Patch predictions (change | semantic T1 | semantic T2)", type="numpy")
                    patch_gallery = gr.Gallery(
                        label="Representative patch cards",
                        columns=3,
                        rows=2,
                        object_fit="contain",
                        height="auto",
                    )
                with gr.Accordion("Saved stitched outputs", open=False):
                    change_prev = gr.Image(label="Predicted change map (stitched)", type="numpy")
                    sem1_prev = gr.Image(label="Predicted semantic T1 (colored, SCD)", type="numpy")
                    sem2_prev = gr.Image(label="Predicted semantic T2 (colored, SCD)", type="numpy")
                out_path = gr.Textbox(label="Output folder used")
                run_status = gr.Textbox(label="Inference log", lines=6)

            with gr.Tab("Evaluation"):
                eval_intro = gr.Markdown(
                    "Use this tab after **Inference** finishes. Predictions from the last run are cached for metrics and shapefile export.\n\n"
                    "**SCD mode:** provide GT_CD, GT_T1, and GT_T2. Metrics follow the **37-class SCD** encoding from training.\n\n"
                    "GT rasters must match **prediction resolution** (same grid as inference). "
                    "Vector GT is rasterized onto the **reference raster** from that run (T1 path if T2→T1, else T2).\n\n"
                    "**Refresh data files** on the Inference tab updates the GT dropdowns here too."
                )
                gt_cd = gr.Dropdown(
                    label="GT_CD — from data/ (raster or .shp → binary mask)",
                    choices=data_choices,
                    value=None,
                    allow_custom_value=False,
                )
                gt_t1 = gr.Dropdown(
                    label="GT_T1 semantics — from data/",
                    choices=data_choices,
                    value=None,
                    allow_custom_value=False,
                )
                gt_t2 = gr.Dropdown(
                    label="GT_T2 semantics — from data/",
                    choices=data_choices,
                    value=None,
                    allow_custom_value=False,
                )
                vec_lbl = gr.Textbox(
                    label="Vector label field (optional)",
                    placeholder="e.g. class_id — for .shp semantic layers; leave empty to auto-pick",
                    lines=1,
                )
                eval_btn = gr.Button("Evaluate vs ground truth", variant="primary")
                metrics = gr.Markdown()
                eval_status = gr.Textbox(label="Evaluation log", lines=8)

        task_mode.change(
            _on_task_mode_change,
            task_mode,
            [
                ckpt_in,
                scene_sem1_prev,
                scene_sem2_prev,
                sem1_prev,
                sem2_prev,
                jl1_legend_md,
                patch_preds,
                eval_intro,
            ],
        )
        load_btn.click(load_model_fn, [task_mode, cfg_in, pretrain_in, ckpt_in, cuda_chk], load_status)
        data_refresh.click(
            _refresh_data_dropdowns,
            inputs=None,
            outputs=[t1, t2, gt_cd, gt_t1, gt_t2, data_scan_status],
        )
        run_btn.click(
            _run_tiled_inference_from_data,
            [t1, t2, psz, mb, odir, grid_align],
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
        patch_choice.change(
            show_patch_details,
            patch_choice,
            [patch_scene, patch_inputs, patch_preds, patch_info],
        )
        eval_btn.click(_run_evaluation_from_data, [gt_cd, gt_t1, gt_t2, vec_lbl], [metrics, eval_status])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_app()
    demo.queue()
    port = _find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} in use; launching on {port}.", file=sys.stderr)
    demo.launch(server_name=args.host, server_port=port, share=args.share)


if __name__ == "__main__":
    main()
