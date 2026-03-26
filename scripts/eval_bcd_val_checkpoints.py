#!/usr/bin/env python3
"""
Binary change detection (BCD) metrics on the cropland val manifest — **ChangeMambaSCD** or **ChangeMambaBCD**.

By default, HZ (**input_quick**) metrics use **train + val** tiles via
``datasets/cropland_bcd_collections/train_stage3_input_quick_trainval.txt`` (``--hz-pool trainval``).
Use ``--hz-pool val`` for the 54-tile val split only, or pass ``--data-list-path`` for a custom manifest.

Checkpoints from **``scripts/train_flagship_bcd_multistage.sh``** land under
``FLAGSHIP_CKPT_ROOT`` (default ``<repo>/models/BiliSakura/ChangeMambaBCD``), with stage dirs:

- Stage 1: ``FlagshipBaseStage1_JL1/`` — ``last_model.pth``, ``{step}_model.pth``, ``checkpoint_step_*.pth``
- Stage 2: ``FlagshipBaseStage2_Joint512/``
- Stage 3: ``FlagshipBaseStage3_InputQuick/``

- **SCD** (`--model scd`): **change head only** (semantic heads ignored).
- **BCD** (`--model bcd`): full 2-class BCD model.
- **Scan** (`--scan-checkpoints-dir` / ``--scan-all-default-dir``): every ``*.pth`` under the tree (BCD).

Examples:
  conda activate mambascd
  python scripts/eval_bcd_val_checkpoints.py

  python scripts/eval_bcd_val_checkpoints.py --model scd \\
    --resume models/BiliSakura/JL1-ChangeMambaSCD/Base/best_model.pth --eval-crop-size 512

  # Last checkpoint from multistage training (same root as FLAGSHIP_CKPT_ROOT in the shell script):
  python scripts/eval_bcd_val_checkpoints.py --resume-flagship-last-stage 3

  python scripts/eval_bcd_val_checkpoints.py --scan-all-default-dir

  # Val-only HZ (54 tiles): --hz-pool val  (default is full HZ train+val, 482 tiles)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ChangeMamba.changedetection.configs.config import get_config
from ChangeMamba.changedetection.datasets.make_data_loader import CroplandCollectionsBCDDataset
from ChangeMamba.changedetection.models.ChangeMambaBCD import ChangeMambaBCD
from ChangeMamba.changedetection.models.ChangeMambaSCD import ChangeMambaSCD
from ChangeMamba.changedetection.utils_func.metrics import Evaluator

_DEFAULT_SCD_CONFIG = _PROJECT_ROOT / "configs/train_changemamba_scd_vmamba_base.yaml"
_DEFAULT_BCD_CONFIG = _PROJECT_ROOT / "configs/flagship_bcd_stage1_jl1_vmamba_base.yaml"
_DEFAULT_SCD_RESUME = _PROJECT_ROOT / "models/BiliSakura/JL1-ChangeMambaSCD/Base/best_model.pth"

# Must match model_save_path in configs/flagship_bcd_stage{1,2,3}_*.yaml and scripts/train_flagship_bcd_multistage.sh
_FLAGSHIP_STAGE_SUBDIR = {
    1: "FlagshipBaseStage1_JL1",
    2: "FlagshipBaseStage2_Joint512",
    3: "FlagshipBaseStage3_InputQuick",
}
_FLAGSHIP_STAGE_TRAIN_CONFIG = {
    1: _PROJECT_ROOT / "configs/flagship_bcd_stage1_jl1_vmamba_base.yaml",
    2: _PROJECT_ROOT / "configs/flagship_bcd_stage2_joint_vmamba_base.yaml",
    3: _PROJECT_ROOT / "configs/flagship_bcd_stage3_input_quick_vmamba_base.yaml",
}

# HZ / input_quick — same manifests as Stage 3 training (see configs/flagship_bcd_stage3_*.yaml)
_HZ_TRAINVAL_MANIFEST = (
    _PROJECT_ROOT / "datasets/cropland_bcd_collections/train_stage3_input_quick_trainval.txt"
)
_HZ_VAL_MANIFEST = _PROJECT_ROOT / "datasets/cropland_bcd_collections/val_stage3_input_quick_val.txt"


def _resolve_hz_sample_names(
    project_root: Path,
    data_list_path: str | None,
    hz_pool: str,
) -> tuple[list[str], str]:
    """Load sample list lines; returns (names, description for logs)."""
    if data_list_path is not None and str(data_list_path).strip():
        p = Path(data_list_path).expanduser()
        p = p.resolve() if p.is_absolute() else (project_root / p).resolve()
        if not p.is_file():
            raise SystemExit(f"data-list-path not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names, str(p)

    if hz_pool == "trainval":
        if not _HZ_TRAINVAL_MANIFEST.is_file():
            raise SystemExit(f"HZ train+val manifest missing: {_HZ_TRAINVAL_MANIFEST}")
        with open(_HZ_TRAINVAL_MANIFEST, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names, f"{_HZ_TRAINVAL_MANIFEST} (hz-pool=trainval)"

    if hz_pool == "val":
        if not _HZ_VAL_MANIFEST.is_file():
            raise SystemExit(f"HZ val manifest missing: {_HZ_VAL_MANIFEST}")
        with open(_HZ_VAL_MANIFEST, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names, f"{_HZ_VAL_MANIFEST} (hz-pool=val)"

    if hz_pool == "train":
        if not _HZ_TRAINVAL_MANIFEST.is_file():
            raise SystemExit(f"HZ manifest missing: {_HZ_TRAINVAL_MANIFEST}")
        with open(_HZ_TRAINVAL_MANIFEST, "r", encoding="utf-8") as f:
            names = [
                ln.strip()
                for ln in f
                if ln.strip() and "input_quick/train/" in ln.strip().replace("\\", "/")
            ]
        return names, f"{_HZ_TRAINVAL_MANIFEST} (hz-pool=train, train tiles only)"

    raise SystemExit(f"unknown --hz-pool: {hz_pool!r} (use val, train, or trainval)")


def _resolve_flagship_ckpt_root(project_root: Path, cli_root: str | None) -> Path:
    """Align with FLAGSHIP_CKPT_ROOT in scripts/train_flagship_bcd_multistage.sh."""
    if cli_root is not None and str(cli_root).strip():
        p = Path(cli_root).expanduser()
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()
    env = os.environ.get("FLAGSHIP_CKPT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()
    return (project_root / "models" / "BiliSakura" / "ChangeMambaBCD").resolve()

_PATH_KEYS = frozenset({
    "cfg",
    "train_dataset_path",
    "train_data_list_path",
    "test_dataset_path",
    "test_data_list_path",
    "pretrained_weight_path",
    "resume",
    "model_param_path",
    "model_save_path",
})

_TRAINING_CKPT_META_KEYS = frozenset(
    {"optimizer", "scheduler", "step", "best_kc", "best_round"}
)


def _training_yaml_to_arg_defaults(raw: dict, project_root: Path) -> dict:
    out = {}
    for k, v in raw.items():
        if v is None:
            continue
        if k in _PATH_KEYS and isinstance(v, str) and v.strip():
            p = Path(v).expanduser()
            if not p.is_absolute():
                p = (project_root / p).resolve()
            out[k] = str(p)
        else:
            out[k] = v
    return out


def _extract_model_state_dict(loaded: object) -> tuple[dict, str]:
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(loaded)}")
    nested = loaded.get("model")
    if isinstance(nested, dict) and nested:
        first = next(iter(nested.values()))
        if isinstance(first, torch.Tensor):
            if _TRAINING_CKPT_META_KEYS & loaded.keys():
                return nested, "training checkpoint (model + optimizer/scheduler/…)"
            return nested, "nested state dict (model key only)"
    if any(isinstance(v, torch.Tensor) for v in loaded.values()):
        return loaded, "weights-only state_dict"
    raise ValueError(
        "Could not find a model state_dict. "
        "Expected flat weights or a dict with a 'model' tensor dict."
    )


def _load_yaml_config(train_config: Path, project_root: Path) -> tuple[object, str]:
    with open(train_config, "r", encoding="utf-8") as f:
        yaml_defaults = _training_yaml_to_arg_defaults(yaml.safe_load(f), project_root)
    pretrained = yaml_defaults.get("pretrained_weight_path")
    if not pretrained:
        raise SystemExit("train-config must set pretrained_weight_path (VMamba backbone).")
    cfg_rel = yaml_defaults["cfg"]
    cfg = str(Path(cfg_rel).expanduser())
    if not Path(cfg).is_absolute():
        cfg = str(project_root / cfg)
    config = get_config(argparse.Namespace(cfg=cfg, opts=yaml_defaults.get("opts")))
    return config, pretrained


def _build_scd(config, pretrained: str, device: torch.device) -> ChangeMambaSCD:
    return ChangeMambaSCD(
        output_cd=2,
        output_clf=7,
        pretrained=pretrained,
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=(
            "auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)
        ),
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
    ).to(device)


def _build_bcd(config, pretrained: str, device: torch.device) -> ChangeMambaBCD:
    return ChangeMambaBCD(
        pretrained=pretrained,
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=(
            "auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)
        ),
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
    ).to(device)


def _load_ckpt_into_model(model: torch.nn.Module, ckpt_path: Path) -> tuple[int, int, str]:
    raw = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_weights, ckpt_kind = _extract_model_state_dict(raw)
    state_dict = model.state_dict()
    matched = {k: v for k, v in ckpt_weights.items() if k in state_dict}
    state_dict.update(matched)
    model.load_state_dict(state_dict)
    return len(matched), len(state_dict), ckpt_kind


def _make_loader(
    dataset_root: Path,
    sample_names: list[str],
    crop: int,
    num_workers: int,
) -> DataLoader:
    if not sample_names:
        raise SystemExit("empty sample list (check --data-list-path or --hz-pool)")
    ds = CroplandCollectionsBCDDataset(
        str(dataset_root),
        sample_names,
        crop,
        None,
        "test",
        train_augment=False,
    )
    nw = max(0, int(num_workers))
    return DataLoader(ds, batch_size=1, num_workers=nw, shuffle=False, drop_last=False)


def _eval_one_checkpoint(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    arch: str,
) -> dict[str, float]:
    ev = Evaluator(num_class=2)
    ev.reset()
    model.eval()
    with torch.no_grad():
        for pre_i, post_i, labels, _idx in tqdm(loader, desc="BCD eval", unit="tile"):
            pre_i = pre_i.to(device).float()
            post_i = post_i.to(device).float()
            labels = labels.to(device).long()
            if arch == "scd":
                out_cd, _, _ = model(pre_i, post_i)
                pred = torch.argmax(out_cd, dim=1).cpu().numpy()
            else:
                out = model(pre_i, post_i)
                pred = torch.argmax(out, dim=1).cpu().numpy()
            lab = labels.cpu().numpy()
            ev.add_batch(lab, pred)
    return {
        "recall": float(ev.Pixel_Recall_Rate()),
        "precision": float(ev.Pixel_Precision_Rate()),
        "oa": float(ev.Pixel_Accuracy()),
        "f1": float(ev.Pixel_F1_score()),
        "iou": float(ev.Intersection_over_Union()),
        "kappa": float(ev.Kappa_coefficient()),
    }


_METRIC_ORDER: tuple[tuple[str, str], ...] = (
    ("Recall", "recall"),
    ("Precision", "precision"),
    ("OA", "oa"),
    ("F1", "f1"),
    ("IoU", "iou"),
    ("Kappa", "kappa"),
)


def _format_run_settings(
    *,
    arch: str,
    eval_crop_size: int,
    num_workers: int,
    n_tiles: int,
    train_config: Path,
    hz_pool: str | None,
    data_list_path: str | None,
    num_classes: int,
) -> str:
    """Single-line run / model context for table columns (no spaces in paths)."""
    parts = [
        f"arch={arch}",
        f"crop={eval_crop_size}",
        f"workers={num_workers}",
        f"tiles={n_tiles}",
        f"sem_classes={num_classes}",
    ]
    if data_list_path is not None and str(data_list_path).strip():
        parts.append("manifest=custom")
    else:
        parts.append(f"hz_pool={hz_pool}")
    parts.append(f"cfg={train_config.name}")
    return " ".join(parts)


def _print_metrics_table(
    metrics: dict[str, float],
    *,
    settings: str,
    title: str | None = None,
    settings_col_width: int = 56,
) -> None:
    """Bordered table: Metric | Value | Model settings (settings repeated per row)."""
    rows = [(label, metrics[key]) for label, key in _METRIC_ORDER]
    w_lbl = max(len(r[0]) for r in rows)
    w_lbl = max(w_lbl, len("Metric"))
    w_val = 10
    w_set = max(len("Model settings"), min(settings_col_width, max(24, len(settings))))
    set_cell = _truncate_cell(settings, w_set)
    header_line = (
        f"| {'Metric':<{w_lbl}} | {'Value':>{w_val}} | {'Model settings':<{w_set}} |"
    )
    width = len(header_line)
    sep = "+" + "-" * (width - 2) + "+"
    print("\n=== binary change metrics ===")
    print(sep)
    if title:
        tw = width - 4
        ti = title if len(title) <= tw else title[: max(0, tw - 1)] + "…"
        print(f"| {ti:<{tw}} |")
        print(sep)
    print(header_line)
    print(sep)
    for lbl, val in rows:
        print(
            f"| {lbl:<{w_lbl}} | {val:>{w_val}.6f} | {set_cell:<{w_set}} |"
        )
    print(sep)


def _truncate_cell(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    keep = max_len - 1
    left = keep // 2
    right = keep - left
    return s[:left] + "…" + s[-right:]


def _print_scan_summary_table(
    rows: list[tuple[str, dict[str, float] | None, str | None]],
    *,
    run_settings: str,
    ckpt_col_width: int = 40,
    settings_col_width: int = 44,
) -> None:
    """Print all checkpoints: checkpoint | model settings | metrics | notes."""
    headers = (
        "checkpoint",
        "model settings",
        "Recall",
        "Prec",
        "OA",
        "F1",
        "IoU",
        "Kappa",
        "notes",
    )
    num_w = 8
    w_notes = 16
    w_ckpt = ckpt_col_width
    w_set = max(len(headers[1]), min(settings_col_width, max(28, len(run_settings))))
    set_cell = _truncate_cell(run_settings, w_set)
    head_row = (
        f"| {headers[0]:<{w_ckpt}} | {headers[1]:<{w_set}} |"
        f" {headers[2]:>{num_w}} | {headers[3]:>{num_w}} | {headers[4]:>{num_w}} |"
        f" {headers[5]:>{num_w}} | {headers[6]:>{num_w}} | {headers[7]:>{num_w}} |"
        f" {headers[8]:<{w_notes}} |"
    )
    inner = len(head_row) - 2
    sep = "+" + "-" * inner + "+"
    print("\n=== metrics summary (table) ===")
    print(sep)
    print(head_row)
    print(sep)
    for rel, m, err in rows:
        ck = _truncate_cell(rel, w_ckpt)
        if m is None:
            err_s = _truncate_cell(err or "?", w_notes)
            line = (
                f"| {ck:<{w_ckpt}} | {set_cell:<{w_set}} |"
                f" {'—':>{num_w}} | {'—':>{num_w}} | {'—':>{num_w}} |"
                f" {'—':>{num_w}} | {'—':>{num_w}} | {'—':>{num_w}} |"
                f" {err_s:<{w_notes}} |"
            )
        else:
            line = (
                f"| {ck:<{w_ckpt}} | {set_cell:<{w_set}} |"
                f" {m['recall']:>{num_w}.4f} | {m['precision']:>{num_w}.4f} | {m['oa']:>{num_w}.4f} |"
                f" {m['f1']:>{num_w}.4f} | {m['iou']:>{num_w}.4f} | {m['kappa']:>{num_w}.4f} |"
                f" {'':<{w_notes}} |"
            )
        print(line)
    print(sep)


def _discover_pth_files(root: Path) -> list[Path]:
    found = sorted(root.rglob("*.pth"), key=lambda p: str(p).lower())
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BCD val metrics: ChangeMambaSCD (CD head) or ChangeMambaBCD; optional scan-all .pth under a folder."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("scd", "bcd"),
        default="scd",
        help="Architecture for single --resume (ignored when --scan-checkpoints-dir is set → always bcd).",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="YAML with cfg + pretrained_weight_path. Defaults: scd→train_changemamba_scd_vmamba_base; bcd/scan→flagship_bcd_stage1_jl1.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Single checkpoint (.pth). Default: JL1 SCD best_model when model=scd and not scanning.",
    )
    parser.add_argument(
        "--resume-flagship-last-stage",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="BCD: load last_model.pth from multistage training (see scripts/train_flagship_bcd_multistage.sh). "
        "Stage 1=FlagshipBaseStage1_JL1, 2=Joint512, 3=InputQuick. Implies --model bcd. "
        "Uses --flagship-ckpt-root / $FLAGSHIP_CKPT_ROOT. Do not combine with --resume.",
    )
    parser.add_argument(
        "--flagship-ckpt-root",
        type=str,
        default=None,
        help="Override checkpoint root (default: env FLAGSHIP_CKPT_ROOT or "
        "<repo>/models/BiliSakura/ChangeMambaBCD). Used by --scan-all-default-dir and --resume-flagship-last-stage.",
    )
    parser.add_argument(
        "--scan-checkpoints-dir",
        type=str,
        default=None,
        help="If set, evaluate every *.pth under this directory (recursive). Uses BCD.",
    )
    parser.add_argument(
        "--scan-all-default-dir",
        action="store_true",
        help="Shorthand: --scan-checkpoints-dir <flagship-ckpt-root> (all .pth from multistage tree).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(_PROJECT_ROOT / "datasets/cropland_bcd_collections"),
        help="Cropland BCD collections root.",
    )
    parser.add_argument(
        "--hz-pool",
        type=str,
        choices=("val", "train", "trainval"),
        default="trainval",
        help="HZ (input_quick) tiles when --data-list-path is omitted: val (54), train-only, or "
        "train+val (default; train_stage3_input_quick_trainval.txt).",
    )
    parser.add_argument(
        "--data-list-path",
        type=str,
        default=None,
        help="Override manifest: one relative path per line under dataset-root. "
        "If set, --hz-pool is ignored.",
    )
    parser.add_argument(
        "--eval-crop-size",
        type=int,
        default=512,
        help="512 for input_quick val tiles; 256 if your list is JL1 256px.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="CUDA is required for VMamba; this flag triggers a clear error.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="In scan mode, abort on first load/eval error instead of skipping the file.",
    )
    args = parser.parse_args()

    flagship_root = _resolve_flagship_ckpt_root(_PROJECT_ROOT, args.flagship_ckpt_root)

    if args.resume_flagship_last_stage is not None and args.resume is not None:
        raise SystemExit("Use either --resume or --resume-flagship-last-stage, not both.")

    train_config_from_flagship: Path | None = None
    if args.resume_flagship_last_stage is not None:
        sub = _FLAGSHIP_STAGE_SUBDIR[args.resume_flagship_last_stage]
        args.resume = str(flagship_root / sub / "last_model.pth")
        args.model = "bcd"
        if args.train_config is None:
            train_config_from_flagship = _FLAGSHIP_STAGE_TRAIN_CONFIG[args.resume_flagship_last_stage]

    scan_dir: Path | None = None
    if args.scan_all_default_dir:
        scan_dir = flagship_root
    if args.scan_checkpoints_dir:
        scan_dir = Path(args.scan_checkpoints_dir).expanduser()
        if not scan_dir.is_absolute():
            scan_dir = (_PROJECT_ROOT / scan_dir).resolve()

    arch = "bcd" if scan_dir is not None else args.model

    if train_config_from_flagship is not None:
        tc = train_config_from_flagship
    elif args.train_config is None:
        tc = _DEFAULT_BCD_CONFIG if arch == "bcd" else _DEFAULT_SCD_CONFIG
    else:
        tc = Path(args.train_config).expanduser()
        if not tc.is_absolute():
            tc = (_PROJECT_ROOT / tc).resolve()

    train_config = Path(tc).resolve()
    if not train_config.is_file():
        raise SystemExit(f"train-config not found: {train_config}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        raise SystemExit("ChangeMamba / VMamba requires CUDA (conda env mambascd + GPU).")

    config, pretrained = _load_yaml_config(train_config, _PROJECT_ROOT)
    num_classes = int(config.MODEL.NUM_CLASSES)

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (_PROJECT_ROOT / dataset_root).resolve()

    sample_names, manifest_desc = _resolve_hz_sample_names(
        _PROJECT_ROOT,
        args.data_list_path,
        args.hz_pool,
    )
    n_tiles = len(sample_names)
    loader = _make_loader(
        dataset_root, sample_names, args.eval_crop_size, args.num_workers
    )
    print(f"=> samples={n_tiles}  {manifest_desc}")

    run_settings = _format_run_settings(
        arch=arch,
        eval_crop_size=args.eval_crop_size,
        num_workers=args.num_workers,
        n_tiles=n_tiles,
        train_config=train_config,
        hz_pool=args.hz_pool,
        data_list_path=args.data_list_path,
        num_classes=num_classes,
    )

    if scan_dir is not None:
        if not scan_dir.is_dir():
            raise SystemExit(f"--scan-checkpoints-dir is not a directory: {scan_dir}")
        ckpts = _discover_pth_files(scan_dir)
        if not ckpts:
            print(f"=> no *.pth files under {scan_dir}")
            return
        print(
            f"=> scan {len(ckpts)} checkpoint(s) under {scan_dir} "
            f"(arch=bcd, tiles={n_tiles}, crop={args.eval_crop_size})"
        )
        rows: list[tuple[str, dict[str, float] | None, str | None]] = []
        for ckpt_path in ckpts:
            try:
                rel = str(ckpt_path.relative_to(scan_dir))
            except ValueError:
                rel = str(ckpt_path)
            try:
                model = _build_bcd(config, pretrained, device)
                n_m, n_tot, kind = _load_ckpt_into_model(model, ckpt_path)
                print(f"\n=> {rel} ({kind}; {n_m}/{n_tot} keys)")
                metrics = _eval_one_checkpoint(model, loader, device, "bcd")
                rows.append((rel, metrics, None))
                print(
                    f"   F1={metrics['f1']:.4f}  IoU={metrics['iou']:.4f}  "
                    f"Kappa={metrics['kappa']:.4f}  OA={metrics['oa']:.4f}"
                )
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                rows.append((rel, None, msg))
                print(f"\n=> {rel}  ERROR: {msg}")
                if args.fail_fast:
                    raise
        _print_scan_summary_table(rows, run_settings=run_settings)
        return

    resume = args.resume
    if resume is None:
        resume = str(_DEFAULT_SCD_RESUME) if arch == "scd" else None
    if resume is None:
        raise SystemExit("Pass --resume for single-checkpoint BCD eval, or use --scan-checkpoints-dir.")
    resume_path = Path(resume).expanduser()
    if not resume_path.is_absolute():
        resume_path = (_PROJECT_ROOT / resume_path).resolve()
    if not resume_path.is_file():
        hint = ""
        if args.resume_flagship_last_stage is not None:
            hint = (
                " Train stages with scripts/train_flagship_bcd_multistage.sh first, "
                "or set FLAGSHIP_CKPT_ROOT / --flagship-ckpt-root if checkpoints live elsewhere."
            )
        raise SystemExit(f"checkpoint not found: {resume_path}{hint}")

    if arch == "scd":
        model = _build_scd(config, pretrained, device)
    else:
        model = _build_bcd(config, pretrained, device)

    n_m, n_tot, kind = _load_ckpt_into_model(model, resume_path)
    print(f"=> resume {resume_path} ({kind}; {n_m}/{n_tot} keys matched)")
    if n_m < n_tot:
        raw = torch.load(str(resume_path), map_location="cpu")
        w, _ = _extract_model_state_dict(raw)
        miss = sorted(set(model.state_dict()) - set(w))[:10]
        print(f"   warning: checkpoint missing {n_tot - n_m} keys (kept init), e.g. {miss}")

    metrics = _eval_one_checkpoint(model, loader, device, arch)
    print(
        f"=> arch={arch}  tiles={n_tiles}  crop={args.eval_crop_size}  "
        f"checkpoint={resume_path}"
    )
    _print_metrics_table(metrics, settings=run_settings)
    print(f"dataset-root: {dataset_root}")
    print(f"manifest: {manifest_desc}")


if __name__ == "__main__":
    main()
