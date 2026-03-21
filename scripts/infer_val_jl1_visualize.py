#!/usr/bin/env python3
"""
Run ChangeMamba SCD inference on the JL1 val split, save each visualization panel
as a separate image, and write a per-sample grid figure (same layout as visualize_jl1_sample,
plus predicted change / semantic maps).

Example:
  conda activate mambascd
  export PYTHONPATH=/path/to/SCD-CropLand-HZ/src
  python scripts/infer_val_jl1_visualize.py
  python scripts/infer_val_jl1_visualize.py --random-one --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Repo root = parent of scripts/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml

from ChangeMamba.changedetection.configs.config import get_config
from ChangeMamba.changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset
from ChangeMamba.changedetection.models.ChangeMambaSCD import ChangeMambaSCD

from datasets.colormap import JL1_COLORMAP, JL1_CLASSES, index2color


def _training_yaml_to_arg_defaults(raw: dict, project_root: Path) -> dict:
    path_keys = frozenset({
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
    out = {}
    for k, v in raw.items():
        if v is None:
            continue
        if k in path_keys and isinstance(v, str) and v.strip():
            p = Path(v).expanduser()
            if not p.is_absolute():
                p = (project_root / p).resolve()
            out[k] = str(p)
        else:
            out[k] = v
    return out


def load_raw_rgb(dataset_path: Path, rel_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load T1/T2 RGB uint8 for display (no ImageNet norm)."""
    t1 = np.asarray(imageio.imread(str(dataset_path / "T1" / rel_name)))
    t2 = np.asarray(imageio.imread(str(dataset_path / "T2" / rel_name)))
    if t1.ndim == 2:
        t1 = np.stack([t1] * 3, axis=-1)
    elif t1.shape[-1] > 3:
        t1 = t1[..., :3]
    if t2.ndim == 2:
        t2 = np.stack([t2] * 3, axis=-1)
    elif t2.shape[-1] > 3:
        t2 = t2[..., :3]
    return t1.astype(np.uint8), t2.astype(np.uint8)


def gt_semantic_rgb(gt: np.ndarray) -> np.ndarray:
    """Colorize semantic GT; white background for class 0 (unlabeled)."""
    gt = np.asarray(gt, dtype=np.int32)
    rgb = index2color(gt).copy()
    rgb[gt == 0] = [255, 255, 255]
    return rgb.astype(np.uint8)


def pred_semantic_rgb(pred: np.ndarray, change_mask: np.ndarray) -> np.ndarray:
    """Colorize semantic predictions; white where no change."""
    pred = np.asarray(pred, dtype=np.int32)
    cm = np.asarray(change_mask, dtype=np.int32)
    rgb = index2color(pred).copy()
    rgb[cm == 0] = [255, 255, 255]
    return rgb.astype(np.uint8)


def save_legend_png(out_path: Path) -> None:
    if out_path.exists():
        return
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="white")
    patches = [
        plt.matplotlib.patches.Patch(color=np.array(c) / 255, label=name)
        for c, name in zip(JL1_COLORMAP, JL1_CLASSES)
    ]
    ax.legend(handles=patches, loc="center", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def build_grid(
    t1: np.ndarray,
    t2: np.ndarray,
    gt_cd: np.ndarray,
    gt_t1_rgb: np.ndarray,
    gt_t2_rgb: np.ndarray,
    pred_cd: np.ndarray,
    pred_t1_rgb: np.ndarray,
    pred_t2_rgb: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """3x3 grid: T1, T2, GT_CD | GT_T1, GT_T2, Pred_CD | Pred_T1, Pred_T2, legend."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 14), facecolor="white")
    axes[0, 0].imshow(t1)
    axes[0, 0].set_title("T1 (before)")
    axes[0, 1].imshow(t2)
    axes[0, 1].set_title("T2 (after)")
    axes[0, 2].imshow(gt_cd)
    axes[0, 2].set_title("GT_CD")

    axes[1, 0].imshow(gt_t1_rgb)
    axes[1, 0].set_title("GT_T1")
    axes[1, 1].imshow(gt_t2_rgb)
    axes[1, 1].set_title("GT_T2")
    axes[1, 2].imshow(pred_cd)
    axes[1, 2].set_title("Pred_CD")

    axes[2, 0].imshow(pred_t1_rgb)
    axes[2, 0].set_title("Pred_T1")
    axes[2, 1].imshow(pred_t2_rgb)
    axes[2, 1].set_title("Pred_T2")

    legend_patches = [
        plt.matplotlib.patches.Patch(color=np.array(c) / 255, label=name)
        for c, name in zip(JL1_COLORMAP, JL1_CLASSES)
    ]
    axes[2, 2].legend(handles=legend_patches, loc="center", fontsize=8)
    axes[2, 2].axis("off")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _to_numpy_hw(x: torch.Tensor) -> np.ndarray:
    a = x.detach().cpu().numpy()
    if a.ndim == 3:
        a = a[0]
    return np.squeeze(a)


_TRAINING_CKPT_META_KEYS = frozenset(
    {"optimizer", "scheduler", "step", "best_kc", "best_round"}
)


def _extract_model_state_dict(loaded: object) -> tuple[dict, str]:
    """Split ChangeMamba training artifacts from the tensor state dict.

    Training writes two shapes (see train_MambaSCD.py):
    - ``best_model.pth``: ``torch.save(model.state_dict())`` — flat parameter dict only.
    - ``{step}_model.pth``: dict with ``model``, ``optimizer``, ``scheduler``, scalars, etc.

    Both are valid for ``--resume``; optimizer tensors are ignored here.
    """
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
        return loaded, "weights-only state_dict (e.g. best_model.pth)"

    raise ValueError(
        "Could not find a model state_dict in this file. "
        "Expected best_model.pth (flat weights) or *_model.pth with a 'model' entry."
    )


def _run(args):
    train_config = Path(args.train_config).expanduser()
    if not train_config.is_absolute():
        train_config = (_PROJECT_ROOT / train_config).resolve()
    with open(train_config, "r", encoding="utf-8") as f:
        yaml_defaults = _training_yaml_to_arg_defaults(yaml.safe_load(f), _PROJECT_ROOT)

    pretrained_weight_path = yaml_defaults.get("pretrained_weight_path")
    if not pretrained_weight_path:
        raise ValueError("train-config must set pretrained_weight_path (VMamba backbone).")

    resume_str = args.resume if args.resume is not None else yaml_defaults.get("resume")
    resume_path = Path(resume_str).expanduser() if resume_str else None
    if resume_path is not None and not resume_path.is_absolute():
        resume_path = (_PROJECT_ROOT / resume_path).resolve()

    cfg_args = argparse.Namespace(
        cfg=yaml_defaults["cfg"],
        opts=yaml_defaults.get("opts"),
    )
    config = get_config(cfg_args)

    if resume_path is None:
        raise ValueError("Set --resume or resume: in train-config.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "VMamba / ChangeMamba inference requires CUDA (selective_scan CUDA ops). "
            "Use a GPU, `conda activate mambascd`, and do not pass --no-cuda."
        )

    deep_model = ChangeMambaSCD(
        output_cd=2,
        output_clf=7,
        pretrained=pretrained_weight_path,
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
    deep_model = deep_model.to(device)

    raw_ckpt = torch.load(str(resume_path), map_location="cpu")
    ckpt_weights, ckpt_kind = _extract_model_state_dict(raw_ckpt)
    state_dict = deep_model.state_dict()
    model_dict = {k: v for k, v in ckpt_weights.items() if k in state_dict}
    state_dict.update(model_dict)
    deep_model.load_state_dict(state_dict)
    deep_model.eval()
    print(f"=> resume {resume_path} ({ckpt_kind}; {len(model_dict)}/{len(state_dict)} params matched)")
    if len(model_dict) < len(state_dict):
        missing = sorted(set(state_dict) - set(model_dict))
        print(f"   warning: checkpoint missing {len(missing)} keys (kept init/pretrained), e.g. {missing[:5]}")

    test_dataset_path = (
        Path(args.test_dataset_path).expanduser()
        if args.test_dataset_path
        else Path(yaml_defaults["test_dataset_path"])
    )
    if not test_dataset_path.is_absolute():
        test_dataset_path = (_PROJECT_ROOT / test_dataset_path).resolve()

    list_path = (
        Path(args.test_data_list_path).expanduser()
        if args.test_data_list_path
        else Path(yaml_defaults["test_data_list_path"])
    )
    if not list_path.is_absolute():
        list_path = (_PROJECT_ROOT / list_path).resolve()

    with open(list_path, "r", encoding="utf-8") as f:
        full_name_list = [line.strip() for line in f if line.strip()]

    if not full_name_list:
        raise RuntimeError(f"Val list is empty: {list_path}")

    if args.random_one:
        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        pick = rng.choice(full_name_list)
        name_list = [pick]
        print(f"=> random-one from val ({len(full_name_list)} tiles): {pick}")
    elif args.max_samples is not None and args.max_samples > 0:
        name_list = full_name_list[: args.max_samples]
    else:
        name_list = full_name_list

    infer_size = args.eval_crop_size
    dataset = SemanticChangeDetectionDatset(
        str(test_dataset_path), name_list, infer_size, None, "test"
    )
    loader = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, drop_last=False
    )

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (_PROJECT_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    save_legend_png(out_root / "legend.png")

    with torch.no_grad():
        for data in tqdm(loader, desc="val infer"):
            pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, names = data
            rel_name = names[0]
            stem = Path(rel_name).stem

            pre_change_imgs = pre_change_imgs.to(device)
            post_change_imgs = post_change_imgs.to(device)

            output_1, output_semantic_t1, output_semantic_t2 = deep_model(
                pre_change_imgs, post_change_imgs
            )

            change_mask = torch.argmax(output_1, dim=1)
            preds_a = torch.argmax(output_semantic_t1, dim=1)
            preds_b = torch.argmax(output_semantic_t2, dim=1)
            preds_a = (preds_a * change_mask.long()).squeeze(0)
            preds_b = (preds_b * change_mask.long()).squeeze(0)

            pred_cd = change_mask.squeeze(0).cpu().numpy().astype(np.uint8)
            pred_cd_viz = (np.stack([pred_cd] * 3, axis=-1) * 255).astype(np.uint8)

            pa = preds_a.cpu().numpy().astype(np.int32)
            pb = preds_b.cpu().numpy().astype(np.int32)
            cm = change_mask.squeeze(0).cpu().numpy().astype(np.int32)
            pred_t1_rgb = pred_semantic_rgb(pa, cm)
            pred_t2_rgb = pred_semantic_rgb(pb, cm)

            gt_cd = _to_numpy_hw(label_cd)
            gt_cd_u8 = (gt_cd * 255).astype(np.uint8)
            gt_cd_viz = np.stack([gt_cd_u8] * 3, axis=-1)

            gt_t1 = _to_numpy_hw(label_clf_t1).astype(np.int32)
            gt_t2 = _to_numpy_hw(label_clf_t2).astype(np.int32)
            gt_t1_rgb = gt_semantic_rgb(gt_t1)
            gt_t2_rgb = gt_semantic_rgb(gt_t2)

            t1_rgb, t2_rgb = load_raw_rgb(test_dataset_path, rel_name)

            sample_dir = out_root / stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            imageio.imwrite(sample_dir / "01_t1.png", t1_rgb)
            imageio.imwrite(sample_dir / "02_t2.png", t2_rgb)
            imageio.imwrite(sample_dir / "03_gt_cd.png", gt_cd_viz)
            imageio.imwrite(sample_dir / "04_gt_t1_sem.png", gt_t1_rgb)
            imageio.imwrite(sample_dir / "05_gt_t2_sem.png", gt_t2_rgb)
            imageio.imwrite(sample_dir / "06_pred_cd.png", pred_cd_viz)
            imageio.imwrite(sample_dir / "07_pred_t1_sem.png", pred_t1_rgb)
            imageio.imwrite(sample_dir / "08_pred_t2_sem.png", pred_t2_rgb)

            grid_path = sample_dir / "grid.png"
            build_grid(
                t1_rgb,
                t2_rgb,
                gt_cd_viz,
                gt_t1_rgb,
                gt_t2_rgb,
                pred_cd_viz,
                pred_t1_rgb,
                pred_t2_rgb,
                f"JL1 val — {stem}",
                grid_path,
            )

    print(f"Done. Outputs under: {out_root}")


def main():
    parser = argparse.ArgumentParser(description="JL1 val inference + per-panel + grid visualization")
    parser.add_argument(
        "--train-config",
        type=str,
        default=str(_PROJECT_ROOT / "configs/train_changemamba_scd_vmamba_base.yaml"),
        help="Training YAML (paths, backbone, etc.)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=str(_PROJECT_ROOT / "models/BiliSakura/JL1-ChangeMambaSCD/Base/best_model.pth"),
        help=(
            "Fine-tuned weights: best_model.pth (state_dict only) or "
            "{step}_model.pth (dict with 'model', 'optimizer', … — only 'model' is used)"
        ),
    )
    parser.add_argument(
        "--test-dataset-path",
        type=str,
        default=None,
        help="Override val dataset root (default: from train-config)",
    )
    parser.add_argument(
        "--test-data-list-path",
        type=str,
        default=None,
        help="Override val list file (default: from train-config)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/val_infer_jl1_base_vis",
        help="Output directory (created if missing)",
    )
    parser.add_argument("--eval-crop-size", type=int, default=256, help="Must match training eval (JL1 full tile)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap val tiles (omit for all when invoking python directly). Use 0 for all when a wrapper passes a default.",
    )
    parser.add_argument(
        "--random-one",
        action="store_true",
        help="Pick one random val tile (from the full list; overrides --max-samples cap on which tiles are eligible).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --random-one (optional)")
    parser.add_argument("--no-cuda", action="store_true", help="Run on CPU (slow)")

    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
