#!/usr/bin/env python3
"""
Gradio UI: load ChangeMambaSCD, run tiled inference on large T1/T2 pairs, save outputs,
then optionally run a separate evaluation step against GT (two buttons).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Repo root: .../src/gradio_large_image_infer.py -> parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import imageio.v2 as imageio
import numpy as np
import torch

import gradio as gr

from ChangeMamba.changedetection.configs.config import get_config
from ChangeMamba.changedetection.datasets import imutils
from ChangeMamba.changedetection.models.ChangeMambaSCD import ChangeMambaSCD
from ChangeMamba.changedetection.utils_func.mcd_utils import SCDD_metrics_from_hist, accuracy, get_hist
from datasets.colormap import NUM_CLASSES, index2color


NUM_SCD_CLASSES = 37


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


def _load_gt_maps(
    gt_cd_path: str | None,
    gt_t1_path: str | None,
    gt_t2_path: str | None,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load GT_CD (0/1), GT_T1, GT_T2 class indices; crop/pad to target H,W."""
    th, tw = target_hw

    def _load_cd(p: str) -> np.ndarray:
        x = np.asarray(imageio.imread(p))
        if x.ndim == 3:
            x = x[..., 0]
        x = (x > 127).astype(np.float32)
        return x

    def _load_sem(p: str) -> np.ndarray:
        x = np.asarray(imageio.imread(p))
        if x.ndim == 3:
            x = x[..., 0]
        return x.astype(np.int32)

    cd = _load_cd(gt_cd_path) if gt_cd_path else None
    t1 = _load_sem(gt_t1_path) if gt_t1_path else None
    t2 = _load_sem(gt_t2_path) if gt_t2_path else None

    for name, arr in (("GT_CD", cd), ("GT_T1", t1), ("GT_T2", t2)):
        if arr is None:
            continue
        if arr.shape[0] != th or arr.shape[1] != tw:
            raise ValueError(
                f"{name} size {arr.shape[:2]} does not match padded image size {(th, tw)}. "
                "Use the same resolution as T1/T2 after padding, or resize GT externally."
            )
    return cd, t1, t2


def _semantic_rgb(pred_cls: np.ndarray, change_mask: np.ndarray) -> np.ndarray:
    """Colorize class map; no-change (mask 0) and class 0 shown as white."""
    rgb = index2color(pred_cls).copy()
    white = np.array([255, 255, 255], dtype=np.uint8)
    rgb[pred_cls == 0] = white
    rgb[change_mask == 0] = white
    return rgb


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
    # Cropped to original image size (h0, w0), same as saved previews — for Evaluate
    pred_change_mask: np.ndarray | None = None
    pred_sem_t1: np.ndarray | None = None
    pred_sem_t2: np.ndarray | None = None


_SESSION = Session()


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
    _SESSION.pred_change_mask = _SESSION.pred_sem_t1 = _SESSION.pred_sem_t2 = None
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
        yield (
            None,
            None,
            None,
            None,
            "Load the model first (Model tab).",
            "",
            "",
        )
        return

    t1_path = (t1_path or "").strip()
    t2_path = (t2_path or "").strip()
    if not t1_path or not t2_path:
        yield (None, None, None, None, "Provide absolute paths to T1 and T2 images.", "", "")
        return

    patch_size = int(patch_size)
    if patch_size <= 0:
        yield (None, None, None, None, "patch_size must be positive.", "", "")
        return
    micro_batch = max(1, int(micro_batch))

    device = _SESSION.device
    model = _SESSION.model

    try:
        pre_img = _load_rgb_f32(t1_path)
        post_img = _load_rgb_f32(t2_path)
    except Exception as e:
        yield (None, None, None, None, f"Failed to read images: {e}", "", "")
        return

    if pre_img.shape[:2] != post_img.shape[:2]:
        yield (
            None,
            None,
            None,
            None,
            f"T1 shape {pre_img.shape[:2]} != T2 shape {post_img.shape[:2]}.",
            "",
            "",
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

    out_root = Path((out_dir or "").strip() or str(_PROJECT_ROOT / "outputs" / "gradio_scd"))
    out_root.mkdir(parents=True, exist_ok=True)
    stem = Path(t1_path).stem
    p_change = out_root / f"{stem}_pred_GT_CD.png"
    p_t1 = out_root / f"{stem}_pred_semantic_T1.png"
    p_t2 = out_root / f"{stem}_pred_semantic_T2.png"
    imageio.imwrite(str(p_change), change_vis)
    imageio.imwrite(str(p_t1), rgb_t1)
    imageio.imwrite(str(p_t2), rgb_t2)

    _SESSION.pred_change_mask = cm.copy()
    _SESSION.pred_sem_t1 = t1_crop.copy()
    _SESSION.pred_sem_t2 = t2_crop.copy()

    status_tail = (
        f"Saved:\n- {p_change}\n- {p_t1}\n- {p_t2}\n\n"
        "Predictions are cached for **Evaluate** (same resolution as T1/T2, unpadded)."
    )

    yield (
        np.stack([change_vis] * 3, axis=-1),
        rgb_t1,
        rgb_t2,
        str(out_root),
        status_tail,
        "",
        "",
    )


def run_evaluation(gt_cd_path: str, gt_t1_path: str, gt_t2_path: str):
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
    try:
        gt_cd, gt_t1, gt_t2 = _load_gt_maps(gt_cd_s, gt_t1_s, gt_t2_s, (h0, w0))
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
            f"Class legend uses **{NUM_CLASSES}** JL1 semantic indices (0–5). "
            "Metrics use the **37-class SCD** encoding from training (`train_MambaSCD` validation). "
            "GT rasters must match **T1/T2 resolution** (unpadded image size)."
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
                change_prev = gr.Image(label="Predicted change map (stitched)", type="numpy")
                sem1_prev = gr.Image(label="Predicted semantic T1 (colored)", type="numpy")
                sem2_prev = gr.Image(label="Predicted semantic T2 (colored)", type="numpy")
                out_path = gr.Textbox(label="Output folder used")
                run_status = gr.Textbox(label="Inference log", lines=6)
                gr.Markdown("### Evaluation (after inference)")
                gt_cd = gr.Textbox(label="GT_CD path", placeholder="…/GT_CD/large.png")
                gt_t1 = gr.Textbox(label="GT_T1 path", placeholder="…/GT_T1/large.png")
                gt_t2 = gr.Textbox(label="GT_T2 path", placeholder="…/GT_T2/large.png")
                eval_btn = gr.Button("Evaluate vs ground truth", variant="primary")
                metrics = gr.Markdown()
                eval_status = gr.Textbox(label="Evaluation log", lines=3)
                run_btn.click(
                    run_tiled_inference,
                    [t1, t2, psz, mb, odir],
                    [change_prev, sem1_prev, sem2_prev, out_path, run_status, metrics, eval_status],
                )
                eval_btn.click(run_evaluation, [gt_cd, gt_t1, gt_t2], [metrics, eval_status])

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
