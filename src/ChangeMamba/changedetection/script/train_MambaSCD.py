import sys
import warnings
from pathlib import Path
_project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_project_root / "src"))

# Suppress noisy third-party warnings (RequestsDependencyWarning, transformers PyTorch check, etc.)
warnings.filterwarnings("ignore", message=".*urllib3.*chardet.*")
warnings.filterwarnings("ignore", message=".*PyTorch.*required.*")
warnings.filterwarnings("ignore", message=".*PyTorch was not found.*")
warnings.filterwarnings("ignore", module="timm.models.layers", category=FutureWarning)

import argparse
import gc
import os
import time

import yaml

# PyTorch multiprocessing "file_system" strategy stores tensor handles under tempfile.gettempdir()
# (= TMPDIR). Default to <repo>/temp (gitignored) instead of /tmp when TMPDIR is unset.
if "TMPDIR" not in os.environ:
    _torch_mp_tmp = _project_root / "temp"
    _torch_mp_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(_torch_mp_tmp.resolve())

import numpy as np

from ChangeMamba.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ChangeMamba.changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader
from ChangeMamba.changedetection.utils_func.metrics import Evaluator
from ChangeMamba.changedetection.models.ChangeMambaSCD import ChangeMambaSCD
import ChangeMamba.changedetection.utils_func.lovasz_loss as L
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

try:
    from prodigyopt import Prodigy
    PRODIGY_AVAILABLE = True
except ImportError:
    Prodigy = None
    PRODIGY_AVAILABLE = False
from ChangeMamba.changedetection.utils_func.mcd_utils import accuracy, AverageMeter, get_hist, SCDD_metrics_from_hist
from ChangeMamba.changedetection.utils_func.swanlab_utils import init_swanlab, log_metrics, finish_swanlab


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.deep_model = ChangeMambaSCD(
            output_cd = 2, 
            output_clf = 7,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
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
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
        self.deep_model = self.deep_model.cuda()
        if getattr(args, 'model_save_path', None):
            self.model_save_path = args.model_save_path.rstrip('/')
        else:
            self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                                args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.start_step = 0
        self.best_kc = 0.0
        self.best_round = []
        self._resume_checkpoint = None
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            # Use same save dir as resume checkpoint
            self.model_save_path = os.path.dirname(args.resume)
            self._resume_checkpoint = torch.load(args.resume, map_location='cpu')
            # Support both formats: full checkpoint dict or legacy model-only state_dict
            if isinstance(self._resume_checkpoint, dict) and 'model' in self._resume_checkpoint:
                self.deep_model.load_state_dict(self._resume_checkpoint['model'])
                self.start_step = self._resume_checkpoint.get('step', 0)
                self.best_kc = self._resume_checkpoint.get('best_kc', 0.0)
                self.best_round = self._resume_checkpoint.get('best_round', [])
            else:
                # Legacy: raw state_dict
                model_dict = {k: v for k, v in self._resume_checkpoint.items() if k in self.deep_model.state_dict()}
                self.deep_model.load_state_dict(model_dict, strict=False)

        # Optimizer: prodigy (parameter-free, lr=1 recommended) or adamw
        total_steps = args.max_iters
        if getattr(args, 'optimizer', 'adamw').lower() == 'prodigy':
            if not PRODIGY_AVAILABLE:
                raise ImportError("Prodigy optimizer requires: pip install prodigyopt")
            self.optim = Prodigy(
                self.deep_model.parameters(),
                lr=args.learning_rate,  # 1.0 recommended for Prodigy
                weight_decay=getattr(args, 'prodigy_weight_decay', args.weight_decay),
                decouple=not getattr(args, 'prodigy_no_decouple', False),  # True=AdamW-style
                d_coef=getattr(args, 'prodigy_d_coef', 1.0),  # >1 larger lr, <1 smaller lr
                slice_p=getattr(args, 'prodigy_slice_p', 1),  # 11 for low memory
                safeguard_warmup=getattr(args, 'prodigy_safeguard_warmup', False),
            )
            self.scheduler = CosineAnnealingLR(self.optim, T_max=total_steps)
        else:
            self.optim = optim.AdamW(self.deep_model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
            self.scheduler = StepLR(self.optim, step_size=10000, gamma=0.5)

        # Restore optimizer and scheduler from resume checkpoint if available
        if self._resume_checkpoint is not None and isinstance(self._resume_checkpoint, dict):
            if 'optimizer' in self._resume_checkpoint:
                try:
                    self.optim.load_state_dict(self._resume_checkpoint['optimizer'])
                except (ValueError, KeyError):
                    pass
            if 'scheduler' in self._resume_checkpoint:
                try:
                    self.scheduler.load_state_dict(self._resume_checkpoint['scheduler'])
                except (ValueError, KeyError):
                    pass
        self._resume_checkpoint = None

        # SwanLab experiment tracking
        swanlab_config = {k: getattr(args, k) for k in vars(args) if not k.startswith('_')}
        init_swanlab(project="ChangeMamba", experiment_name=f"{args.model_type}_{args.dataset}", config=swanlab_config)

    def training(self):
        best_kc = self.best_kc
        best_round = list(self.best_round)
        save_interval = getattr(self.args, 'save_interval', 10000)
        torch.cuda.empty_cache()
        # Epoch loop + explicit iterator (no itertools.cycle on DataLoader): avoids unbounded prefetch / iterator stacking and shrinks worker shm footprint.
        step_idx = self.start_step
        pbar = tqdm(total=self.args.max_iters, initial=self.start_step, desc="train", unit="step")
        try:
            while step_idx < self.args.max_iters:
                for data in self.train_data_loader:
                    if step_idx >= self.args.max_iters:
                        break
                    step = step_idx + 1
                    pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, _ = data

                    pre_change_imgs = pre_change_imgs.cuda()
                    post_change_imgs = post_change_imgs.cuda()
                    label_cd = label_cd.cuda().long()
                    label_clf_t1 = label_clf_t1.cuda().long()
                    label_clf_t2 = label_clf_t2.cuda().long()

                    # Clone before mutating ignore mask: in-place on DataLoader tensors can corrupt shared worker buffers.
                    label_clf_t1_m = label_clf_t1.clone()
                    label_clf_t2_m = label_clf_t2.clone()
                    label_clf_t1_m[label_clf_t1_m == 0] = 255
                    label_clf_t2_m[label_clf_t2_m == 0] = 255

                    output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

                    self.optim.zero_grad(set_to_none=True)

                    ce_loss_cd = F.cross_entropy(output_1, label_cd, ignore_index=255)
                    lovasz_loss_cd = L.lovasz_softmax(F.softmax(output_1, dim=1), label_cd, ignore=255)

                    ce_loss_clf_t1 = F.cross_entropy(output_semantic_t1, label_clf_t1_m, ignore_index=255)
                    lovasz_loss_clf_t1 = L.lovasz_softmax(F.softmax(output_semantic_t1, dim=1), label_clf_t1_m, ignore=255)

                    ce_loss_clf_t2 = F.cross_entropy(output_semantic_t2, label_clf_t2_m, ignore_index=255)
                    lovasz_loss_clf_t2 = L.lovasz_softmax(F.softmax(output_semantic_t2, dim=1), label_clf_t2_m, ignore=255)

                    # Mask for similarity loss (label == 255)
                    similarity_mask = (label_clf_t1_m == 255).float().unsqueeze(1).expand_as(output_semantic_t1)

                    similarity_loss = F.mse_loss(
                        F.softmax(output_semantic_t1, dim=1) * similarity_mask,
                        F.softmax(output_semantic_t2, dim=1) * similarity_mask,
                        reduction='mean',
                    )

                    main_loss = (
                        ce_loss_cd
                        + 0.5 * (ce_loss_clf_t1 + ce_loss_clf_t2 + 0.5 * similarity_loss)
                        + 0.75 * (lovasz_loss_cd + 0.5 * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2))
                    )
                    final_loss = main_loss

                    # Scalars only (no tensor refs) for print / SwanLab after backward frees the graph.
                    log_cd = log_clf = log_total = None
                    if step % 10 == 0:
                        with torch.no_grad():
                            log_cd = float((ce_loss_cd + lovasz_loss_cd).item())
                            log_clf = float(((ce_loss_clf_t1 + ce_loss_clf_t2 + lovasz_loss_clf_t1 + lovasz_loss_clf_t2) / 2).item())
                            log_total = float(final_loss.item())

                    final_loss.backward()

                    self.optim.step()
                    self.scheduler.step()

                    if step % 10 == 0 and log_cd is not None:
                        print(f'iter is {step}, change detection loss is {log_cd}, classification loss is {log_clf}')
                        log_metrics({"train/cd_loss": log_cd, "train/clf_loss": log_clf, "train/loss": log_total}, step=step)

                    if step % save_interval == 0 or step == self.args.max_iters:
                        self.deep_model.eval()
                        kappa_n0, Fscd, IoU_mean, Sek, oa = self.validation()
                        log_metrics({"val/OA": oa, "val/F1": Fscd, "val/mIoU": IoU_mean, "val/SeK": Sek,
                                    "val/kappa": kappa_n0}, step=step)
                        if Sek > best_kc:
                            torch.save(self.deep_model.state_dict(),
                                       os.path.join(self.model_save_path, 'best_model.pth'))
                            best_kc = Sek
                            best_round = [oa, Fscd, IoU_mean, Sek, kappa_n0]
                        ckpt = {
                            'model': self.deep_model.state_dict(),
                            'optimizer': self.optim.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            'step': step,
                            'best_kc': best_kc,
                            'best_round': best_round,
                        }
                        torch.save(ckpt, os.path.join(self.model_save_path, f'{step}_model.pth'))
                        self.deep_model.train()
                        gc.collect()

                    step_idx += 1
                    pbar.update(1)
        finally:
            pbar.close()

        print('The accuracy of the best round (OA, F1, mIoU, SeK, Kappa) is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        eval_size = getattr(self.args, 'eval_crop_size', 256)
        dataset = SemanticChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, eval_size, None, 'test')
        nw = max(0, int(getattr(self.args, "num_workers", 0)))
        val_kw = {"prefetch_factor": 1, "persistent_workers": False} if nw > 0 else {}
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=nw, drop_last=False, **val_kw)
        torch.cuda.empty_cache()
        acc_meter = AverageMeter()
        num_class = 37
        # Streaming confusion matrix only — do not store every full-frame pred (was O(val_images × H × W) RAM per eval).
        hist = np.zeros((num_class, num_class), dtype=np.float64)

        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_cd, labels_clf_t1, labels_clf_t2, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_cd = labels_cd.cuda().long()
                labels_clf_t1 = labels_clf_t1.cuda().long()
                labels_clf_t2 = labels_clf_t2.cuda().long()

                output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

                labels_cd_np = labels_cd.cpu().numpy()
                labels_A = labels_clf_t1.cpu().numpy()
                labels_B = labels_clf_t2.cpu().numpy()

                change_mask = torch.argmax(output_1, axis=1).cpu().numpy()

                preds_A = torch.argmax(output_semantic_t1, dim=1).cpu().numpy()
                preds_B = torch.argmax(output_semantic_t2, dim=1).cpu().numpy()

                preds_scd = (preds_A - 1) * 6 + preds_B
                preds_scd[change_mask == 0] = 0

                labels_scd = (labels_A - 1) * 6 + labels_B
                labels_scd[labels_cd_np == 0] = 0

                for pred_scd, label_scd in zip(preds_scd, labels_scd):
                    acc_A, _ = accuracy(pred_scd, label_scd)
                    hist += get_hist(pred_scd, label_scd, num_class)
                    acc_meter.update(acc_A)

                del output_1, output_semantic_t1, output_semantic_t2

        kappa_n0, Fscd, IoU_mean, Sek = SCDD_metrics_from_hist(hist)
        del hist
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Kappa coefficient rate is {kappa_n0}, F1 is {Fscd}, OA is {acc_meter.avg}, '
              f'mIoU is {IoU_mean}, SeK is {Sek}')

        return kappa_n0, Fscd, IoU_mean, Sek, acc_meter.avg


_TRAINING_PATH_KEYS = frozenset({
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


def _training_yaml_to_arg_defaults(raw, project_root):
    """Turn training YAML dict into argparse defaults; resolve relative paths."""
    if not raw:
        return {}
    flat = dict(raw)
    out = {}
    for k, v in flat.items():
        if v is None:
            continue
        if k == "opts" and v is not None:
            if not isinstance(v, (list, tuple)):
                raise ValueError("training config 'opts' must be a YAML list of KEY VALUE pairs")
            out[k] = [str(x) for x in v]
            continue
        if k in _TRAINING_PATH_KEYS and isinstance(v, str) and v.strip():
            p = Path(v).expanduser()
            if not p.is_absolute():
                p = (project_root / p).resolve()
            out[k] = str(p)
        else:
            out[k] = v
    return out


def _parse_args_with_training_yaml(project_root):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="YAML with training paths and hyperparameters (repo configs/*.yaml).",
    )
    pre_args, argv_rest = pre.parse_known_args()

    yaml_defaults = {}
    train_config_path = pre_args.train_config
    if train_config_path:
        tc_path = Path(train_config_path).expanduser()
        if not tc_path.is_absolute():
            tc_path = (project_root / tc_path).resolve()
        if not tc_path.is_file():
            raise FileNotFoundError(f"train_config not found: {tc_path}")
        with open(tc_path, "r", encoding="utf-8") as f:
            yaml_defaults = _training_yaml_to_arg_defaults(yaml.safe_load(f), project_root)

    _cfg_default = Path(__file__).resolve().parents[1] / "configs" / "vssm1" / "vssm_base_224.yaml"
    parser = argparse.ArgumentParser(description="Training on SECOND dataset")
    parser.add_argument("--train_config", type=str, default=train_config_path, help="Training YAML used to seed defaults")
    parser.add_argument("--cfg", type=str, default=str(_cfg_default))
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )
    parser.add_argument("--pretrained_weight_path", type=str)

    parser.add_argument("--dataset", type=str, default="SECOND")
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--train_dataset_path", type=str, default="/data/ggeoinfo/datasets/xBD/train")
    parser.add_argument("--train_data_list_path", type=str, default="/data/ggeoinfo/datasets/xBD/xBD_list/train_all.txt")
    parser.add_argument("--test_dataset_path", type=str, default="/data/ggeoinfo/datasets/xBD/test")
    parser.add_argument("--test_data_list_path", type=str, default="/data/ggeoinfo/datasets/xBD/xBD_list/val_all.txt")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes (user-defined; 0 = load in main process only). "
        "Higher values improve I/O throughput.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=224, help="Train random crop size (224 for vssm_base_224)")
    parser.add_argument("--eval_crop_size", type=int, default=256, help="Eval/infer size (full image for JL1)")
    parser.add_argument("--train_data_name_list", type=list)
    parser.add_argument("--test_data_name_list", type=list)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--max_iters", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--save_interval", type=int, default=10000, help="Save checkpoint and run val eval every N steps")
    parser.add_argument("--model_type", type=str, default="ChangeMambaSCD")
    parser.add_argument("--model_param_path", type=str, default="../saved_models")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Direct path for checkpoints; overrides model_param_path/dataset/model_type",
    )

    parser.add_argument("--resume", type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--optimizer", type=str, default="prodigy", choices=["adamw", "prodigy"])

    parser.add_argument(
        "--prodigy_weight_decay",
        type=float,
        default=0.01,
        help="Prodigy weight decay (0, 0.001, 0.01, 0.1)",
    )
    parser.add_argument(
        "--prodigy_slice_p",
        type=int,
        default=1,
        help="Prodigy slice_p: 11 for limited memory, 1 default",
    )
    parser.add_argument(
        "--prodigy_d_coef",
        type=float,
        default=1.0,
        help="Prodigy d_coef: >1 larger lr estimate, <1 smaller (default 1.0)",
    )
    parser.add_argument(
        "--prodigy_no_decouple",
        action="store_true",
        help="Prodigy standard L2 (Adam-style). Default: decouple=True (AdamW-style)",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        default=False,
        help="Prodigy safeguard when using warm-up",
    )

    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(argv_rest)
    args.train_config = train_config_path
    return args


def main():
    # file_system: worker tensor handoff uses files under tempfile.gettempdir() (here: repo temp/ via TMPDIR).
    _mp = os.environ.get("TORCH_MP_SHARING_STRATEGY", "file_system").strip().lower()
    if _mp in ("file_descriptor", "fd"):
        torch.multiprocessing.set_sharing_strategy("file_descriptor")
    else:
        torch.multiprocessing.set_sharing_strategy("file_system")

    args = _parse_args_with_training_yaml(_project_root)
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    print(f"DataLoader num_workers={args.num_workers} (train and validation).")

    trainer = Trainer(args)
    trainer.training()
    finish_swanlab()


if __name__ == "__main__":
    main()
