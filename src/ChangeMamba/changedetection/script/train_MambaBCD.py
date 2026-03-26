import sys
import warnings
from pathlib import Path
_project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_project_root / "src"))

warnings.filterwarnings("ignore", message=".*urllib3.*chardet.*")
warnings.filterwarnings("ignore", message=".*PyTorch.*required.*")
warnings.filterwarnings("ignore", message=".*PyTorch was not found.*")
warnings.filterwarnings("ignore", module="timm.models.layers", category=FutureWarning)

import argparse
import os
import time

import yaml

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
from ChangeMamba.changedetection.datasets.make_data_loader import (
    ChangeDetectionDatset,
    CroplandCollectionsBCDDataset,
    make_data_loader,
)
from ChangeMamba.changedetection.utils_func.metrics import Evaluator
from ChangeMamba.changedetection.models.ChangeMambaBCD import ChangeMambaBCD

import ChangeMamba.changedetection.utils_func.lovasz_loss as L
from ChangeMamba.changedetection.utils_func.swanlab_utils import init_swanlab, log_metrics, finish_swanlab

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator = Evaluator(num_class=2)

        self.deep_model = ChangeMambaBCD(
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
        if getattr(args, "model_save_path", None):
            self.model_save_path = args.model_save_path.rstrip("/")
        else:
            self.model_save_path = os.path.join(
                args.model_param_path, args.dataset, args.model_type + "_" + str(time.time())
            )
        self.lr = args.learning_rate

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

        # SwanLab experiment tracking
        swanlab_config = {k: getattr(args, k) for k in vars(args) if not k.startswith('_')}
        init_swanlab(project="ChangeMamba", experiment_name=f"{args.model_type}_{args.dataset}", config=swanlab_config)

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        max_iters = int(self.args.max_iters)
        loader = self.train_data_loader
        train_it = iter(loader)
        for itera in tqdm(range(max_iters)):
            try:
                data = next(train_it)
            except StopIteration:
                train_it = iter(loader)
                data = next(train_it)
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()
            ce_loss_1 = F.cross_entropy(output_1, labels, ignore_index=255)
            lovasz_loss = L.lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=255)
            main_loss = ce_loss_1 + 0.75 * lovasz_loss
            final_loss = main_loss

            final_loss.backward()
            self.optim.step()
            step = itera + 1
            if step % 10 == 0:
                print(f'iter is {step}, overall loss is {final_loss}')
                log_metrics({"train/loss": float(final_loss.item())}, step=step)
                if step % 500 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    log_metrics({"val/recall": rec, "val/precision": pre, "val/oa": oa, "val/f1_score": f1_score, "val/iou": iou, "val/kappa": kc}, step=step)
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{step}_model.pth'))
                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        torch.save(
            self.deep_model.state_dict(),
            os.path.join(self.model_save_path, "last_model.pth"),
        )
        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        val_crop = int(getattr(self.args, "crop_size", 256))
        if "CROPLAND_BCD_COLLECTIONS" in self.args.dataset:
            dataset = CroplandCollectionsBCDDataset(
                self.args.test_dataset_path,
                self.args.test_data_name_list,
                val_crop,
                None,
                'test',
            )
        else:
            dataset = ChangeDetectionDatset(
                self.args.test_dataset_path,
                self.args.test_data_name_list,
                val_crop,
                None,
                'test',
            )
        nw_val = getattr(self.args, "num_workers", None)
        if nw_val is None:
            nw_val = 4
        else:
            nw_val = max(0, int(nw_val))
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=nw_val, drop_last=False)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()

                self.evaluator.add_batch(labels, output_1)
                
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc


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
    parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
    parser.add_argument("--train_config", type=str, default=train_config_path, help="Training YAML used to seed defaults")
    parser.add_argument("--cfg", type=str, default=str(_cfg_default))
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )
    parser.add_argument("--pretrained_weight_path", type=str)
    parser.add_argument("--dataset", type=str, default="SYSU")
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--train_dataset_path", type=str, default="/home/songjian/project/datasets/SYSU/train")
    parser.add_argument("--train_data_list_path", type=str, default="/home/songjian/project/datasets/SYSU/train_list.txt")
    parser.add_argument("--test_dataset_path", type=str, default="/home/songjian/project/datasets/SYSU/test")
    parser.add_argument("--test_data_list_path", type=str, default="/home/songjian/project/datasets/SYSU/test_list.txt")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker processes; omit for dataset-specific default.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument(
        "--train_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If disabled (--no-train_augment), skip random crop / flip / rot on training pairs.",
    )
    parser.add_argument("--train_data_name_list", type=list)
    parser.add_argument("--test_data_name_list", type=list)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--max_iters", type=int, default=240000)
    parser.add_argument("--model_type", type=str, default="ChangeMambaBCD")
    parser.add_argument("--model_param_path", type=str, default="../saved_models")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Direct path for checkpoints; overrides model_param_path/dataset/model_type timestamp dir",
    )

    parser.add_argument("--resume", type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(argv_rest)
    args.train_config = train_config_path
    return args


def main():
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

    _nw = getattr(args, "num_workers", None)
    print(f"DataLoader num_workers={_nw if _nw is not None else 'dataset default'} (train and validation).")

    trainer = Trainer(args)
    trainer.training()
    finish_swanlab()


if __name__ == "__main__":
    main()
