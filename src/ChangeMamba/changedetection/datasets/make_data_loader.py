import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import ChangeMamba.changedetection.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot



class ChangeDetectionDatset(Dataset):
    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        max_iters=None,
        type='train',
        data_loader=img_loader,
        train_augment=True,
    ):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.train_augment = bool(train_augment)

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
        post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
        label_path = os.path.join(self.dataset_path, 'GT', self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type and self.train_augment:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


class CroplandCollectionsBCDDataset(Dataset):
    """
    BCD dataset over collections root with per-source subfolders.

    Supported data_list entries:
      - "source/split/name.png"  (recommended, used by *_all.txt)
      - "source/name.png"        (split inferred from `type`)
      - "name.png"               (fallback to dataset_path directly)
    """

    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        max_iters=None,
        type='train',
        data_loader=img_loader,
        train_augment=True,
    ):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.train_augment = bool(train_augment)

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def _resolve_sample_paths(self, sample_entry):
        e = str(sample_entry).replace("\\", "/")
        parts = [p for p in e.split("/") if p]
        root = self.dataset_path

        if len(parts) >= 3:
            source, split = parts[0], parts[1]
            name = parts[-1]
            base = os.path.join(root, source, split)
        elif len(parts) == 2:
            source, name = parts
            split = self.type
            base_candidate = os.path.join(root, source, split)
            base = base_candidate if os.path.isdir(base_candidate) else os.path.join(root, source)
        else:
            name = parts[0] if parts else e
            base_candidate = os.path.join(root, self.type)
            base = base_candidate if os.path.isdir(base_candidate) else root

        pre_path = os.path.join(base, 'T1', name)
        post_path = os.path.join(base, 'T2', name)
        label_path = os.path.join(base, 'GT', name)
        return pre_path, post_path, label_path

    def __getitem__(self, index):
        pre_path, post_path, label_path = self._resolve_sample_paths(self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type and self.train_augment:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


class SemanticChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            # Data augmentation: horizontal flip only (vertical flip and rotation disabled for JL1 256px)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.ascontiguousarray(np.transpose(pre_img, (2, 0, 1)))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.ascontiguousarray(np.transpose(post_img, (2, 0, 1)))

        cd_label = np.ascontiguousarray(cd_label)
        t1_label = np.ascontiguousarray(t1_label)
        t2_label = np.ascontiguousarray(t2_label)
        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index] + '.png')
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index] + '.png')
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index] + '.png')
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index] + '.png')
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index] + '.png')
        else:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        # Ensure contiguous arrays (avoids PyTorch "negative strides" error in DataLoader collate)
        pre_img = np.ascontiguousarray(pre_img)
        post_img = np.ascontiguousarray(post_img)
        cd_label = np.ascontiguousarray(cd_label)
        t1_label = np.ascontiguousarray(t1_label)
        t2_label = np.ascontiguousarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, t2_label, data_idx

    def __len__(self):
        return len(self.data_list)


class DamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type: 
            parts = self.data_list[index].rsplit('_', 2)

            pre_img_name = f"{parts[0]}_pre_disaster_{parts[1]}_{parts[2]}.png"
            post_img_name = f"{parts[0]}_post_disaster_{parts[1]}_{parts[2]}.png"

            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            
            loc_label_path = os.path.join(self.dataset_path, 'masks', pre_img_name)
            clf_label_path = os.path.join(self.dataset_path, 'masks', post_img_name)
        else:
            pre_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_pre_disaster.png')
            post_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_post_disaster.png')
            loc_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_pre_disaster.png')
            clf_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_post_disaster.png')

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        loc_label = self.loader(loc_label_path)[:,:,0]
        clf_label = self.loader(clf_label_path)[:,:,0]

        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            clf_label[clf_label == 0] = 255
        else:
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)



class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size
        self.scale_list = [0.5, 1, 0.75, 1, 0.9, 1, 1.1, 1, 1.25, 1, 1.5]

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            # scale_factor = choice(self.scale_list)
            # if scale_factor != 1:
            #     img_height, img_width, _ = pre_img.shape
            #     new_height, new_width = scale_factor * img_height, scale_factor * img_width
            #     pre_img = transform.resize(pre_img, (int(new_height), int(new_width)))
            #     post_img = transform.resize(post_img, (int(new_height), int(new_width)))
            #     label = transform.resize(label, (int(new_height), int(new_width)))

            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster'  + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', self.data_list[index] + '_building_damage'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)


def _dataloader_num_workers(args, branch_default):
    """Prefer args.num_workers; else dataset-specific default. Use 0 to load in main process only."""
    v = getattr(args, "num_workers", None)
    if v is None:
        return branch_default
    return max(0, int(v))


def _dataloader_memory_kwargs(num_workers):
    """Limit worker prefetch to reduce RAM/shm use; avoid piling batches when cycling epochs."""
    if num_workers <= 0:
        return {}
    # prefetch_factor=1 (min allowed with num_workers>0) keeps fewer batches alive per worker
    return {"prefetch_factor": 1, "persistent_workers": False}


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    _ta = getattr(args, "train_augment", True)
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU' in args.dataset:
        dataset = ChangeDetectionDatset(
            args.train_dataset_path,
            args.train_data_name_list,
            args.crop_size,
            None,
            args.type,
            train_augment=_ta,
        )
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        nw = _dataloader_num_workers(args, 16)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            **_dataloader_memory_kwargs(nw),
            **kwargs,
            num_workers=nw,
            drop_last=False,
        )
        return data_loader
    elif 'CROPLAND_BCD_COLLECTIONS' in args.dataset:
        dataset = CroplandCollectionsBCDDataset(
            args.train_dataset_path,
            args.train_data_name_list,
            args.crop_size,
            None,
            args.type,
            train_augment=_ta,
        )
        nw = _dataloader_num_workers(args, 8)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            **_dataloader_memory_kwargs(nw),
            **kwargs,
            num_workers=nw,
            drop_last=False,
        )
        return data_loader
    elif 'xBD' in args.dataset:
        dataset = DamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, None, args.type)
        nw = _dataloader_num_workers(args, 6)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            **_dataloader_memory_kwargs(nw),
            **kwargs,
            num_workers=nw,
            drop_last=False,
        )
        return data_loader

    elif 'SECOND' in args.dataset or 'JL1' in args.dataset:
        # max_iters=None: keep only the file list; step count / epoch repeats are handled in the trainer (no huge duplicated lists, no cycle+loader quirks).
        dataset = SemanticChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, None, args.type)
        nw = _dataloader_num_workers(args, 8)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            **_dataloader_memory_kwargs(nw),
            **kwargs,
            num_workers=nw,
            drop_last=False,
        )
        return data_loader

    elif 'BRIGHT' in args.dataset:
        dataset = MultimodalDamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, None, args.type)
        nw = _dataloader_num_workers(args, 4)
        data_loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=args.shuffle,
            **_dataloader_memory_kwargs(nw),
            **kwargs,
            num_workers=nw,
            drop_last=False,
        )
        return data_loader
    
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='WHUBCD')
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())
