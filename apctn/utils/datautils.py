import collections
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def create_image_label(image_list):
    """Read a split file containing '<relative_image_path> <class_id>' per line."""
    with open(image_list, "r", encoding="utf-8") as f:
        records = [line.strip() for line in f if line.strip()]
    image_index = [x.rsplit(" ", 1)[0] for x in records]
    label_list = np.array([int(x.rsplit(" ", 1)[1]) for x in records], dtype=np.int64)
    return image_index, label_list


def get_class_map(image_list):
    class_map = {}
    with open(image_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(" ", 1)
            key = int(label)
            if key not in class_map:
                parent = os.path.basename(os.path.dirname(path.replace("\\", "/")))
                class_map[key] = parent or str(key)
    return collections.OrderedDict(sorted(class_map.items()))


def get_class_num(image_list):
    class_map = get_class_map(image_list)
    if not class_map:
        raise ValueError("Empty split file: {}".format(image_list))
    return max(class_map.keys()) + 1


def describe_image_list(image_list):
    _, label_list = create_image_label(image_list)
    label_cnt = np.bincount(label_list)
    print(
        'Image list "{}":\n'
        '\tTotal instances: {}\n'
        '\tTotal class: {}\n'
        '\tmax # of class: {}\n'
        '\tmin # of class: {}\n'
        '\tmean # of class: {:.2f}\n'
        '\tmedian # of class: {:.2f}\n'
        '\tvar: {:.2f}'.format(
            image_list,
            len(label_list),
            len(label_cnt),
            np.max(label_cnt),
            np.min(label_cnt),
            np.mean(label_cnt),
            np.median(label_cnt),
            np.var(label_cnt),
        )
    )


def get_fewshot_index(lbd_dataset, whl_dataset):
    lbd_imgs = lbd_dataset.imgs
    whl_imgs = whl_dataset.imgs
    fewshot_indices = [whl_imgs.index(path) for path in lbd_imgs]
    fewshot_labels = lbd_dataset.labels
    return fewshot_indices, fewshot_labels


class Imagelists(torch.utils.data.Dataset):
    def __init__(
        self,
        image_list,
        root,
        transform=None,
        target_transform=None,
        keep_in_mem=False,
        ret_index=False,
    ):
        imgs, labels = create_image_label(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.root = os.path.abspath(root)
        self.ret_index = ret_index
        self.keep_in_mem = keep_in_mem
        self.loader = pil_loader

        if self.keep_in_mem:
            self.images = [self._load_image(i) for i in range(len(self.imgs))]

    def _load_image(self, index):
        rel = self.imgs[index].replace("/", os.sep).replace("\\", os.sep)
        path = os.path.join(self.root, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Image listed in split file was not found: {}".format(path)
            )
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        img = self.images[index] if self.keep_in_mem else self._load_image(index)
        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.ret_index:
            return index, img, target
        return img, target

    def __len__(self):
        return len(self.imgs)


means = {"imagenet": [0.485, 0.456, 0.406]}
stds = {"imagenet": [0.229, 0.224, 0.225]}


def get_augmentation(trans_type="aug_0", image_size=224, stat="imagenet"):
    mean, std = means["imagenet"], stds["imagenet"]
    image_s = image_size + 32
    data_transforms = {
        "raw": transforms.Compose(
            [
                transforms.Resize((image_s, image_s)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "aug_0": transforms.Compose(
            [
                transforms.Resize((image_s, image_s)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "aug_1": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }
    if trans_type not in data_transforms:
        raise ValueError("Unknown augmentation: {}".format(trans_type))
    return data_transforms[trans_type]


def split_file_path(split_root, name, txt):
    return os.path.join(split_root, name, txt + ".txt")


def dataset_root(data_root, name):
    return os.path.join(data_root, name)


def create_dataset(
    name,
    domain,
    txt="",
    suffix="",
    keep_in_mem=False,
    ret_index=False,
    image_transform=None,
    use_mean_std=False,
    image_size=224,
    data_root="./data",
    split_root="./data/splits",
):
    if suffix:
        suffix = "_" + suffix
    if not txt:
        txt = "{}{}".format(domain, suffix)

    transform = image_transform
    if image_transform is not None and isinstance(image_transform, str):
        transform = get_augmentation(image_transform, image_size=image_size)

    image_list = split_file_path(split_root, name, txt)
    root = dataset_root(data_root, name)
    if not os.path.isfile(image_list):
        raise FileNotFoundError(
            "Split file not found: {}. See README.md for the required split-file layout.".format(
                image_list
            )
        )

    return Imagelists(
        image_list,
        root,
        keep_in_mem=keep_in_mem,
        ret_index=ret_index,
        transform=transform,
    )


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def worker_init_seed(worker_id):
    np.random.seed(12 + worker_id)
    random.seed(12 + worker_id)


def create_loader(dataset, batch_size=32, num_workers=4, is_train=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        num_workers=num_workers,
        shuffle=is_train,
        drop_last=is_train,
        pin_memory=True,
        worker_init_fn=worker_init_seed,
    )
