import os
import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

import albumentations as A


@dataclass
class DataCorruptionConfig:
    corruption_probability: float = 0.1
    corruption_fraction_range: tuple = (0.1, 0.9)


final_classes = {
    'Sky': 0, 'Building': 1, 'Vehicle': 2, 'Vegetation': 3,
    'Sign/Pole': 4, 'Ground': 5, 'Other': 6
}

vkitti_rgb2final = {
    (210, 0, 200): 'Ground', (90, 200, 255): 'Sky', (0, 199, 0): 'Vegetation',
    (90, 240, 0): 'Vegetation', (140, 140, 140): 'Building', (100, 60, 100): 'Ground',
    (250, 100, 255): 'Sign/Pole', (255, 255, 0): 'Sign/Pole', (200, 200, 0): 'Sign/Pole',
    (255, 130, 0): 'Sign/Pole', (80, 80, 80): 'Other', (160, 60, 60): 'Vehicle',
    (255, 127, 80): 'Vehicle', (0, 139, 139): 'Vehicle', (0, 0, 0): 'Other'
}

kitti_rgb2final = {
    (128, 64, 128): 'Ground', (244, 35, 232): 'Ground', (70, 70, 70): 'Building',
    (102, 102, 156): 'Building', (190, 153, 153): 'Building', (153, 153, 153): 'Sign/Pole',
    (250, 170, 30): 'Sign/Pole', (220, 220, 0): 'Sign/Pole', (107, 142, 35): 'Vegetation',
    (152, 251, 152): 'Vegetation', (70, 130, 180): 'Sky', (0, 0, 142): 'Vehicle',
    (0, 0, 70): 'Vehicle', (0, 60, 100): 'Vehicle', (0, 80, 100): 'Vehicle',
    (119, 11, 32): 'Vehicle', (81, 0, 81): 'Ground', (0, 0, 0): 'Other'
}


def rgb_mask_to_class(mask_img, rgb_mapping, final_classes):
    mask = np.array(mask_img)
    h, w, _ = mask.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, cname in rgb_mapping.items():
        class_id = final_classes[cname]
        matches = np.all(mask == rgb, axis=-1)
        class_mask[matches] = class_id
    return class_mask

def class_to_rgb(mask_np, class_to_color):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_to_color.items():
        rgb[mask_np == class_id] = color
    return rgb

class DataRaterDataset(ABC):
    @abstractmethod
    def corrupt_samples(self, samples: torch.Tensor, corruption_fraction: float) -> torch.Tensor:
        pass

    @abstractmethod
    def get_loaders(self, batch_size: int, train_split_ratio: float,
                    train_corruption_config: DataCorruptionConfig) -> tuple:
        pass



class SegDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, dataset_type='vkitti', aug_case=None, size=(512, 256)):
        """
        image_dirs: str or list of str paths to image folders
        mask_dirs: str or list of str paths to mask folders
        """
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        if isinstance(mask_dirs, str):
            mask_dirs = [mask_dirs]

        self.image_paths = []
        self.mask_paths = []

        for img_dir, msk_dir in zip(image_dirs, mask_dirs):
            img_files = sorted([p for ext in ('*.jpg', '*.png') for p in glob.glob(os.path.join(img_dir, ext))])
            msk_files = sorted([p for ext in ('*.jpg', '*.png') for p in glob.glob(os.path.join(msk_dir, ext))])
            self.image_paths.extend(img_files)
            self.mask_paths.extend(msk_files)

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of images and masks must match across all folders.")

        self.size = size
        self.aug_case = aug_case
        self.transform = self.get_augmentations(aug_case)
        self.dataset_type = dataset_type.lower()

        if self.dataset_type == 'vkitti':
            self.rgb_mapping = vkitti_rgb2final
        elif self.dataset_type == 'kitti':
            self.rgb_mapping = kitti_rgb2final
        else:
            raise ValueError("dataset_type must be 'vkitti' or 'kitti'")

    def __len__(self):
        return len(self.image_paths)

    def get_augmentations(self, case):
        """Photometric augmentations applied only to images."""
        if case is None:
            return A.Resize(*self.size)
        if case == 'CJ':
            return A.Compose([
                A.Resize(*self.size),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            ])
        elif case == 'RGBS':
            return A.Compose([
                A.Resize(*self.size),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            ])
        elif case == 'BLUR':
            return A.Compose([
                A.Resize(*self.size),
                A.GaussianBlur(blur_limit=(3,7), p=0.5),
            ])
        elif case == 'EQUAL':
            return A.Compose([
                A.Resize(*self.size),
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                    A.GaussianBlur(blur_limit=(3,7), p=1.0),
                ], p=1.0),
            ])
        else:
            raise ValueError("Invalid augmentation case. Choose from 'CJ','RGBS','BLUR','EQUAL'")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = Image.open(mask_path).convert("RGB")
        mask = mask.resize((self.size[1],self.size[0]), Image.NEAREST)
        mask_np = np.array(mask)
        mask_np = rgb_mask_to_class(mask_np, self.rgb_mapping, final_classes)

        if self.transform:
            img_aug = self.transform(image=img)['image']
        else:
            img_aug = img

        img_t = torch.from_numpy(img_aug.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask_np).long()

        return img_t, mask_t

class Seg2Dataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, dataset_type='vkitti', size=(512, 256)):
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        if isinstance(mask_dirs, str):
            mask_dirs = [mask_dirs]

        self.size = size
        self.dataset_type = dataset_type.lower()
        self.image_paths, self.mask_paths = [], []

        for img_dir, msk_dir in zip(image_dirs, mask_dirs):
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            if not os.path.exists(msk_dir):
                raise FileNotFoundError(f"Mask directory not found: {msk_dir}")

            imgs = sorted([p for ext in ('*.jpg','*.png') for p in glob.glob(os.path.join(img_dir, ext))])
            msks = sorted([p for ext in ('*.jpg','*.png') for p in glob.glob(os.path.join(msk_dir, ext))])

            if len(imgs) == 0 or len(msks) == 0:
                raise ValueError("No image or mask files found.")

            self.image_paths.extend(imgs)
            self.mask_paths.extend(msks)

        assert len(self.image_paths) == len(self.mask_paths), "Mismatch image/mask count"

        self.rgb_mapping = vkitti_rgb2final if self.dataset_type == "vkitti" else kitti_rgb2final
        self.resize = A.Resize(height=size[1], width=size[0])
        self.blur = A.GaussianBlur(blur_limit=(3,7), p=1.0)
        self.cj = A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)
        self.rgb = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
      img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
      mask = Image.open(self.mask_paths[idx]).convert("RGB")
      mask = mask.resize(self.size, Image.NEAREST)
      mask_np = rgb_mask_to_class(np.array(mask), self.rgb_mapping, final_classes)
      mask_t = torch.from_numpy(mask_np).long()

      orig = self.resize(image=img)["image"]

      if self.dataset_type == "vkitti":
          variants = [
              orig,
              self.blur(image=orig)["image"],
              self.cj(image=orig)["image"],
              self.rgb(image=orig)["image"]
          ]
          probs = [0.25, 0.25, 0.25, 0.25]  
          aug_idx = random.choices(range(len(variants)), weights=probs, k=1)[0]
          chosen = variants[aug_idx]
          chosen_t = torch.from_numpy(chosen.transpose(2,0,1)).float()/255.0

          # return the augmentation index along with image and mask
          return chosen_t, mask_t, aug_idx

      else:
          img_t = torch.from_numpy(orig.transpose(2,0,1)).float()/255.0
          return img_t, mask_t




class MyDataset(DataRaterDataset):
    def __init__(self, image_dirs, mask_dirs):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs

    def corrupt_samples(self, samples: torch.Tensor, corruption_fraction: float) -> torch.Tensor:
        return samples

    def get_loaders(self, batch_size, train_split_ratio, train_corruption_config):
        train_dataset = Seg2Dataset(self.image_dirs, self.mask_dirs, dataset_type="vkitti")
        print(f"[INFO] Loaded {len(train_dataset)} training images.")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_test = Seg2Dataset("/content/kitti/training/image_2",
                               "/content/kitti/training/semantic",
                               dataset_type="kitti", size=(512,256))
        size = len(val_test)//2
        gen = torch.Generator().manual_seed(42)
        val_split, test_split = random_split(val_test, [size, size], generator=gen)

        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader


def get_dataset_loaders(config):
    image_dirs = [
    "/content/vkitti_rgb/Scene01/clone/frames/rgb/Camera_0",
    "/content/vkitti_rgb/Scene02/clone/frames/rgb/Camera_0",
    "/content/vkitti_rgb/Scene06/clone/frames/rgb/Camera_0",
    "/content/vkitti_rgb/Scene18/clone/frames/rgb/Camera_0",
    "/content/vkitti_rgb/Scene20/clone/frames/rgb/Camera_0",
]

    mask_dirs = [
        "/content/vkitti_gt/Scene01/clone/frames/classSegmentation/Camera_0/",
        "/content/vkitti_gt/Scene02/clone/frames/classSegmentation/Camera_0/",
        "/content/vkitti_gt/Scene06/clone/frames/classSegmentation/Camera_0/",
        "/content/vkitti_gt/Scene18/clone/frames/classSegmentation/Camera_0/",
        "/content/vkitti_gt/Scene20/clone/frames/classSegmentation/Camera_0/",
    ]

    if config.dataset_name.lower() == "kitti":
        dataset_handler = MyDataset(image_dirs, mask_dirs)
    
    loaders = dataset_handler.get_loaders(config.batch_size, config.train_split_ratio, DataCorruptionConfig())
    return dataset_handler, loaders
