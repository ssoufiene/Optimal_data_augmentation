import os
import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A

# ----------------------------
# 1. Data corruption config
# ----------------------------
@dataclass
class DataCorruptionConfig:
    corruption_probability: float = 0.1
    corruption_fraction_range: tuple = (0.1, 0.9)

# ----------------------------
# 2. Class mappings
# ----------------------------
final_classes = {
    'Sky': 0, 'Building': 1, 'Vehicle': 2, 'Vegetation': 3,
    'Sign/Pole': 4, 'Ground': 5, 'Other': 6
}

vkitti_rgb2final = {
    (210, 0, 200): 'Ground',
    (90, 200, 255): 'Sky',
    (0, 199, 0): 'Vegetation',
    (90, 240, 0): 'Vegetation',
    (140, 140, 140): 'Building',
    (100, 60, 100): 'Ground',
    (250, 100, 255): 'Sign/Pole',
    (255, 255, 0): 'Sign/Pole',
    (200, 200, 0): 'Sign/Pole',
    (255, 130, 0): 'Sign/Pole',
    (80, 80, 80): 'Other',
    (160, 60, 60): 'Vehicle',
    (255, 127, 80): 'Vehicle',
    (0, 139, 139): 'Vehicle',
    (0, 0, 0): 'Other'
}

kitti_rgb2final = {
    (128, 64, 128): 'Ground',
    (244, 35, 232): 'Ground',
    (70, 70, 70): 'Building',
    (102, 102, 156): 'Building',
    (190, 153, 153): 'Building',
    (153, 153, 153): 'Sign/Pole',
    (250, 170, 30): 'Sign/Pole',
    (220, 220, 0): 'Sign/Pole',
    (107, 142, 35): 'Vegetation',
    (152, 251, 152): 'Vegetation',
    (70, 130, 180): 'Sky',
    (0, 0, 142): 'Vehicle',
    (0, 0, 70): 'Vehicle',
    (0, 60, 100): 'Vehicle',
    (0, 80, 100): 'Vehicle',
    (119, 11, 32): 'Vehicle',
    (81, 0, 81): 'Ground',
    (0, 0, 0): 'Other'
}

# ----------------------------
# 3. Utility functions
# ----------------------------
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

# ----------------------------
# 4. Abstract dataset class
# ----------------------------
class DataRaterDataset(ABC):
    @abstractmethod
    def corrupt_samples(self, samples: torch.Tensor, corruption_fraction: float) -> torch.Tensor:
        pass

    @abstractmethod
    def get_loaders(self, batch_size: int, train_split_ratio: float,
                    train_corruption_config: DataCorruptionConfig) -> tuple:
        pass

# ----------------------------
# 5. Seg2Dataset (vkitti/kitti)
# ----------------------------
class Seg2Dataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, dataset_type='vkitti', size=(512, 256)):
        print(f"[DEBUG] Initializing Seg2Dataset with dataset_type={dataset_type}")
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        if isinstance(mask_dirs, str):
            mask_dirs = [mask_dirs]

        self.size = size
        self.dataset_type = dataset_type.lower()
        self.image_paths, self.mask_paths = [], []

        for img_dir, msk_dir in zip(image_dirs, mask_dirs):
            print(f"[DEBUG] Checking directories: {img_dir}, {msk_dir}")
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            if not os.path.exists(msk_dir):
                raise FileNotFoundError(f"Mask directory not found: {msk_dir}")

            imgs = sorted([p for ext in ('*.jpg','*.png') for p in glob.glob(os.path.join(img_dir, ext))])
            msks = sorted([p for ext in ('*.jpg','*.png') for p in glob.glob(os.path.join(msk_dir, ext))])

            print(f"[DEBUG] Found {len(imgs)} images and {len(msks)} masks in {img_dir}")
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
        print(f"[DEBUG] Getting item {idx}")
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        mask = mask.resize(self.size, Image.NEAREST)
        mask_np = rgb_mask_to_class(np.array(mask), self.rgb_mapping, final_classes)
        mask_t = torch.from_numpy(mask_np).long()

        orig = self.resize(image=img)["image"]

        if self.dataset_type == "vkitti":
            v_orig = orig
            v_blur = self.blur(image=orig)["image"]
            v_cj = self.cj(image=orig)["image"]
            v_rgb = self.rgb(image=orig)["image"]
            variants = [v_orig, v_blur, v_cj, v_rgb]
            variants_t = [torch.from_numpy(v.transpose(2,0,1)).float()/255.0 for v in variants]
            return variants_t, mask_t
        else:
            img_t = torch.from_numpy(orig.transpose(2,0,1)).float()/255.0
            return img_t, mask_t

class MyDataset(DataRaterDataset):
    def __init__(self, image_dirs, mask_dirs):
        print("[DEBUG] Initializing MyDataset")
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs

    def corrupt_samples(self, samples: torch.Tensor, corruption_fraction: float) -> torch.Tensor:
        return samples

    def get_loaders(self, batch_size, train_split_ratio, train_corruption_config):
        print("[DEBUG] Creating train_dataset")
        train_dataset = Seg2Dataset(self.image_dirs, self.mask_dirs, dataset_type="vkitti", size=(512,256))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        print("[DEBUG] Creating val_test dataset")
        val_test = Seg2Dataset("/content/kitti/training/image_2",
                               "/content/kitti/training/semantic",
                               dataset_type="kitti", size=(512,256))
        size = len(val_test)//2
        gen = torch.Generator().manual_seed(42)
        val_split, test_split = random_split(val_test, [size, size], generator=gen)

        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=0)

        print("[DEBUG] DataLoaders created successfully")
        return train_loader, val_loader, test_loader

# ----------------------------
# 7. Loader function
# ----------------------------
def get_dataset_loaders(config):
    print(f"[DEBUG] get_dataset_loaders called with dataset_name={config.dataset_name}")
    image_dirs = ["/content/vkitti_rgb/Scene01/clone/frames/rgb/Camera_0"]
    mask_dirs = ["/content/vkitti_gt/Scene01/clone/frames/classSegmentation/Camera_0/"]

    if config.dataset_name.lower() == "kitti":
        dataset_handler = MyDataset(image_dirs, mask_dirs)
    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported.")

    loaders = dataset_handler.get_loaders(config.batch_size, config.train_split_ratio, DataCorruptionConfig())
    print("[DEBUG] Dataset loaders ready")
    return dataset_handler, loaders
