import os
import glob
import numpy as np
from PIL import Image

final_classes = {
    'Sky': 0,
    'Building': 1,
    'Vehicle': 2,
    'Vegetation': 3,
    'Sign/Pole': 4,
    'Ground': 5,
    'Other': 6
}

vkitti_rgb2final = {
    (210, 0, 200): 'Ground',       # Terrain
    (90, 200, 255): 'Sky',         # Sky
    (0, 199, 0): 'Vegetation',     # Tree
    (90, 240, 0): 'Vegetation',    # Vegetation
    (140, 140, 140): 'Building',   # Building
    (100, 60, 100): 'Ground',      # Road
    (250, 100, 255): 'Sign/Pole',  # GuardRail
    (255, 255, 0): 'Sign/Pole',    # TrafficSign
    (200, 200, 0): 'Sign/Pole',    # TrafficLight
    (255, 130, 0): 'Sign/Pole',    # Pole
    (80, 80, 80): 'Other',         # Misc
    (160, 60, 60): 'Vehicle',      # Truck
    (255, 127, 80): 'Vehicle',     # Car
    (0, 139, 139): 'Vehicle',      # Van
    (0, 0, 0): 'Other'             # Undefined
}

kitti_rgb2final = {
    (128, 64, 128): 'Ground',     # road
    (244, 35, 232): 'Ground',     # sidewalk
    (70, 70, 70): 'Building',     # building
    (102, 102, 156): 'Building',  # wall
    (190, 153, 153): 'Building',  # fence
    (153, 153, 153): 'Sign/Pole', # pole
    (250, 170, 30): 'Sign/Pole',  # traffic light
    (220, 220, 0): 'Sign/Pole',   # traffic sign
    (107, 142, 35): 'Vegetation', # vegetation
    (152, 251, 152): 'Vegetation',# terrain
    (70, 130, 180): 'Sky',        # sky
    (0, 0, 142): 'Vehicle',       # car
    (0, 0, 70): 'Vehicle',        # truck
    (0, 60, 100): 'Vehicle',      # bus
    (0, 80, 100): 'Vehicle',      # train
    (119, 11, 32): 'Vehicle',     # bicycle
    (81, 0, 81): 'Ground',        # ground
    (0, 0, 0): 'Other'            # unlabeled
}

final_classes_colors = {
    0: (90, 200, 255),   # Sky
    1: (140, 140, 140),  # Building
    2: (255, 127, 80),   # Vehicle
    3: (90, 240, 0),     # Vegetation
    4: (255, 255, 0),    # Sign/Pole
    5: (210, 0, 200),    # Ground
    6: (80, 80, 80)      # Other
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
    """Convert integer class mask back to RGB for visualization."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_to_color.items():
        rgb[mask_np == class_id] = color
    return rgb