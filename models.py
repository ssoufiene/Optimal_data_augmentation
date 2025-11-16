import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
import numpy as np 



class DataRater(nn.Module):
    """The outer model (meta-learner) that learns to rate data quality."""

    def __init__(self, temperature=1.0):
        super(DataRater, self).__init__()
        self.temperature = temperature
        
        # Progressive downsampling with more capacity
        self.layers = nn.Sequential(
            # Input: [B, 3, 512, 256]
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # [B, 32, 256, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 32, 128, 64]
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # [B, 64, 64, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, 32, 16]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 16, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 128, 8, 4]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 8, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),  # [B, 256, 1, 1]
            nn.Flatten()  # [B, 256]
        )
        
        # MLP head with more capacity
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.layers(x)
        return self.head(features).squeeze(-1)



class SimpleSegNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(SimpleSegNet, self).__init__()

        resnet = models.resnet18(pretrained=pretrained)
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up4(e4) + e3
        d3 = self.up3(d4) + e2
        d2 = self.up2(d3) + e1
        d1 = self.up1(d2) + e0

        out = self.out_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

def train_model(model, train_loader, val_loader, num_epochs=15, lr=1e-3, device='cuda'):
    model = model.to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_classes=7
    best_val_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = ce_loss(outputs, masks) + 0.5
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        iou_per_class = torch.zeros(num_classes, device=device)
        eps = 1e-6

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = ce_loss(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                for cls in range(1, num_classes):  
                    pred_cls = preds == cls
                    mask_cls = masks == cls
                    intersection = torch.logical_and(pred_cls, mask_cls).sum().float()
                    union = torch.logical_or(pred_cls, mask_cls).sum().float()
                    iou_per_class[cls] += intersection / (union + eps)

        val_loss /= len(val_loader)
        mean_iou = (iou_per_class[1:] / len(val_loader)).mean().item()

        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val mIoU={mean_iou:.4f}")

    print("Training complete.")



def load_data_rater_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load DataRater model using the existing construct_model function.
    """
    model = construct_model('DataRater')
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(state_dict)
    
    model.eval()
    model = model.to(device)
    
    return model

def train_k_top(model, train_loader, val_loader, best_augs_dict, 
                num_epochs=15, lr=1e-3, device='cuda'):
    model = model.to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_classes = 7
    best_val_iou = 0.0
    
    # Define augmentations
    blur = A.GaussianBlur(blur_limit=(3, 7), p=1.0)
    cj = A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)
    rgb = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)
    
    aug_dict = {
        "original": lambda x: x,
        "blur": lambda x: blur(image=x)["image"],
        "colorjitter": lambda x: cj(image=x)["image"],
        "rgbshift": lambda x: rgb(image=x)["image"]
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            imgs, masks = imgs.to(device), masks.to(device)
            
            augmented_imgs = torch.zeros_like(imgs)
            for i in range(imgs.size(0)):
                # Get the image index in the dataset
                image_idx = batch_idx * train_loader.batch_size + i
                
                # Get the best augmentation for this image
                aug_name = best_augs_dict.get(image_idx, "original")
                
                img_np = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                aug_fn = aug_dict[aug_name]
                aug_img = aug_fn(img_np)
                aug_tensor = torch.from_numpy(aug_img.transpose(2, 0, 1)).float() / 255.0
                expected_shape = (3, 256, 512)
                if aug_tensor.shape != expected_shape:
                    raise ValueError(
                        f"Augmented image has wrong shape {aug_tensor.shape}, expected {expected_shape} "
                        f"(index={i}, augmentation={aug_name})"
    )
                augmented_imgs[i] = aug_tensor
            
            augmented_imgs = augmented_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(augmented_imgs)
            loss = ce_loss(outputs, masks) + 0.5
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        iou_per_class = torch.zeros(num_classes, device=device)
        eps = 1e-6

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = ce_loss(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                for cls in range(1, num_classes):
                    pred_cls = preds == cls
                    mask_cls = masks == cls
                    intersection = torch.logical_and(pred_cls, mask_cls).sum().float()
                    union = torch.logical_or(pred_cls, mask_cls).sum().float()
                    iou_per_class[cls] += intersection / (union + eps)

        val_loss /= len(val_loader)
        mean_iou = (iou_per_class[1:] / len(val_loader)).mean().item()

        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val mIoU={mean_iou:.4f}")

    print("Training complete.")




def evaluate_segmentation(model, dataloader, num_classes=7, device='cuda', eps=1e-6):
    """
    Evaluate a segmentation model on a given dataloader.

    Args:
        model: PyTorch segmentation model
        dataloader: DataLoader providing (images, masks)
        num_classes: number of segmentation classes
        device: 'cuda' or 'cpu'
        eps: small epsilon to avoid division by zero

    Returns:
        iou_per_class: Tensor of IoU per class
        mean_iou: Mean IoU across classes
    """
    model.to(device)
    model.eval()
    iou_per_class = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(num_classes):
                pred_cls = preds == cls
                mask_cls = masks == cls

                intersection = torch.logical_and(pred_cls, mask_cls).sum().float()
                union = torch.logical_or(pred_cls, mask_cls).sum().float()
                iou_per_class[cls] += intersection / (union + eps)

    # Average IoU per class
    iou_per_class /= len(dataloader)
    mean_iou = iou_per_class.mean().item()

    return iou_per_class, mean_iou

def construct_model(model_class):
    if model_class == 'unet':
      return SimpleSegNet()
    
    elif model_class == 'DataRater':
        return DataRater()

    else:
        raise ValueError(f"Model {model_class} not found")
