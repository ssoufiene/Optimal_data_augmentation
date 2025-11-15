import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from scipy import stats
import os 
import matplotlib.pyplot as plt 


import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import os

def analyze_with_individual_augmentations(
    data_rater, data_loader, current_step=None, device='cuda', save_dir=None
):
    """
    For each image in the dataset, generate all 4 augmentation variants individually
    and score each one. Creates a scatter plot and optionally saves the best augmentation for each image.

    Args:
        data_rater: The DataRater model
        data_loader: DataLoader with images
        current_step: Meta-step for logging
        device: Device to use
        save_dir: Directory to save augmentation choices (if None, doesn't save)

    Returns:
        all_aug_scores: Dictionary with scores for each augmentation
        best_augmentations: Dictionary mapping image_idx to best augmentation name
    """
    print(f"\n--- Individual Augmentation Analysis (MetaStep: {current_step}) ---")
    data_rater.eval()

    # Albumentations augmentations
    blur = A.GaussianBlur(blur_limit=(3, 7), p=1.0)
    cj = A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)
    rgb = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)

    augmentations = {
        "original": lambda x: x,
        "blur": lambda x: blur(image=x)["image"],
        "colorjitter": lambda x: cj(image=x)["image"],
        "rgbshift": lambda x: rgb(image=x)["image"]
    }

    all_aug_scores = {aug_name: [] for aug_name in augmentations.keys()}
    aug_indices = {aug_name: [] for aug_name in augmentations.keys()}
    best_augmentations = {}

    image_counter = 0

    for batch_idx, (batch_images, _) in enumerate(data_loader):
        batch_images = batch_images.to(device)
        batch_size = batch_images.size(0)

        for img_idx in range(batch_size):
            single_image = batch_images[img_idx]  # [3, H, W]
            img_np = (single_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # HWC for Albumentations

            scores_for_image = {}

            for aug_name, aug_fn in augmentations.items():
                aug_img = aug_fn(img_np)  # HWC
                aug_tensor = torch.from_numpy(aug_img.transpose(2, 0, 1)).float() / 255.0  # CHW
                aug_batch = aug_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

                # Shape check
                expected_shape = (1, 3, 256, 512)
                if aug_batch.shape != expected_shape:
                    raise ValueError(
                        f"Expected input shape {expected_shape} for DataRater, but got {aug_batch.shape} "
                        f"(image_idx={image_counter}, augmentation={aug_name})"
                    )

                with torch.no_grad():
                    score = data_rater(aug_batch).squeeze(-1).item()

                all_aug_scores[aug_name].append(score)
                aug_indices[aug_name].append(image_counter)
                scores_for_image[aug_name] = score

            best_aug = max(scores_for_image, key=scores_for_image.get)
            best_augmentations[image_counter] = best_aug
            image_counter += 1

    # Scatter plot
    plt.figure(figsize=(14, 8))
    colors = {'original': 'blue', 'blur': 'red', 'colorjitter': 'green', 'rgbshift': 'orange'}

    for aug_name in augmentations.keys():
        scores = all_aug_scores[aug_name]
        indices = aug_indices[aug_name]
        jittered_indices = np.array(indices) + np.random.normal(0, 0.1, len(indices))
        plt.scatter(jittered_indices, scores, alpha=0.5, label=aug_name,
                    color=colors[aug_name], s=30)

    plt.xlabel('Image Index', fontsize=12)
    plt.ylabel('DataRater Score', fontsize=12)
    plt.title(f'Individual Augmentation Scores per Image (MetaStep {current_step})', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Stats summary
    print("\nAugmentation Score Statistics:")
    print("-" * 60)
    stats_dict = {}
    for aug_name in augmentations.keys():
        scores = np.array(all_aug_scores[aug_name])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        stats_dict[aug_name] = {"mean": mean_score, "std": std_score}
        print(f"{aug_name:15s}: mean={mean_score:7.4f}, std={std_score:6.4f}, n={len(scores)}")

    print("\n" + "-" * 60)
    print("Augmentation Ranking (by mean score):")
    ranking = sorted(stats_dict.items(), key=lambda x: x[1]["mean"], reverse=True)
    for i, (aug_name, s) in enumerate(ranking, 1):
        print(f"  {i}. {aug_name:15s}: {s['mean']:7.4f}")
    print("-" * 60)

    # Save best augmentations
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best_augmentations.pt")
        torch.save(best_augmentations, save_path)
        print(f"✓ Saved best augmentations to: {save_path}\n")


        
def visualize_samples_with_scores(
    data_rater, data_loader, num_samples=4, current_step=None, device='cuda'
):
    print(f"\n--- Visualizing {num_samples} Samples (MetaStep={current_step}) ---")
    data_rater.eval()

    # Albumentations
    blur = A.GaussianBlur(blur_limit=(3, 7), p=1.0)
    cj = A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)
    rgb = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)

    augmentations = {
        "Original": lambda x: x,
        "Blur": lambda x: blur(image=x)["image"],
        "ColorJitter": lambda x: cj(image=x)["image"],
        "RGBShift": lambda x: rgb(image=x)["image"],
    }
    aug_order = list(augmentations.keys())

    # Collect all images from all batches
    images = []
    for img_batch, _ in data_loader:
        for i in range(img_batch.size(0)):
            images.append(img_batch[i:i+1].to(device))
    
    # Select random samples from entire dataset
    idxs = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    selected_imgs = [images[i] for i in idxs]

    samples = []

    for img in selected_imgs:
        # img shape: (1, 3, 256, 512)
        # Transpose to (1, 3, 512, 256) for landscape
        img = img.permute(0, 1, 3, 2)  # Swap last two dims
        img_np = (img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        # img_np shape: (512, 256, 3)
        
        sample_dict = {"images": {}, "scores": {}}

        for aug_name in aug_order:
            aug_img = augmentations[aug_name](img_np)
            sample_dict["images"][aug_name] = aug_img

            aug_tensor = torch.tensor(aug_img.transpose(2, 0, 1)).float() / 255.
            aug_tensor = aug_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                score = data_rater(aug_tensor).item()

            sample_dict["scores"][aug_name] = score

        samples.append(sample_dict)

    rows = len(samples)
    cols = len(aug_order)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4.5, rows * 2.2),
        dpi=150
    )
    axes = np.atleast_2d(axes)

    plt.subplots_adjust(
        left=0.01, right=0.99,
        top=0.92, bottom=0.05,
        wspace=0.02, hspace=0.12
    )

    fig.suptitle(
        f"DataRater Visualization (MetaStep {current_step})",
        fontsize=15, fontweight="bold"
    )

    for r in range(rows):
        for c, aug_name in enumerate(aug_order):
            ax = axes[r][c]
            img = samples[r]["images"][aug_name]
            score = samples[r]["scores"][aug_name]

            ax.imshow(img)
            ax.axis('off')
            ax.set_aspect('auto')

            ax.set_title(
                f"{aug_name}  |  {score:.4f}",
                fontsize=10,
                pad=4
            )

    plt.show()