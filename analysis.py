import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from scipy import stats



def analyze_with_individual_augmentations(data_rater, data_loader, current_step=None, device='cuda', save_dir=None):
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
            single_image = batch_images[img_idx:img_idx+1]  # [1, 3, H, W]
            
            img_np = (single_image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            scores_for_image = {}
            
            for aug_name, aug_fn in augmentations.items():
          
                aug_img = aug_fn(img_np)
                
             
                aug_tensor = torch.from_numpy(aug_img.transpose(2, 0, 1)).float() / 255.0
                aug_batch = aug_tensor.unsqueeze(0).to(device)  
                
          
                with torch.no_grad():
                    score = data_rater(aug_batch).squeeze(-1).item()  
                
                all_aug_scores[aug_name].append(score)
                aug_indices[aug_name].append(image_counter)
                scores_for_image[aug_name] = score
            
            best_aug = max(scores_for_image, key=scores_for_image.get)
            best_augmentations[image_counter] = best_aug
            
            image_counter += 1
    
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
    
    print("\nAugmentation Score Statistics:")
    print("-" * 60)
    
    stats_dict = {}
    for aug_name in augmentations.keys():
        scores = np.array(all_aug_scores[aug_name])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        stats_dict[aug_name] = {"mean": mean_score, "std": std_score}
        
        print(f"{aug_name:15s}: mean={mean_score:7.4f}, std={std_score:6.4f}, n={len(scores)}")
    
    # Ranking
    print("\n" + "-" * 60)
    print("Augmentation Ranking (by mean score):")
    ranking = sorted(stats_dict.items(), key=lambda x: x[1]["mean"], reverse=True)
    for i, (aug_name, s) in enumerate(ranking, 1):
        print(f"  {i}. {aug_name:15s}: {s['mean']:7.4f}")
    
    print("-" * 60)
    
    # Count best augmentations
    print("\nBest Augmentation Distribution (TOP 1):")
    print("-" * 60)
    aug_counts = {}
    for aug in best_augmentations.values():
        aug_counts[aug] = aug_counts.get(aug, 0) + 1
    
    total_images = len(best_augmentations)
    for aug_name in sorted(aug_counts.keys()):
        count = aug_counts[aug_name]
        percentage = (count / total_images) * 100
        print(f"  {aug_name:15s}: {count:5d} images ({percentage:5.1f}%)")
    
    print("-" * 60 + "\n")
    
    # Save augmentation choices if save_dir is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best_augmentations_step_{current_step}.pt")
        torch.save(best_augmentations, save_path)
        print(f"✓ Saved best augmentations to: {save_path}\n")
        
        # Also save statistics
        stats_path = os.path.join(save_dir, f"aug_statistics_step_{current_step}.pt")
        torch.save(stats_dict, stats_path)
        print(f"✓ Saved statistics to: {stats_path}\n")
    
    return all_aug_scores, best_augmentations, stats_dict
    

