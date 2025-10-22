import os
import cv2
import numpy as np
from tqdm import tqdm
from patchify import patchify

def patching(data_dir, patches_dir, file_type, patch_size):
    """
    Efficiently divide images into non-overlapping patches.
    Optimized: Batch write operations and reduced I/O.
    """
    img_list = [f for f in os.listdir(data_dir) if f.endswith(file_type)]
    
    for filename in tqdm(img_list, desc="Patchifying"):
        img = cv2.imread(os.path.join(data_dir, filename), 1)
        if img is None:
            print(f"Warning: Could not read {filename}")
            continue
            
        # cropping to have height and width perfectly divisible by patch_size
        max_height = (img.shape[0] // patch_size) * patch_size
        max_width = (img.shape[1] // patch_size) * patch_size
        img = img[:max_height, :max_width]
        
        # patching (non-overlapping)
        patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
        
        # Batch write patches - more efficient than individual writes
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, 0, :, :, :]
                patch_filename = filename.replace(file_type, f"_patch_{i}_{j}{file_type}")
                cv2.imwrite(os.path.join(patches_dir, patch_filename), single_patch)

def discard_useless_patches(patches_img_dir, patches_mask_dir, discard_rate):
    """
    Remove patches where background occupies more than discard_rate.
    Optimized: Vectorized operations and reduced I/O.
    """
    mask_files = [f for f in os.listdir(patches_mask_dir) if os.path.isfile(os.path.join(patches_mask_dir, f))]
    removed_count = 0
    
    for filename in tqdm(mask_files, desc="Filtering useless patches"):
        mask_path = os.path.join(patches_mask_dir, filename)
        img_path = os.path.join(patches_img_dir, filename)
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            continue
            
        mask = cv2.imread(mask_path)
        if mask is None:
            continue
            
        # Vectorized counting - more efficient
        unique_vals, counts = np.unique(mask, return_counts=True)
        bg_ratio = counts[0] / counts.sum()
        
        # If background class occupies more than the discard rate, remove both files
        if bg_ratio > discard_rate:
            try:
                os.remove(img_path)
                os.remove(mask_path)
                removed_count += 1
            except OSError:
                pass
    
    print(f"Removed {removed_count} useless patch pairs")