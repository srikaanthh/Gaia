import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):

    """
    Optimized landcover.ai dataset loader with efficient mask processing.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        all_classes (list): all available classes
        classes (list): classes to extract from segmentation mask
        augmentation (albumentations.Compose): data augmentation pipeline
        preprocessing (albumentations.Compose): data preprocessing pipeline
    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            all_classes,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = sorted(os.listdir(images_dir))  # Sort for consistency
        self.images = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Pre-compute class values to avoid repeated lookups during training
        self.class_values = np.array([all_classes.index(cls.lower()) for cls in classes])

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        """
        Optimized: vectorized mask operations and efficient I/O.
        """
        # Read data
        image = cv2.imread(self.images[i])
        if image is None:
            raise ValueError(f"Failed to read image: {self.images[i]}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        mask = cv2.imread(self.masks[i], 0)
        if mask is None:
            raise ValueError(f"Failed to read mask: {self.masks[i]}")

        # Vectorized mask extraction - more efficient than list comprehension
        mask_binary = (mask[..., np.newaxis] == self.class_values).astype(np.float32)

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_binary)
            image, mask_binary = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask_binary)
            image, mask_binary = sample['image'], sample['mask']

        return image, mask_binary

    def __len__(self):
        return len(self.ids)