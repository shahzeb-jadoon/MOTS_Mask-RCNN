"""
Data Augmentation
data_augmentation.py

Artificially generating new data from existing data
"""

from torchvision.transforms import v2 # https://pytorch.org/vision/stable/transforms.html

color_aug = v2.Compose([
    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), #and hue=0.5
    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # v2.Grayscale(p=0.1),
    # v2.HorizontalFlip(p=0.5),
    # v2.VerticalFlip(p=0.5),
    # v2.Rotation(45),
    # v2.ResizedCrop(256, scale=(0.5, 1.0)),
    # v2.Affine(degrees=45),
    # v2.RandomPerspective(),
    # v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])