from torchvision import transforms
import numpy as np
import torch

# посчитанные на трейне
TRAIN_MEAN = (0.5691487789154053, 0.5446962714195251, 0.4932887554168701)
TRAIN_STD  = (0.18759706616401672, 0.18629783391952515, 0.1906362920999527)

# TRAIN_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
# TRAIN_STD = [0.229, 0.224, 0.225]   # ImageNet std

def get_base_transforms():
    """
    Без аугментаций.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=TRAIN_MEAN,
            std=TRAIN_STD,
        ),
    ])


class AddGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std
        return tensor + noise
    
    def __repr__(self):
        return self.__class__.__name__ + f'(std={self.std})'

class Cutout:
    def __init__(self, n_holes=1, length=12):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones((h, w), dtype=img.dtype, device=img.device)

        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0

        mask = mask.expand_as(img)
        return img * mask

    def __repr__(self):
        return f"{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length})"



def get_train_transforms():
    """
    Аугментации для трейна:
    - RandomCrop с паддингом 4 и отражением
    - RandomHorizontalFlip
    - ColorJitter (brightness, contrast, saturation, hue)
    - AutoAugment с политикой CIFAR10
    - Cutout (1 квадрат 12x12)
    - Gaussian Noise (std=0.01)
    """
    return transforms.Compose([
        
        transforms.RandomCrop(
            40, 
            padding=4, 
            padding_mode='reflect'
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        
        transforms.ColorJitter(
            brightness=0.4,   
            contrast=0.4,     
            saturation=0.4,   
            hue=0.1           
        ),
        
        transforms.AutoAugment(
            policy=transforms.AutoAugmentPolicy.CIFAR10,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
        
        # Cutout(n_holes=1, length=12),  # Затирает квадрат 12x12 = 9% изображения
        AddGaussianNoise(std=0.01)
    ])