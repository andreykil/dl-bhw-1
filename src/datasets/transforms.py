from torchvision import transforms

def get_base_transforms():
    """
    Без аугментаций.
    """
    return transforms.Compose([
        transforms.ToTensor(),
    ])