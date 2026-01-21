from torchvision import transforms

# посчитанные на трейне
TRAIN_MEAN = (0.5691487789154053, 0.5446962714195251, 0.4932887554168701)
TRAIN_STD  = (0.18759706616401672, 0.18629783391952515, 0.1906362920999527)

def get_base_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=TRAIN_MEAN,
            std=TRAIN_STD,
        )
    ])

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomCrop(40, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
        transforms.RandomErasing(p=0.20),
    ])