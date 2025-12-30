from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class ImageClassificationDataset(Dataset):
    """
    Класс для датасета изображений.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_df: Optional[pd.DataFrame],
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform

        if labels_df is not None:
            # train / val
            self.ids = labels_df["Id"].values
            self.labels = labels_df["Category"].values
        else:
            # test
            self.ids = sorted([p.name for p in self.images_dir.iterdir()])
            self.labels = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_path = self.images_dir / image_id

        # Загружаем изображение
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            
        # train / val
        if self.labels is not None:
            label = int(self.labels[idx])
            return image, label

        # test
        return image, image_id
