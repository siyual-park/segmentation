from typing import Tuple

import numpy as np
import torch
from torch.utils import data

from src.common_types import size_2_t
from src.data.dataset import SegmentationDataset


class SegmentationDataLoader(data.Dataset):
    def __init__(
            self,
            dataset: SegmentationDataset,
            image_size: size_2_t,
            batch_size: int = 1,
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.__dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size

    def shuffle(self):
        self.__dataset.shuffle()

    def __len__(self):
        return len(self.__dataset) // self.batch_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        origin_images = []
        mask_images = []

        for i in range(self.batch_size):
            origin_image, mask_image = self.__dataset[idx * self.batch_size + i]

            origin_image = origin_image.resize(self.image_size)
            mask_image = mask_image.resize(self.image_size)

            origin_image = np.array(origin_image).astype('float32')
            mask_image = np.array(mask_image).astype('float32')
            mask_image = mask_image / 255

            origin_image = torch.from_numpy(origin_image)
            mask_image = torch.from_numpy(mask_image)

            origin_images.append(origin_image)
            mask_images.append(mask_image)

        return torch.stack(origin_images), torch.stack(mask_images)
