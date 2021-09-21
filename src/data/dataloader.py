from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils import data
from torchvision import transforms

from src.common_types import size_2_t
from src.data.dataset import SegmentationDataset, SegmentationDetectDataset


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

        self.__image_to_tensor = transforms.ToTensor()
        self.__normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def shuffle(self):
        self.__dataset.shuffle()

    def __len__(self):
        return len(self.__dataset) // self.batch_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        origin_images = []
        mask_images = []

        for i in range(self.batch_size):
            origin_image, mask_image = self.__dataset[idx * self.batch_size + i]
            if origin_image is None or mask_image is None:
                continue

            origin_image = origin_image.resize(self.image_size)
            mask_image = mask_image.resize(self.image_size)

            origin_image = self.__image_to_tensor(origin_image)
            mask_image = self.__image_to_tensor(mask_image)

            origin_image = self.__normalize(origin_image)

            origin_images.append(origin_image)
            mask_images.append(mask_image)

        return torch.stack(origin_images), torch.stack(mask_images)


class SegmentationDetectDataLoader(data.Dataset):
    def __init__(
            self,
            dataset: SegmentationDetectDataset,
            image_size: size_2_t,
            batch_size: int = 1,
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.__dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size

        self.__image_to_tensor = transforms.ToTensor()
        self.__normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def shuffle(self):
        self.__dataset.shuffle()

    def __len__(self):
        return len(self.__dataset) // self.batch_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, List[Path], List[Tuple[int, int]]]:
        origin_images = []
        image_paths = []
        image_sizes = []

        for i in range(self.batch_size):
            origin_image, image_path = self.__dataset[idx * self.batch_size + i]
            if origin_image is None:
                continue

            image_sizes.append(origin_image.size)

            origin_image = origin_image.resize(self.image_size)

            origin_image = self.__image_to_tensor(origin_image)
            origin_image = self.__normalize(origin_image)

            origin_images.append(origin_image)
            image_paths.append(image_path)

        return torch.stack(origin_images), image_paths, image_sizes
