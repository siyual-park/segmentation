import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.dataset import COCOSegmentationRemoteDataset
from src.data.utils import get_data_size


class COCOSegmentationCacheGenerator:
    def __init__(
            self,
            dataset: COCOSegmentationRemoteDataset,
            path: str or Path,
            format: str
    ):
        path = Path(path)

        self.__dataset = dataset
        self.__path = path.joinpath(dataset.dataset)
        self.__format = format

    def generate(self, force: bool = False) -> None:
        if os.path.exists(self.__path) and force:
            os.remove(self.__path)

        self.__path.mkdir(parents=True, exist_ok=True)

        existed_data_size = get_data_size(self.__path)
        current_data_index = 0

        print(f'Cache coco images from {self.__dataset.data_path} to {self.__path}')
        for image, masks in tqdm(self.__dataset):
            current_data_index += 1
            if current_data_index <= existed_data_size:
                continue

            image_dir = self.__path.joinpath(str(current_data_index - 1))
            image_dir.mkdir(parents=True)

            image.save(image_dir.joinpath(f'origin.{self.__format}'))

            for i, mask in enumerate(masks):
                mask.save(image_dir.joinpath(f'{i}.{self.__format}'))
