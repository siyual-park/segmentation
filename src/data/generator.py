import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.dataset import COCOSegmentationDataset
from src.data.utils import get_data_size


class COCOSegmentationGenerator:
    def __init__(
            self,
            dataset: COCOSegmentationDataset,
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
            for i, mask in enumerate(masks):
                current_data_index += 1
                if current_data_index <= existed_data_size:
                    continue

                image_dir = self.__path.joinpath(str(current_data_index - 1))
                image_dir.mkdir(parents=True)

                bbox = self.get_bbox(mask)
                if all(map(lambda x: x == 0, bbox)):
                    continue

                origin_image = image.crop(bbox)
                mask_image = mask.crop(bbox)

                origin_image.save(image_dir.joinpath(f'origin.{self.__format}'))
                mask_image.save(image_dir.joinpath(f'mask.{self.__format}'))

    def get_bbox(self, image: Image.Image) -> List[int]:
        mask = np.array(image).astype('uint8')
        y_index, x_index = np.where(mask != 0)

        if np.size(x_index) == 0 or np.size(y_index) == 0:
            return [0, 0, 0, 0]

        x1 = np.min(x_index)
        y1 = np.min(y_index)

        x2 = np.max(x_index)
        y2 = np.max(y_index)

        return [x1, y1, x2 + 1, y2 + 1]

