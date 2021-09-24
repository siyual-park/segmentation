from pathlib import Path
from random import shuffle
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
from pycocotools import coco, mask as coco_mask
from torch.utils import data

from src.data.utils import get_data_size


def load_annotated_ids(coco: coco.COCO):
    whole_image_ids = coco.getImgIds()

    image_ids = []

    for id in whole_image_ids:
        annotations_ids = coco.getAnnIds(imgIds=id, iscrowd=False)
        if len(annotations_ids) > 0:
            image_ids.append(id)

    return image_ids


class COCOSegmentationDataset(data.Dataset):
    def __init__(
            self,
            path: str or Path,
            dataset: str,
    ):
        path = Path(path)

        self.dataset = dataset
        self.annotations_path = path.joinpath('annotations')
        self.data_path = path.joinpath(dataset)

        self.__coco = coco.COCO(
            self.annotations_path
                .joinpath(f'instances_{dataset}.json')
        )

        self.__image_ids = load_annotated_ids(self.__coco)

    def __len__(self) -> int:
        return len(self.__image_ids)

    def __getitem__(self, idx) -> Tuple[Image.Image, List[Image.Image]]:
        image_id = self.__image_ids[idx]

        image_info = self.__coco.loadImgs(image_id)[0]
        path = self.data_path.joinpath(image_info['file_name'])
        image = Image.open(path).convert('RGB')

        annotations = self.__coco.loadAnns(self.__coco.getAnnIds(imgIds=image_id))
        masks = self._gen_seg_masks(annotations, image_info['height'], image_info['width'])

        return image, masks

    def _gen_seg_masks(self, annotations, height, width):
        masks = []
        for annotation in annotations:
            rle = coco_mask.frPyObjects(annotation['segmentation'], height, width)
            mask = coco_mask.decode(rle)
            if len(mask.shape) >= 3:
                mask = np.sum(mask, axis=2) > 0
            masks.append(Image.fromarray(np.uint8(mask * 255), 'L'))
        return masks


class SegmentationDataset(data.Dataset):
    def __init__(
            self,
            path: str or Path,
            dataset: str,
            format: str
    ):
        path = Path(path)

        self.dataset = dataset
        self.data_path = path.joinpath(dataset)

        self.__format = format

        data_size = get_data_size(self.data_path)
        self.__image_ids = list(range(data_size))

    def shuffle(self):
        shuffle(self.__image_ids)

    def __len__(self):
        return len(self.__image_ids)

    def __getitem__(self, idx) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        id = self.__image_ids[idx]

        images_path = self.data_path.joinpath(str(id))

        origin_image_path = images_path.joinpath(f'origin.{self.__format}')
        mask_image_path = images_path.joinpath(f'mask.{self.__format}')

        if not origin_image_path.exists() or not mask_image_path.exists():
            return None, None

        origin_image = Image.open(origin_image_path).convert('RGB')
        mask_image = Image.open(mask_image_path).convert('L')

        return origin_image, mask_image


class SegmentationDetectDataset(data.Dataset):
    def __init__(
            self,
            path: str or Path,
            dataset: str,
            format: str
    ):
        path = Path(path)

        self.dataset = dataset
        self.data_path = path.joinpath(dataset)

        self.__format = format

        data_size = get_data_size(self.data_path)
        self.__image_ids = list(range(data_size))

    def shuffle(self):
        shuffle(self.__image_ids)

    def __len__(self):
        return len(self.__image_ids)

    def __getitem__(self, idx) -> Tuple[Optional[Image.Image], Path]:
        id = self.__image_ids[idx]

        images_path = self.data_path.joinpath(str(id))

        origin_image_path = images_path.joinpath(f'origin.{self.__format}')
        mask_image_path = images_path.joinpath(f'mask.{self.__format}')

        if not origin_image_path.exists():
            return None, images_path
        if mask_image_path.exists():
            return None, images_path

        origin_image = Image.open(origin_image_path).convert('RGB')

        return origin_image, images_path
