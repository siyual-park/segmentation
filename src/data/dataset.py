import abc
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from pycocotools import coco, mask as coco_mask
from torch.utils import data


class COCOSegmentationDataset(data.Dataset, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplemented

    @abc.abstractmethod
    def __getitem__(self, idx) -> Tuple[Image.Image, List[Image.Image]]:
        raise NotImplemented


def load_annotated_ids(coco: coco.COCO):
    whole_image_ids = coco.getImgIds()

    image_ids = []

    for id in whole_image_ids:
        annotations_ids = coco.getAnnIds(imgIds=id, iscrowd=False)
        if len(annotations_ids) > 0:
            image_ids.append(id)

    return image_ids


class COCOSegmentationRemoteDataset(COCOSegmentationDataset):
    def __init__(
            self,
            path: str or Path,
            dataset: str
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
