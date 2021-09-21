from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from pycocotools import coco, mask as coco_mask
from torch.utils import data


def load_annotated_ids(coco: coco.COCO):
    whole_image_ids = coco.getImgIds()

    image_ids = []

    for id in whole_image_ids:
        annotations_ids = coco.getAnnIds(imgIds=id, iscrowd=False)
        if len(annotations_ids) > 0:
            image_ids.append(id)

    return image_ids


class COCODataset(data.Dataset):
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

    def __len__(self):
        return len(self.__image_ids)

    def __getitem__(self, idx) -> Tuple[Image.Image, np.ndarray]:
        image_id = self.__image_ids[idx]

        image_info = self.__coco.loadImgs(image_id)[0]
        path = self.data_path.joinpath(image_info['file_name'])
        image = Image.open(path).convert('RGB')

        annotations_ids = self.__coco.loadAnns(self.__coco.getAnnIds(imgIds=image_id))
        masks = Image.fromarray(self._gen_seg_masks(annotations_ids, image_info['height'], image_info['width']))

        return image, masks

    def _gen_seg_masks(self, target, h, w):
        masks = []
        for instance in target:
            rle = coco_mask.frPyObjects(instance['Segmentation'], h, w)
            m = coco_mask.decode(rle)
            if len(m.shape) < 3:
                masks.append(m)
            else:
                masks.append((np.sum(m, axis=2)) > 0)
        return masks

