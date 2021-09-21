import argparse
import os
from pathlib import Path
from time import time

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from src.data.dataset import SegmentationDetectDataset
from src.model.segmentation import Mask


def detect(
        model: Mask,
        date_path: Path or str,
        dataset: str
):
    format = 'jpg'

    dataset = SegmentationDetectDataset(
        path=date_path,
        dataset=dataset,
        format=format
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    total_time = 0.0

    to_image = transforms.ToPILImage()

    with torch.no_grad():
        for origin_images, image_path in tqdm(dataset):
            origin_images = origin_images.to(device)

            start = time()
            mask = model(origin_images)
            end = time()

            total_time += end - start

            mask_image = to_image(mask)
            mask_image.save(image_path.joinpath(f'mask.{format}'))

    print(
        '{:5.2f}s'.format(total_time / len(dataset)),
        flush=True
    )


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))

    root_path = path.parent
    root_parent_path = path.parent.parent

    data_path = root_parent_path.joinpath('data')
    checkpoints_path = root_path.joinpath('checkpoints')

    coco_data_path = data_path.joinpath('coco')
    date_path = data_path.joinpath('coco_segment')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='detect')
    parser.add_argument('--deep', type=int, default=2)

    args = parser.parse_args()

    mask = Mask(
        channels=32,
        deep=args.deep,
        expansion=0.5,
        dropout_prob=0.0
    )

    detect(
        model=mask,
        date_path=date_path,
        dataset=args.dataset,
    )
