import argparse
import os
from pathlib import Path

from src.data.dataset import COCODataset
from src.data.generator import SegmentationImageGenerator


def generate(
        origin_path: str or Path,
        path: str or Path,
        dataset: str,
        force: bool = False
):
    origin_path = Path(origin_path)
    path = Path(path)

    coco = COCODataset(
        path=origin_path,
        dataset=dataset
    )

    segmentation_image_generator = SegmentationImageGenerator(
        dataset=coco,
        path=path,
        format='jpg'
    )

    segmentation_image_generator.generate(force=force)


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent

    origin_path = root_path.joinpath('data').joinpath('coco')
    date_path = root_path.joinpath('data').joinpath('instances')

    parser = argparse.ArgumentParser()

    parser.add_argument('--origin_path', type=str, default=str(origin_path))
    parser.add_argument('--path', type=str, default=str(date_path))
    parser.add_argument('--dataset', type=str, nargs='+', default=['train2017', 'val2017'])
    parser.add_argument('--force', type=bool, default=False)

    args = parser.parse_args()

    for dataset in args.dataset:
        generate(
            args.origin_path,
            args.path,
            dataset,
            args.force
        )
