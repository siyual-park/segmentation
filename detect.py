import argparse
import os
from pathlib import Path
from time import time

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from src.data.dataloader import SegmentationDetectDataLoader
from src.data.dataset import SegmentationDetectDataset
from src.model.segmentation import Mask
from src.train.checkpoint import HardCheckpoint, Checkpoint


def detect(
        model: Mask,
        checkpoint: Checkpoint,
        date_path: Path or str,
        dataset: str,
        image_size: int,
        batch_size: int,
):
    format = 'jpg'

    dataset = SegmentationDetectDataset(
        path=date_path,
        dataset=dataset,
        format=format
    )
    data_loader = SegmentationDetectDataLoader(
        dataset=dataset,
        image_size=image_size,
        batch_size=batch_size
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    checkpoint.load(map_location=device)

    model.eval()

    total_time = 0.0
    total_size = 0

    to_image = transforms.ToPILImage()

    with torch.no_grad():
        for origin_images, image_paths, image_sizes in tqdm(data_loader):
            origin_images = origin_images.to(device)

            start = time()
            masks = model(origin_images)
            end = time()

            total_time += end - start
            total_size += len(image_paths)

            for i, mask in enumerate(masks):
                image_path = image_paths[i]
                image_size = image_sizes[i]

                mask_image = to_image(mask)
                mask_image = mask_image.resize(image_size)
                mask_image.save(image_path.joinpath(f'mask.{format}'))

    print(
        '{:5.2f}s'.format(total_time / total_size),
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

    parser.add_argument('--checkpoint', type=str, default=str(int(time())))

    parser.add_argument('--dataset', type=str, default='detect')

    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--deep', type=int, default=8)

    args = parser.parse_args()

    mask = Mask(
        channels=args.channels,
        deep=args.deep,
        expansion=0.5,
        dropout_prob=0.0
    )

    checkpoint = HardCheckpoint(
        path=checkpoints_path.joinpath(args.checkpoint),
        model=mask,
        optimizer=None,
        epoch=0,
        loss=float('inf')
    )

    detect(
        model=mask,
        checkpoint=checkpoint,
        date_path=date_path,
        dataset=args.dataset,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
