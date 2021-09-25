import argparse
import asyncio
import os
from pathlib import Path
from time import time

from src.data.dataloader import SegmentationDataLoader
from src.data.dataset import SegmentationDataset
from src.model.segmentation import Mask
from src.train.trainer import MaskTrainer


def train(
        model: Mask,
        date_path: Path or str,
        train_dataset: str,
        val_dataset: str,
        image_size: int,
        batch_size: int,
        checkpoint: Path or str,
        lr: float,
        k: int,
        alpha: float,
        epochs: int
):
    train_dataset = SegmentationDataset(
        path=date_path,
        dataset=train_dataset,
        format='jpg'
    )
    val_dataset = SegmentationDataset(
        path=date_path,
        dataset=val_dataset,
        format='jpg'
    )

    train_dataset = SegmentationDataLoader(
        dataset=train_dataset,
        image_size=image_size,
        batch_size=batch_size
    )
    val_dataset = SegmentationDataLoader(
        dataset=val_dataset,
        image_size=image_size,
        batch_size=batch_size
    )

    trainer = MaskTrainer(
        checkpoint=checkpoint,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=lr,
        k=k,
        alpha=alpha
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(trainer.run(epochs=epochs))


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

    parser.add_argument('--train', type=str, default='train2017')
    parser.add_argument('--val', type=str, default='val2017')

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--deep', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.4)

    args = parser.parse_args()

    mask = Mask(
        channels=args.channels,
        deep=args.deep,
        expansion=0.5,
        dropout_prob=args.dropout_prob
    )

    train(
        model=mask,
        date_path=date_path,
        train_dataset=args.train,
        val_dataset=args.val,
        image_size=args.image_size,
        batch_size=args.batch_size,
        checkpoint=checkpoints_path.joinpath(args.checkpoint),
        lr=args.lr,
        k=args.k,
        alpha=args.alpha,
        epochs=args.epochs
    )

