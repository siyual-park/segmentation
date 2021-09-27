import abc
import math
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from src.data.dataloader import SegmentationDataLoader
from src.loss import dice_loss
from src.model.segmentation import Mask
from src.optimiser.lookahead import Lookahead
from src.optimiser.radam import RAdam
from src.train.checkpoint import HardCheckpoint, SoftCheckpoint


class Trainer:
    def __init__(
            self,
            checkpoint: str or Path,
            model: nn.Module,
            optimizer: Optimizer,
    ):
        checkpoint = Path(checkpoint)

        checkpoint.mkdir(parents=True, exist_ok=True)

        best_checkpoint_path = checkpoint.joinpath('best.pt')
        last_checkpoint_path = checkpoint.joinpath('last.pt')

        self._model = model
        self._optimizer = optimizer

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._model.to(self._device)

        self._best_checkpoint = HardCheckpoint(
            path=best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=0,
            loss=float('inf')
        )

        self._last_checkpoint = HardCheckpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=0,
            loss=float('inf')
        )

        self._in_memory_checkpoint = SoftCheckpoint(
            model=model,
            optimizer=optimizer,
            epoch=0,
            loss=float('inf')
        )

    async def run(self, epochs: int) -> None:
        self.load()
        await self.sync_best_checkpoint()

        print(
            'Training start. Final epochs is {:3d}, pre best loss is {:5.2f}.'.format(
                epochs,
                self._best_checkpoint.loss
            ),
            flush=True
        )

        start_time = time()

        for epoch in range(self._last_checkpoint.epoch + 1, epochs + 1):
            self._last_checkpoint.epoch = epoch

            epoch_start_time = time()

            self._last_checkpoint.loss = await self.train()
            self._last_checkpoint.loss = await self.evaluate()

            epoch_end_time = time()

            print(
                '{:3d} epoch, {:5.2f} loss, {:8.2f} ppl, {:5.2f}s'.format(
                    epoch,
                    self._last_checkpoint.loss,
                    math.exp(self._last_checkpoint.loss),
                    (epoch_end_time - epoch_start_time),
                ),
                flush=True
            )

            self.save()

        end_time = time()

        print('Training finish.', flush=True)
        print(
            '{:3d} epoch, {:5.2f} valid loss, {:8.2f} ppl, {:5.2f}s'.format(
                self._best_checkpoint.epoch,
                self._best_checkpoint.loss,
                math.exp(self._best_checkpoint.loss),
                (end_time - start_time),
            ),
        )

    def load(self):
        self._in_memory_checkpoint.save()
        self._best_checkpoint.load(map_location=self._device)
        self._in_memory_checkpoint.load(map_location=self._device)

        self._last_checkpoint.load(map_location=self._device)

    def save(self):
        self._last_checkpoint.save()

        if self._best_checkpoint.loss >= self._last_checkpoint.loss:
            self._best_checkpoint.loss = self._last_checkpoint.loss
            self._best_checkpoint.epoch = self._last_checkpoint.epoch

            self._best_checkpoint.save()

    async def sync_best_checkpoint(self):
        self._in_memory_checkpoint.save()

        loaded = self._best_checkpoint.load(map_location=self._device)
        if loaded:
            self._best_checkpoint.loss = await self.evaluate()

        self._in_memory_checkpoint.load(map_location=self._device)

    @abc.abstractmethod
    async def train(self) -> float:
        raise NotImplemented

    @abc.abstractmethod
    async def evaluate(self) -> float:
        raise NotImplemented


class MaskTrainer(Trainer):
    def __init__(
            self,
            checkpoint: str or Path,
            model: Mask,
            train_dataset: SegmentationDataLoader,
            val_dataset: SegmentationDataLoader,
            lr: float,
            k: int,
            alpha: float
    ):
        optimizer = RAdam(model.parameters(), lr=lr)
        optimizer = Lookahead(optimizer, k=k, alpha=alpha)

        super().__init__(
            checkpoint,
            model,
            optimizer
        )

        self.__train_dataset = train_dataset
        self.__val_dataset = val_dataset

        self.__criterion = nn.BCELoss()
        self.__criterion.to(self._device)

    async def train(self) -> float:
        self._model.train()

        self.__train_dataset.shuffle()

        total_loss = 0.0

        data = tqdm(self.__train_dataset)
        for i, (origin_images, mask_images, image_sizes) in enumerate(data):
            origin_images = origin_images.to(self._device)
            mask_images = mask_images.to(self._device)
            image_sizes = image_sizes.to(self._device)

            self._optimizer.zero_grad()

            result = self._model(origin_images)

            loss_1 = self.__criterion(result, mask_images)
            loss_2 = dice_loss(result, mask_images)

            image_size = self.__train_dataset.image_size
            weight = (image_sizes[:, 0] * image_sizes[:, 1]) / (image_size[0] * image_size[1])

            loss = loss_1 + loss_2
            loss = loss * weight

            loss.backward()

            total_loss += loss.item()

            self._optimizer.step()

            cur_loss = total_loss / (i + 1)
            data.set_description('{:3d} epoch, {:5.2f} loss, {:8.2f} ppl'.format(
                self._last_checkpoint.epoch,
                cur_loss,
                math.exp(cur_loss)
            ))

        return total_loss / len(data)

    async def evaluate(self) -> float:
        self._model.eval()

        self.__val_dataset.shuffle()

        total_loss = 0.0

        with torch.no_grad():
            for origin_images, mask_images in tqdm(self.__val_dataset):
                origin_images = origin_images.to(self._device)
                mask_images = mask_images.to(self._device)

                result = self._model(origin_images)

                loss_1 = self.__criterion(result, mask_images)
                loss_2 = dice_loss(result, mask_images)

                loss = loss_1 + loss_2

                total_loss += loss.item()

        return total_loss / len(self.__val_dataset)
