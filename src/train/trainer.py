import abc
import math
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.optim import Optimizer

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
