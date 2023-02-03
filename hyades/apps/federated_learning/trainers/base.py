import logging
import os
import torch
from abc import ABC, abstractmethod
from typing import Tuple
from hyades.config import Config
from hyades.utils.share_memory_handler import ShareBase


class Trainer(ShareBase, ABC):
    def __init__(self):
        ShareBase.__init__(self, client_id=0)
        self.device = "cpu"  # TODO: avoid hard-coding
        self.client_id = 0
        self.model = None
        self.name = None

    def set_client_id(self, client_id):
        self.client_id = client_id
        self._set_client_id(client_id=client_id)  # for ShareBase

        self.name = "Server"
        if self.client_id > 0:
            self.name = f"Client {self.client_id}"

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_no = self.client_id % device_count

            # walkaround for resource sharing
            # device_count = 4
            # device_no = self.client_id % device_count + 4
            self.device = f"cuda:{device_no}"

        logging.info(f"{self.name}'s trainer will use device {self.device}.")

    def switch_cuda_device(self, retry_time):
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_no = (self.client_id + retry_time) % device_count
            self.device = f"cuda:{device_no}"
        return self.device

    def extract_weights(self):
        return self.model.cpu().state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)

    def save_model(self, filename=None):
        model_name = Config().app.trainer.model_name
        model_dir = Config().app.trainer.model_dir \
            if hasattr(Config().app.trainer, "model_dir") \
            else "./models/pretrained/"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_name}.pth'
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, filename=None):
        model_dir = Config().app.trainer.model_dir \
            if hasattr(Config().app.trainer, "model_dir") \
            else "./models/pretrained/"
        model_name = Config().app.trainer.model_name

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_name}.pth'

        self.model.load_state_dict(torch.load(model_path))

    @abstractmethod
    def train(self, trainset, sampler, cut_layer=None) -> Tuple[bool, float]:
        """ The main training loop in a federated learning workload. """

    @abstractmethod
    def test(self, testset) -> float:
        """ Testing the model using the provided test dataset. """

    @abstractmethod
    async def server_test(self, testset):
        """ Testing the model on the server using the provided test dataset. """
