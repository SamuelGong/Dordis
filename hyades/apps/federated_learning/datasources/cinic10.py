import logging
import os
import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import Subset

from hyades.config import Config
from hyades.apps.federated_learning.datasources import base


class DataSource(base.DataSource):
    def __init__(self, client_id, quiet=False):
        super().__init__(client_id, quiet)
        meta_path = Config().app.data.data_path \
            if hasattr(Config().app.data, "data_path") \
            else "./data"
        _path = os.path.join(meta_path, "CINIC-10")

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                 [0.24205776, 0.23828046, 0.25874835])
        ])

        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple":
            to_notify_clients = False
            signal_file_path = os.path.join(meta_path, "cinic10-signal.txt")
            if not os.path.isfile(signal_file_path):
                if self.client_id == 0:
                    to_notify_clients = True
                    logging.info(f"[Simulation] Fetching the data...")
                else:
                    # wait until the server end downloads all the folders
                    waited_sec = 1
                    while not os.path.isfile(signal_file_path):
                        time.sleep(1)
                        logging.info(f"[Simulation] [{waited_sec}s] Client #{self.client_id} "
                                     f"waiting for the server to complete the fetching.")
                        waited_sec += 1

        if not os.path.exists(_path):
            logging.info(
                "Downloading the CINIC-10 dataset. This may take a while.")
            url = Config().app.data.download_url if hasattr(
                Config().app.data, 'download_url'
            ) else 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'
            DataSource.download(url, _path, quiet=self.quiet)

        self.trainset = datasets.ImageFolder(root=os.path.join(_path, 'train'),
                                             transform=_transform)
        self.trainset_size = 90000

        testset = datasets.ImageFolder(root=os.path.join(_path, 'test'),
                                            transform=_transform)
        self.testset_size = len(testset) // 3
        taken = np.random.choice(len(testset), self.testset_size)
        self.testset = Subset(testset, taken)

        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple":
            if self.client_id == 0 and to_notify_clients:
                logging.info(f"[Simulation] Fetched successfully. "
                             f"Notifying the clients to proceed...")
                with open(signal_file_path, "w") as fout:
                    fout.writelines(["Downloaded."])

        self.print_partition_size()

    def num_train_examples(self):
        return self.trainset_size

    def num_test_examples(self):
        return self.testset_size
