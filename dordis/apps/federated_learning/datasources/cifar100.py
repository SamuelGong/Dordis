import logging

from torchvision import datasets, transforms
import os
import time
from dordis.config import Config
from dordis.apps.federated_learning.datasources import base


class DataSource(base.DataSource):
    def __init__(self, client_id, quiet=False):
        super().__init__(client_id, quiet)
        _path = Config().app.data.data_path \
            if hasattr(Config().app.data, "data_path") \
            else "./data"

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.5074, 0.4867, 0.4411], [0.2011, 0.1987, 0.2025])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5074, 0.4867, 0.4411], [0.2011, 0.1987, 0.2025])
        ])

        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple":
            to_notify_clients = False
            signal_file_path = os.path.join(_path, "cifar100-signal.txt")
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

        self.trainset = datasets.CIFAR100(root=_path,
                                         train=True,
                                         download=True,
                                         transform=train_transform)
        self.testset = datasets.CIFAR100(root=_path,
                                        train=False,
                                        download=True,
                                        transform=test_transform)

        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple":
            if self.client_id == 0 and to_notify_clients:
                logging.info(f"[Simulation] Fetched successfully. "
                             f"Notifying the clients to proceed...")
                with open(signal_file_path, "w") as fout:
                    fout.writelines(["Downloaded."])

        self.print_partition_size()

    def num_train_examples(self):
        return 50000

    def num_test_examples(self):
        return 10000
