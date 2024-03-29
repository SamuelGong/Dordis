import os
import time
import logging
from dordis.config import Config
from torchvision import datasets, transforms
from dordis.apps.federated_learning.datasources import base


class DataSource(base.DataSource):
    def __init__(self, client_id, quiet=False):
        super().__init__(client_id, quiet)
        _path = Config().app.data.data_path \
            if hasattr(Config().app.data, "data_path") \
            else "./data"

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple":
            to_notify_clients = False
            signal_file_path = os.path.join(_path, "mnist-signal.txt")
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

        self.trainset = datasets.MNIST(root=_path,
                                       train=True,
                                       download=True,
                                       transform=_transform)
        self.testset = datasets.MNIST(root=_path,
                                      train=False,
                                      download=True,
                                      transform=_transform)

        if hasattr(Config(), "simulation") \
                and Config().simulation.type == "simple":
            if self.client_id == 0 and to_notify_clients:
                logging.info(f"[Simulation] Fetched successfully. "
                             f"Notifying the clients to proceed...")
                with open(signal_file_path, "w") as fout:
                    fout.writelines(["Downloaded."])

        self.print_partition_size()

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)
