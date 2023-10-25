import json
import logging
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dordis.config import Config
from dordis.apps.federated_learning.datasources import base


class CustomDictDataset(Dataset):
    def __init__(self, loaded_data, transform=None):
        super().__init__()
        self.loaded_data = loaded_data
        self.transform = transform

    def __getitem__(self, index):
        sample = self.loaded_data['x'][index]
        target = self.loaded_data['y'][index]
        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.loaded_data['y'])


class ReshapeListTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, img):
        return np.array(img, dtype=np.float32).reshape(self.new_shape)


class DataSource(base.DataSource):
    def __init__(self, client_id, quiet=False):
        super().__init__(client_id, quiet)

        _path = Config().app.data.data_path \
            if hasattr(Config().app.data, "data_path") \
            else "./data"
        root_path = os.path.join(_path, 'FEMNIST', 'packaged_data')

        data_urls = {}
        if self.client_id == 0:
            # If we are on the federated learning server
            data_dir = os.path.join(root_path, 'test')
            data_urls[self.client_id] = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/FEMNIST/test/" \
                                        + str(self.client_id) + ".zip"
        else:
            data_dir = os.path.join(root_path, 'train')
            data_urls[self.client_id] = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/FEMNIST/train/" \
                                        + str(self.client_id) + ".zip"

            if hasattr(Config().app.data, "augment") \
                    and Config().app.data.augment.type == "simple":
                repeat = Config().app.data.augment.repeat
                total_clients = Config().clients.total_clients
                for i in range(0, repeat):
                    client_id = (i + 1) * total_clients + self.client_id
                    data_urls[client_id] = "https://jiangzhifeng.s3.us-east-2.amazonaws.com/FEMNIST/train/" \
                                           + str(client_id) + ".zip"

        for client_id, data_url in data_urls.items():
            if not os.path.exists(os.path.join(data_dir, str(client_id))):
                logging.info(
                    f"Downloading the Federated EMNIST dataset for client {client_id}. "
                )
                self.download(url=data_url, data_path=data_dir, quiet=self.quiet)

        loaded_data = {"x": [], "y": []}
        for client_id in data_urls.keys():
            _data = DataSource.read_data(
                file_path=os.path.join(data_dir, str(client_id), 'data.json'))
            loaded_data["x"] += _data["x"]
            loaded_data["y"] += _data["y"]

        logging.info(f"Physical client(s)' data loaded: {list(data_urls.keys())}.")

        _transform = transforms.Compose([
            ReshapeListTransform((28, 28, 1)),
            transforms.ToPILImage(),
            transforms.RandomCrop(28,
                                  padding=2,
                                  padding_mode="constant",
                                  fill=1.0),
            transforms.RandomResizedCrop(28,
                                         scale=(0.8, 1.2),
                                         ratio=(4. / 5., 5. / 4.)),
            transforms.RandomRotation(5, fill=1.0),
            transforms.ToTensor(),
            transforms.Normalize(0.9637, 0.1597),
        ])

        # Currently we are using c5.4xlarge as the server in our used cluster
        # because it has no GPUs, evaluation of aggregated models can take
        # a long time period if the size of testing dataset is too large.
        # However, in real environment, it is not uncommon for the server to have GPU.
        # To recover a reasonable server runtime overhead in a used cluster, we thus do a down-sampling.
        # Note that server's testing datasets are IID and we are thus safe to do that.
        # if self.client_id == 0 and not torch.cuda.is_available():
        if self.client_id == 0:  # temporary use
            num_reserved_samples = 10000  # TODO: avoid hard-coding
            if hasattr(Config().app.data, "num_test_samples"):
                num_reserved_samples = Config().app.data.num_test_samples

            loaded_data["x"] = loaded_data["x"][:num_reserved_samples]
            loaded_data["y"] = loaded_data["y"][:num_reserved_samples]

        dataset = CustomDictDataset(loaded_data=loaded_data,
                                    transform=_transform)

        if self.client_id == 0:
            self.testset = dataset
        else:
            self.trainset = dataset

        self.print_partition_size()

    @staticmethod
    def read_data(file_path):
        with open(file_path, 'r') as fin:
            loaded_data = json.load(fin)
        return loaded_data

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)
