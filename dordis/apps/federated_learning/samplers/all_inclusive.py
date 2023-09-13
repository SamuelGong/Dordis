from dordis.apps.federated_learning.samplers import base


class Sampler(base.Sampler):
    def __init__(self, dataset, client_id):
        super().__init__(client_id)
        self.client_id = client_id

        self.all_inclusive = range(dataset.num_train_examples())

    def get(self):
        # return random.shuffle(self.all_inclusive)
        from torch.utils.data import SubsetRandomSampler
        return SubsetRandomSampler(self.all_inclusive)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.all_inclusive)
