from abc import abstractmethod


class Handler:
    def __init__(self):
        pass

    @abstractmethod
    def create_shares(self, secret, t, n):
        """ secret to share_list """

    @abstractmethod
    def combine_shares(self, share_list, aux=None):
        """ share_list to secret """
