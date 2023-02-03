from abc import abstractmethod


class Handler:
    def __init__(self):
        pass

    @abstractmethod
    def generate_key_pairs(self):
        """ """

    @staticmethod
    def secret_key_to_bytes(sk):
        """ """

    @staticmethod
    def public_key_to_bytes(pk):
        """ """

    @staticmethod
    def bytes_to_secret_key(bytes):
        """ """

    @staticmethod
    def bytes_to_public_key(bytes):
        """ """

    @staticmethod
    def generate_shared_key(sk, pk):
        """ """