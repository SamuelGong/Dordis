from abc import abstractmethod


class Handler:
    def __init__(self):
        pass

    @abstractmethod
    def set_key(self, key):
        """ """

    @abstractmethod
    def encrypt(self, plaintext):
        """ """

    @abstractmethod
    def decrypt(self, ciphertext):
        """ """
