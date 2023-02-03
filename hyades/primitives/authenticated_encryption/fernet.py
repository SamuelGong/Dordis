from cryptography.fernet import Fernet
from hyades.primitives.authenticated_encryption import base


class Handler(base.Handler):
    def __init__(self):
        super().__init__()
        self.key = None
        self.engine = None

    def set_key(self, key):
        # the key should be url safe for Fernet
        self.key = key
        self.engine = Fernet(key)

    def encrypt(self, plaintext):
        return self.engine.encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.engine.decrypt(ciphertext)
