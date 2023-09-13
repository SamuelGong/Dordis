import logging
import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from dordis.primitives.key_agreement import base


class Handler(base.Handler):
    def __init__(self):
        super().__init__()
        pass

    def generate_key_pairs(self):
        sk = ec.generate_private_key(
            curve=ec.SECP384R1(),
        )
        pk = sk.public_key()
        return sk, pk

    @staticmethod
    def secret_key_to_bytes(sk):
        return sk.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @staticmethod
    def public_key_to_bytes(pk):
        return pk.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    @staticmethod
    def bytes_to_secret_key(bytes):
        return serialization.load_pem_private_key(
            data=bytes,
            password=None
        )

    @staticmethod
    def bytes_to_public_key(bytes):
        return serialization.load_pem_public_key(
            data=bytes
        )

    @staticmethod
    def generate_shared_key(sk, pk):
        key_material = sk.exchange(ec.ECDH(), pk)
        shared_key = HKDF(
            algorithm=SHA256(),
            length=32,
            salt=None,
            info=None
        ).derive(
            key_material=key_material
        )

        return base64.urlsafe_b64encode(shared_key)
