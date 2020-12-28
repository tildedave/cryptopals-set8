from binascii import hexlify, unhexlify
from hashlib import sha256
from numtheory import mod_inverse, random_prime
from typing import NamedTuple


class RSAKeypair(NamedTuple):
    public: int
    secret: int
    exponent: int

    @staticmethod
    def create(e: int) -> 'RSAKeypair':
        p, q = random_prime(e), random_prime(e)
        n = p * q
        et = (p - 1) * (q - 1)
        d = mod_inverse(e, et)

        return RSAKeypair(e, d, n)


class RSASignature(NamedTuple):
    message: str
    signature: int


def rsa_encrypt(msg: str, keypair: RSAKeypair) -> int:
    n = int(hexlify(msg.encode()), 16)
    assert n < keypair.exponent
    return pow(n, keypair.public, keypair.exponent)


def rsa_decrypt(cipher: int, keypair: RSAKeypair) -> str:
    decoded = pow(cipher, keypair.secret, keypair.exponent)
    # So this is the hex string associated with the msg
    hex_string = '{:02x}'.format(decoded)
    return unhexlify(hex_string).decode()


def hash_msg(msg: str):
    hash_obj = sha256(msg.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16)


def rsa_sign(msg: str, keypair: RSAKeypair) -> int:
    # TODO - pad message
    h = hash_msg(msg) % keypair.exponent
    assert h < keypair.exponent
    return RSASignature(msg, pow(h, keypair.public, keypair.exponent))


def rsa_verify(sig: RSASignature, keypair: RSAKeypair) -> bool:
    h = hash_msg(sig.message) % keypair.exponent
    return pow(sig.signature, keypair.secret, keypair.exponent) == h


def test_create_keypair():
    keypair = RSAKeypair.create(e=3)
    assert rsa_decrypt(rsa_encrypt("bob", keypair), keypair) == "bob"


def test_signature():
    keypair = RSAKeypair.create(e=random_prime())
    other_keypair = RSAKeypair.create(e=random_prime())
    signature = rsa_sign('Bill the Butcher', keypair)

    assert rsa_verify(signature, keypair)
    assert rsa_verify(signature, other_keypair) == False
