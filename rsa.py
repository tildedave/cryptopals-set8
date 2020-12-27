from binascii import hexlify, unhexlify
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


def rsa_encrypt(msg: str, keypair: RSAKeypair) -> int:
    n = int(hexlify(msg.encode()), 16)
    assert n < keypair.exponent
    return pow(n, keypair.public, keypair.exponent)


def rsa_decrypt(cipher: int, keypair: RSAKeypair) -> str:
    decoded = pow(cipher, keypair.secret, keypair.exponent)
    # So this is the hex string associated with the msg
    hex_string = '{:02x}'.format(decoded)
    return unhexlify(hex_string).decode()
