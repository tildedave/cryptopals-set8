from binascii import hexlify, unhexlify
import random
from typing import NamedTuple

import gmpy2

from numtheory import mod_inverse


MAX_TRIES = 1_000


def random_prime(e, range_start=2**120, range_end=2**121) -> int:
    for _ in range(0, MAX_TRIES):
        p = random.randrange(range_start, range_end)
        if not gmpy2.is_prime(p):
            continue

        if (p - 1) % e != 0:
            return p

    raise ValueError(f'Could not find random prime with totient prime to {e}')


class RSAKeypair(NamedTuple):
    public: int
    secret: int
    exponent: int


def encrypt(msg: str, keypair: RSAKeypair) -> int:
    str = int(hexlify(msg.encode()), 16)
    return pow(str, keypair.public, keypair.exponent)


def decrypt(cipher: int, keypair: RSAKeypair) -> str:
    decoded = pow(cipher, keypair.secret, keypair.exponent)
    # So this is the hex string associated with the msg
    hex_string = '{:02x}'.format(decoded)
    return unhexlify(hex_string).decode()


def test_random_prime():
    e = 3
    p, q = random_prime(e), random_prime(e)
    n = p * q
    et = (p - 1) * (q - 1)
    d = mod_inverse(e, et)

    keypair = RSAKeypair(e, d, n)
    assert decrypt(encrypt("bob", keypair), keypair) == "bob"
