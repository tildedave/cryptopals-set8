from hashlib import sha1
import random
from random import randint
from typing import NamedTuple, Optional

import gmpy2

from numtheory import mod_inverse


p = 0x800000000000000089e1855218a0e7dac38136ffafa72eda7859f2171e25e65eac698c1702578b07dc2a1076da241c76c62d374d8389ea5aeffd3226a0530cc565f3bf6b50929139ebeac04f48c3c84afb796d61e5a4f9a8fda812ab59494232c7d2b4deb50aa18ee9e132bfa85ac4374d7f9091abc3d015efc871a584471bb1
q = 0xf4f47f05794b256174bba6e9b396a7707e563c5b
g = 0x5958c9d3898b224b12672c0b98e06c60df923cb8bc999d119458fef538b8fa4046c8db53039db620c094c9fa077ef389b5322a559946a71903f990f1f7e0e025e2d7f7cf494aff1a0470f5b64c36b625a097f1651fe775323556fe00b3608c887892878480e99041be601a62166ca6894bdd41a7054ec89f756ba9fc95302291

def test_parameters():
    assert gmpy2.is_prime(p)
    assert gmpy2.is_prime(q)

SIGN_MSG = """For those that envy a MC it can be hazardous to your health
So be friendly, a matter of life and death, just like a etch-a-sketch
"""


class DSAKeypair(NamedTuple):
    public: int
    private: int
    g: int
    p: int
    q: int


class DSASignature(NamedTuple):
    msg: str
    r: int
    s: int


def hash_msg(msg: str):
    hash_obj = sha1(msg.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16)


def dsa_sign(msg: str, keypair: DSAKeypair, k: Optional[int] = None):
    r = 0
    while r == 0:
        k_ = randint(1, keypair.q) if not k else k
        r = pow(g, k_, keypair.p) % q
    s = 0

    while s == 0:
        s = (mod_inverse(k_, q) * (hash_msg(msg) + keypair.private * r)) % q

    return DSASignature(msg, r, s)


def dsa_verify(sig: DSASignature, keypair: DSAKeypair):
    p, q = keypair.p, keypair.q
    assert 0 < sig.r < q
    assert 0 < sig.s < q

    w = mod_inverse(sig.s, q)
    u1 = (hash_msg(sig.msg) * w) % q
    u2 = (sig.r * w) % q
    v = ((pow(g, u1, p) * pow(keypair.public, u2, p)) % p) % q

    return v == sig.r


def test_dsa_sign():
    x = randint(1, q)
    y = pow(g, x, p)
    test_keypair = DSAKeypair(y, x, g, p , q)
    sig = dsa_sign(SIGN_MSG, test_keypair)
    assert dsa_verify(sig, test_keypair)


def test_dsa_attack():
    x = randint(1, q)
    y = pow(g, x, p)
    test_keypair = DSAKeypair(y, x, g, p , q)
    k = 710184121617032319901844589822607487992346783541
    sig = dsa_sign(SIGN_MSG, test_keypair, k=k)

    attack_keypair = get_attack_keypair(SIGN_MSG, sig, y, k)
    assert attack_keypair.private == x

    dsa_sign(SIGN_MSG, attack_keypair, k=k)



def get_attack_keypair(msg, sig, y, k):
    r_inv = mod_inverse(sig.r, q)
    x = ((sig.s * k - hash_msg(msg)) * r_inv) % q
    return DSAKeypair(y, x, g, p, q)


def test_recover_keypair():
    y = 0x84ad4719d044495496a3201c8ff484feb45b962e7302e56a392aee4abab3e4bdebf2955b4736012f21a08084056b19bcd7fee56048e004e44984e2f411788efdc837a0d2e5abb7b555039fd243ac01f0fb2ed1dec568280ce678e931868d23eb095fde9d3779191b8c0299d6e07bbb283e6633451e535c45513b2d33c99ea17
    sig = DSASignature(SIGN_MSG,
                       r=548099063082341131477253921760299949438196259240,
                       s=857042759984254168557880549501802188789837994940)

    # We know nonce is between 0 and 2^16
    for k in range(0, 2**16):
        attack_keypair = get_attack_keypair(SIGN_MSG, sig, y, k)
        test_sig = dsa_sign(SIGN_MSG, attack_keypair, k=k)
        if sig.r == test_sig.r and sig.s == test_sig.s:
            secret_str = '{:02x}'.format(attack_keypair.private)
            hash_obj = sha1(secret_str.encode('utf-8'))
            digest = hash_obj.hexdigest()
            assert digest == '0954edd5e0afe5542a4adf012611a91912a3ec16'
            break
