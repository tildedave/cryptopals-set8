from hashlib import sha1
import random
from random import randint
from typing import NamedTuple, Optional

import gmpy2

from numtheory import mod_inverse, random_prime


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
    g, p, q = keypair.g, keypair.p, keypair.q
    r = 0
    while r == 0:
        k_ = randint(1, q) if not k else k
        r = pow(g, k_, p) % q
    s = 0

    while s == 0:
        s = (mod_inverse(k_, q) * (hash_msg(msg) + keypair.private * r)) % q

    return DSASignature(msg, r, s)


def dsa_verify(sig: DSASignature, keypair: DSAKeypair):
    g, p, q = keypair.g, keypair.p, keypair.q
    assert 0 < sig.r < q
    assert 0 < sig.s < q

    w = mod_inverse(sig.s, q)
    u1 = (hash_msg(sig.msg) * w) % q
    u2 = (sig.r * w) % q
    v = ((pow(g, u1, p) * pow(keypair.public, u2, p)) % p) % q

    return v == sig.r


def test_dsa_sign():
    p = 0x800000000000000089e1855218a0e7dac38136ffafa72eda7859f2171e25e65eac698c1702578b07dc2a1076da241c76c62d374d8389ea5aeffd3226a0530cc565f3bf6b50929139ebeac04f48c3c84afb796d61e5a4f9a8fda812ab59494232c7d2b4deb50aa18ee9e132bfa85ac4374d7f9091abc3d015efc871a584471bb1
    q = 0xf4f47f05794b256174bba6e9b396a7707e563c5b
    g = 0x5958c9d3898b224b12672c0b98e06c60df923cb8bc999d119458fef538b8fa4046c8db53039db620c094c9fa077ef389b5322a559946a71903f990f1f7e0e025e2d7f7cf494aff1a0470f5b64c36b625a097f1651fe775323556fe00b3608c887892878480e99041be601a62166ca6894bdd41a7054ec89f756ba9fc95302291
    x = randint(1, q)
    y = pow(g, x, p)
    test_keypair = DSAKeypair(y, x, g, p , q)
    sig = dsa_sign("Bob the message", test_keypair)
    assert dsa_verify(sig, test_keypair)
