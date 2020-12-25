import random
from hashlib import sha256
from typing import NamedTuple

from elliptic_curve import EllipticCurvePoint, WeierstrassCurve
from numtheory import mod_divide, mod_inverse

from diffie_hellman import DiffieHellman, ECDHConfig, ECDHKeypair

SIGN_MSG = """Call me Ishmael. Some years ago - never mind how long precisely -
having little or no money in my purse, and nothing particular to interest me on
shore, I thought I would sail about a little and see the watery part of the
world."""


def create_dh() -> DiffieHellman:
    p = 233970423115425145524320034830162017933
    curve = WeierstrassCurve(p, -95051, 11279326)
    point = (182, 85518893674295321206118380980485522083)
    point_order = 29246302889428143187362802287225875743

    return DiffieHellman(curve, point, point_order)


class ECDSASignature(NamedTuple):
    point_x: int
    hash: int


def hash_msg(msg: str):
    hash_obj = sha256(msg.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16)


def ecdsa_sign(msg: str, keypair: ECDHKeypair):
    n = keypair.config.n
    k = keypair.config.rand_scalar()
    r = keypair.config.scalar_mult_point(k)[0]
    s = mod_divide(hash_msg(msg) + keypair.secret * r, k, n)

    return ECDSASignature(r, hash=s)


def ecdsa_verify(msg: str, sig: ECDSASignature, keypair: ECDHKeypair):
    curve = keypair.config.curve
    n = keypair.config.n
    u1 = mod_divide(hash_msg(msg), sig.hash, n)
    u2 = mod_divide(sig.point_x, sig.hash, n)
    R = curve.add_points(
        keypair.config.scalar_mult_point(u1),
        curve.scalar_mult(keypair.public, u2))

    return R[0] == sig.point_x


def test_sign_and_verify():
    dh = create_dh()
    keypair = dh.generate_keypair()

    sig = ecdsa_sign(SIGN_MSG, keypair)
    assert ecdsa_verify(SIGN_MSG, sig, keypair), 'Should have verified'


def test_eve_attack():
    random.seed(0)

    dh = create_dh()
    alice_keypair = dh.generate_keypair()
    assert alice_keypair.is_valid()

    config = alice_keypair.config
    sig = ecdsa_sign(SIGN_MSG, alice_keypair)
    assert ecdsa_verify(SIGN_MSG, sig, alice_keypair)

    curve = config.curve
    eve_secret = random.randint(1, config.n)
    u1 = mod_divide(hash_msg(SIGN_MSG), sig.hash, config.n)
    u2 = mod_divide(sig.point_x, sig.hash, config.n)
    R = curve.add_points(
        config.scalar_mult_point(u1),
        curve.scalar_mult(alice_keypair.public, u2))

    assert curve.is_identity(curve.scalar_mult(R, config.n))

    t = mod_inverse(u1 + u2 * eve_secret, config.n)
    G_ = curve.scalar_mult(R, t)
    Q_ = curve.scalar_mult(G_, eve_secret)

    attacker_config = ECDHConfig(curve, G_, config.n)
    assert attacker_config.is_valid()  # Validates point order of G'
    eve_keypair = ECDHKeypair(attacker_config, eve_secret, public=Q_)
    assert eve_keypair.is_valid()  # Validates the public key matches secret

    assert ecdsa_verify(SIGN_MSG, sig, eve_keypair), 'Should have verified'
