from typing import NamedTuple

from diffie_hellman import ECDHKeypair
from elliptic_curve import EllipticCurvePoint
from numtheory import mod_divide
from rsa import hash_msg

class ECDSASignature(NamedTuple):
    point_x: int
    hash: int


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
