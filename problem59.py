from random import randint
from typing import Tuple

from elliptic_curve import (
    EllipticCurve,
    EllipticCurvePoint,
    add_point,
    find_point_on_curve,
    invert_point,
    scalar_mult_point,
)
from numtheory import small_factors

p = 233970423115425145524320034830162017933
curve = EllipticCurve(p, -95051, 11279326)
point = (182, 85518893674295321206118380980485522083)
assert point in curve, 'Point was not on curve (somehow)'

given_order = 29246302889428143187362802287225875743
pt = scalar_mult_point(point, given_order, curve)
assert pt == curve.identity, 'Point did not have expected order'


def generate_keypair() -> Tuple[int, EllipticCurvePoint]:
    """
    Generate an ECDH keypair.

    Alice and Bob agree to use the given curve and start with a given point on
    a curve.  (Public information)

    Alice and Bob both generate a random integer between 0 and the order of the
    point on the curve. (Secret information)

    Diffie-Hellmen then works as follows:
    Alice keypair: (secret_A, g^secret_A)
    Bob keypair: (secret_B, g^secret_B)

    Alice and Bob then create the shared key:
        (g^secret_A)^(secret_B) = (g^secret_B)^(secret_A)
    """
    secret = randint(0, given_order)
    public = scalar_mult_point(point, secret, curve)

    return (secret, public)


def compute_secret(peer_public: EllipticCurvePoint,
                   self_secret: int):
    """
    Compute the public key ^ secret.
    """
    return scalar_mult_point(peer_public, self_secret, curve)


if __name__ == "__main__":
    (alice_secret, alice_public) = generate_keypair()
    (bob_secret, bob_public) = generate_keypair()

    alice_key = compute_secret(bob_public, alice_secret)
    bob_key = compute_secret(alice_public, bob_secret)

    assert alice_key == bob_key, 'Key should have been shared'

    # Begin Subgroup Confinement Attack
    bad_curve1 = EllipticCurve(p, -95051, 210)
    print(find_point_on_curve(bad_curve1))
    order_curve1 = 233970423115425145550826547352470124412
    # print(small_factors(order_curve1))

    bad_curve2 = EllipticCurve(p, -95051, 504)
    order_curve2 = 233970423115425145544350131142039591210
    # print(small_factors(order_curve2))

    bad_curve3 = EllipticCurve(p, -95051, 727)
    order_curve3 = 233970423115425145545378039958152057148
    # print(small_factors(order_curve3))
