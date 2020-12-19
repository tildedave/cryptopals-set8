from random import randint
from typing import Tuple
from functools import reduce
from operator import itemgetter, mul

from elliptic_curve import (
    EllipticCurveIdentity,
    EllipticCurvePoint,
    WeierstrassCurve,
    add_point,
    find_point_on_curve,
)
from numtheory import small_factors, crt_inductive

p = 233970423115425145524320034830162017933
curve = WeierstrassCurve(p, -95051, 11279326)
point = (182, 85518893674295321206118380980485522083)
assert point in curve, 'Point was not on curve (somehow)'

given_order = 29246302889428143187362802287225875743
pt = curve.scalar_mult(point, given_order)
assert pt == EllipticCurveIdentity, 'Point did not have expected order'


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
    public = curve.scalar_mult(point, secret)

    return (secret, public)


def compute_secret(peer_public: EllipticCurvePoint,
                   self_secret: int) -> EllipticCurvePoint:
    """
    Compute the public key ^ secret.
    """
    return curve.scalar_mult(peer_public, self_secret)


def find_point_of_order(r: int, curve_order: int, curve: WeierstrassCurve):
    tries = 0
    while tries < 10:
        tries += 1
        pt = find_point_on_curve(curve)
        candidate_point = curve.scalar_mult(pt, curve_order // r)
        if candidate_point != EllipticCurveIdentity:
            assert curve.scalar_mult(candidate_point, r) == EllipticCurveIdentity
            return candidate_point

    raise ValueError(f'Unable to find point of order {r} on curve {curve}')


def subgroup_confinement_residues(curve: WeierstrassCurve,
                                  curve_order: int):
    """
    Return the residue of the secret key mod some value
    """
    for r in small_factors(curve_order):
        if r == 2:
            # Seem to have nothing but bugs for r = 2
            continue

        try:
            bogus_point = find_point_of_order(r, curve_order, curve)
            confused_response = compute_secret(bogus_point, bob_secret)
            # brute for which r for confused response
            current_point = bogus_point
            for x in range(1, r):
                if current_point == confused_response:
                    yield (x, r)
                    break
                current_point = add_point(current_point, bogus_point, curve)
        except ValueError as err:
            continue


if __name__ == "__main__":
    (alice_secret, alice_public) = generate_keypair()
    (bob_secret, bob_public) = generate_keypair()

    alice_key = compute_secret(bob_public, alice_secret)
    bob_key = compute_secret(alice_public, bob_secret)

    assert alice_key == bob_key, 'Key should have been shared'

    # Begin Subgroup Confinement Attack
    bad_curve1 = WeierstrassCurve(p, -95051, 210)
    order_curve1 = 233970423115425145550826547352470124412
    bad_curve2 = WeierstrassCurve(p, -95051, 504)
    order_curve2 = 233970423115425145544350131142039591210
    bad_curve3 = WeierstrassCurve(p, -95051, 727)
    order_curve3 = 233970423115425145545378039958152057148
    bad_curves = [bad_curve1, bad_curve2, bad_curve3]
    curve_orders = [order_curve1, order_curve2, order_curve3]

    crt_residues = []
    for bad_curve, bad_curve_order in zip(bad_curves, curve_orders):
        print(f'Computing residues for {bad_curve}')
        crt_residues += list(subgroup_confinement_residues(bad_curve,
                                                           bad_curve_order))
        # Must remove duplicates because the given bad curves might end up
        # with the same residues.  In the event of a bug where we have
        # different residue values mod the same prime the algorithm will
        # still explode.
        crt_residues = list(set(crt_residues))
        q = reduce(mul, map(itemgetter(1), crt_residues), 1)
        if q > given_order:
            break

    # Ready to attack
    x, _ = crt_inductive(crt_residues)
    assert x == bob_secret, 'Brute forced secret with bogus points'
    print(f'All done!  Used {len(crt_residues)} residues to determine {x} == {bob_secret}')
