from random import randint
from typing import Tuple
from functools import reduce
from operator import itemgetter, mul

from diffie_hellman import DiffieHellman, ECDHKeypair
from elliptic_curve import (
    EllipticCurveIdentity,
    WeierstrassCurve,
)
from numtheory import small_factors, crt_inductive

p = 233970423115425145524320034830162017933
given_curve = WeierstrassCurve(p, -95051, 11279326)
point = (182, 85518893674295321206118380980485522083)
assert point in given_curve, 'Point was not on curve (somehow)'

given_order = 29246302889428143187362802287225875743
pt = given_curve.scalar_mult(point, given_order)
assert pt == EllipticCurveIdentity, 'Point did not have expected order'


def subgroup_confinement_residues(bob_keypair: ECDHKeypair,
                                  curve: WeierstrassCurve,
                                  curve_order: int):
    """
    Return the residue of the secret key mod some value
    """
    for r in small_factors(curve_order):
        if r == 2:
            # Seem to have nothing but bugs for r = 2
            continue

        try:
            bogus_point = curve.find_point_of_order(r, curve_order)
            confused_response = bob_keypair.compute_secret(bogus_point)
            # brute for which r for confused response
            current_point = bogus_point
            for x in range(1, r):
                if current_point == confused_response:
                    yield (x, r)
                    break
                current_point = curve.add_points(current_point, bogus_point)
        except ValueError as err:
            continue


if __name__ == "__main__":
    dh = DiffieHellman(given_curve, point, point_order=given_order)
    alice_keypair = dh.generate_keypair()
    bob_keypair = dh.generate_keypair()

    alice_key = alice_keypair.compute_secret(bob_keypair.public)
    bob_key = bob_keypair.compute_secret(alice_keypair.public)

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
        crt_residues += list(subgroup_confinement_residues(bob_keypair,
                                                           bad_curve,
                                                           bad_curve_order))
        # Must remove duplicates because the given bad curves might end up
        # with the same residues.  In the event of a bug where we have
        # different residue values mod the same prime the algorithm will
        # still explode.
        crt_residues = list(set(crt_residues))
        q = reduce(mul, map(itemgetter(1), crt_residues), 1)
        if q > given_order:
            break

    for m, r in crt_residues:
        assert bob_keypair.secret % r == m

    # Ready to attack
    x, _ = crt_inductive(crt_residues)

    assert x == bob_keypair.secret, 'Brute forced secret with bogus points'
    print(f'All done!  Used {len(crt_residues)} residues to determine '
          f'{x} == {bob_keypair.secret}')
