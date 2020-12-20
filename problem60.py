from functools import reduce
from itertools import combinations
from operator import itemgetter
import random
from random import randint
from typing import Tuple, List

import pytest

from diffie_helman import DiffieHelman
from numtheory import crt_inductive, mod_sqrt, small_factors
from elliptic_curve import (
    EllipticCurve, MontgomeryCurve,
    montgomery_ladder,
    montgomery_point_inverse,
    montgomery_point_test,
)

p = 233970423115425145524320034830162017933
curve = MontgomeryCurve(p, 534, 1)

given_point_order = 29246302889428143187362802287225875743
given_group_order = 233970423115425145498902418297807005944


def test_montgomery_inverse():
    given_v = 85518893674295321206118380980485522083
    assert montgomery_point_inverse(curve, 4) == (given_v, p - given_v)


def test_ladder_attack():
    bogus_u = 76600469441198017145391791613091732004
    assert montgomery_ladder(curve, bogus_u, 11) == 0

    v = (bogus_u ** 3 + 534 * bogus_u ** 2 + bogus_u) % p
    # v is not a quadratic residue mod p
    with pytest.raises(ValueError):
        mod_sqrt(v, p)


def test_twist_order():
    bogus_u = 76600469441198017145391791613091732004
    total_points = 2 * p + 2
    twist_order = total_points - given_group_order

    assert montgomery_ladder(curve, bogus_u, twist_order) == 0
    assert montgomery_ladder(curve, bogus_u, 11) == 0

    bogus_u = find_twist_point_with_order(curve, 107, twist_order)
    result = montgomery_ladder(curve, bogus_u, 107)
    assert result == 0, f'{result} not 0'


def find_twist_point_with_order(curve: MontgomeryCurve,
                                q: int,
                                twist_order: int,
                                ):
    p = curve.prime

    NUM_TRIES = 1_000
    for _ in range(0, NUM_TRIES):
        u = randint(1, p)
        if montgomery_point_test(curve, u):
            continue

        return montgomery_ladder(curve, u, twist_order // q)


def find_power(curve: MontgomeryCurve, r: int, twist_order: int) -> int:
    bogus_u = find_twist_point_with_order(curve, r, twist_order)
    confused_response = montgomery_ladder(curve, bogus_u, alice_secret)

    for x in range(1, r):
        current_point = montgomery_ladder(curve, bogus_u, x)
        if current_point == confused_response:
            return x
    else:
        assert False


def subgroup_confinement_residues(curve: MontgomeryCurve,
                                  twist_curve_order: int):
    """
    Return the residue of the secret key mod some value
    """
    for r in small_factors(twist_curve_order, max_factor=2**16):  # should be 24
        if r == 2:
            continue

        print(f'Factor {r}')

        # Either alice_secret == x mod r OR alice_secret == -x mod r
        # Need to clarify.  (Clarification will be done later through CRT.)
        x = find_power(curve, r, twist_curve_order)
        yield (x, r)


def filter_moduli(curve: MontgomeryCurve,
                  twist_order: int,
                  residue_list: List[Tuple[int, int]]):
    residues = map(itemgetter(1), residue_list)
    x, r = residue_list[0]
    residue_possibilities = [[(x, r)], [(r - x, r)]]
    for x, r in residue_list[1:]:
        # for each element of possibilities, add (x, r) and (x, -r) to them
        new_possibilities = []
        for possibility in residue_possibilities:
            new_possibilities.append(possibility + [(x, r)])
            new_possibilities.append(possibility + [(r - x, r)])

        residue_possibilities = new_possibilities

    print(f'{len(residue_possibilities)} possibilties now')
    # Now we try to winnow down the possibilities by getting pairwise CRT
    # residues
    for r1, r2 in combinations(residues, 2):
        order = r1 * r2
        x = find_power(curve, order, twist_order)
        print(f'{x} mod {order} OR {order - x} mod {order}')
        # Now elimate possibilities that don't match this
        new_possibilities = []
        for possibility in residue_possibilities:
            r1_value = next(filter(lambda x: x[1] == r1, possibility))[0]
            r2_value = next(filter(lambda x: x[1] == r2, possibility))[0]

            matches_x = x % r1 == r1_value and x % r2 == r2_value
            matches_minus_x = (
                (order - x) % r1 == r1_value
                and (order - x) % r2 == r2_value)

            if matches_x or matches_minus_x:
                new_possibilities.append(possibility)

        residue_possibilities = new_possibilities

        if len(residue_possibilities) == 2:
            break

    return residue_possibilities


if __name__ == "__main__":
    point = (4, 85518893674295321206118380980485522083)
    given_order = 29246302889428143187362802287225875743
    dh = DiffieHelman(curve, point)
    random.seed(0)
    (alice_secret, alice_public) = dh.generate_keypair(given_order)
    (bob_secret, bob_public) = dh.generate_keypair(given_order)

    alice_key = dh.compute_secret(bob_public, alice_secret)
    bob_key = dh.compute_secret(alice_public, bob_secret)

    assert alice_key == bob_key, 'Key should have been shared'

    assert montgomery_ladder(curve, 4, given_point_order) == 0
    given_group_order = 233970423115425145498902418297807005944
    total_points = 2 * p + 2
    twist_order = total_points - given_group_order
    given_group_order = 233970423115425145498902418297807005944
    # assert montgomery_ladder(curve, )
    print(f'{twist_order} (twist order)')
    print(f'{given_group_order} (curve group order)')

    residue_list = list(subgroup_confinement_residues(curve, twist_order))
    option1, option2 = filter_moduli(curve, twist_order, residue_list)

    # So these values are actually duplicative since r1 == r2 and m2 = r1 - m1
    m1, r1 = crt_inductive(option1)
    m2, r2 = crt_inductive(option2)

    assert alice_secret % r1 == m1 or alice_secret % r2 == m2

    # We need to kangaroo attack from m1, r2 and m2, r2

