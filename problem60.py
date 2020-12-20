from functools import reduce
from itertools import combinations
from operator import itemgetter
import random
from random import randint
from typing import Tuple, List

import pytest

from diffie_helman import DiffieHelman, ECDHKeypair
from numtheory import crt_inductive, mod_sqrt, small_factors
from elliptic_curve import (
    EllipticCurve, EllipticCurvePoint, MontgomeryCurve,
    montgomery_find_points,
    montgomery_ladder,
    montgomery_point_test,
)

p = 233970423115425145524320034830162017933
curve = MontgomeryCurve(p, 534, 1)

given_point_order = 29246302889428143187362802287225875743
given_group_order = 233970423115425145498902418297807005944


def test_montgomery_find_point():
    given_v = 85518893674295321206118380980485522083
    points = montgomery_find_points(curve, 4)
    assert points[0] == (4, given_v)
    assert points[1] == (4, p - given_v)


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


def find_index(curve: MontgomeryCurve,
               keypair: ECDHKeypair,
               r: int, twist_order: int) -> int:
    bogus_u = find_twist_point_with_order(curve, r, twist_order)
    confused_response = montgomery_ladder(curve, bogus_u, keypair.secret)

    for x in range(1, r):
        current_point = montgomery_ladder(curve, bogus_u, x)
        if current_point == confused_response:
            return x
    else:
        assert False


def subgroup_confinement_residues(curve: MontgomeryCurve,
                                  keypair: ECDHKeypair,
                                  twist_curve_order: int):
    """
    Return the residue of the secret key mod some value
    """
    for r in small_factors(twist_curve_order, max_factor=2**16):
        if r == 2:
            continue

        # Either alice_secret == x mod r OR alice_secret == -x mod r
        # Need to clarify.  (Clarification will be done later through CRT.)
        x = find_index(curve, keypair, r, twist_curve_order)
        yield (x, r)


def filter_moduli(curve: MontgomeryCurve,
                  keypair: ECDHKeypair,
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
        x = find_index(curve, keypair, order, twist_order)
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


def find_generator_point(curve: MontgomeryCurve, given_group_order: int):
    """
    Find a point that generates the group.  This assumes that the group order
    has no small factors (similar to the given group) so most randomly selected
    points will generate the group
    """
    MAX_TRIES = 10
    for _ in range(MAX_TRIES):
        u = randint(1, curve.prime)
        for factor in small_factors(given_group_order - 1):
            if montgomery_ladder(curve, u, factor) == 0:
                # Bad order
                break
        else:
            return montgomery_find_points(curve, u)[0]


def kangaroo_attack(curve: MontgomeryCurve,
                    g: EllipticCurvePoint,  # cyclic group generator
                    point_order: int,  # order of point
                    y: EllipticCurvePoint,  # start point
                    a: int,  # lower bound of y
                    b: int,  # upper bound of y
                    ):
    # Seems like we can just take the first parameter
    k = 20  # Kangaroo parameter

    def pseudorandom_map(u: int):  # u is the first parameter of (u, v)
        return 2 ** (u % k)

    f = pseudorandom_map
    N = 0
    for i in range(0, k):
        N += pseudorandom_map(i)
    N = 4 * (N // k)

    # "Tame Kangaroo"
    print(f'Tame kangaroo start; N={N}')
    xT = 0  # Scalar
    yT = montgomery_ladder(curve, g[0], b)  # Group element
    yExp = b

    for i in range(0, N):
        xT += f(yT)
        yExp += f(yT)
        yT = montgomery_ladder(curve, g[0], yExp)

    assert yT == montgomery_ladder(curve, g[0], b + xT), \
        'Tame kangaroo did not have expected value'

    # Tame kangaroo in place, now we run the wild kangaroo

    print('Wild kangaroo start')
    xW = 0  # Again a scalar
    yW = y[0]  # Again a group element
    yExp = 1
    iterations = 0

    while xW < b - a + xT:
        if iterations % 1000 == 0:
            print(iterations, xW, b - a + xT)
        iterations += 1
        xW += f(yW)
        yExp += f(yW)
        yW = montgomery_ladder(curve, y[0], yExp)

        if yW == yT:
            # Boom
            print(f'Finished in {iterations} iterations')
            return b + xT - xW

    print('All done!')


if __name__ == "__main__":
    random.seed(0)

    point = (4, 85518893674295321206118380980485522083)
    dh = DiffieHelman(curve, point, point_order=given_point_order)

    alice_keypair = dh.generate_keypair()
    bob_keypair = dh.generate_keypair()

    alice_key = alice_keypair.compute_secret(bob_keypair.public)
    bob_key = bob_keypair.compute_secret(alice_keypair.public)

    assert alice_key == bob_key, 'Key should have been shared'

    assert montgomery_ladder(curve, 4, given_point_order) == 0
    given_group_order = 233970423115425145498902418297807005944
    total_points = 2 * p + 2
    twist_order = total_points - given_group_order
    given_group_order = 233970423115425145498902418297807005944
    # assert montgomery_ladder(curve, )
    print(f'{twist_order} (twist order)')
    print(f'{given_group_order} (curve group order)')

    residues = subgroup_confinement_residues(curve, alice_keypair, twist_order)
    option1, option2 = filter_moduli(curve, alice_keypair, twist_order,
                                     list(residues))

    # So these values are actually duplicative since r1 == r2 and m2 = r1 - m1
    m1, r = crt_inductive(option1)
    m2, r_ = crt_inductive(option2)
    assert r == r_
    assert alice_keypair.secret % r in [m1, r - m1]
    assert alice_keypair.secret % r in [m1, m2]
    # So we can forget about m2 for now and just work with m1.
    # So n is either r - m1 or m1
    # x = n + m * r --> n is known, m is unknown

    # We know alice_public is g^alice_secret
    # y = alice_public

    print(f'We know {alice_keypair.secret} = ±{m1} + m * {r}.  Solve for m')
    g = point[0]  # Given starting point
    g_ = montgomery_find_points(curve, montgomery_ladder(curve, g, r))[0]
    g_m1_inverse = montgomery_ladder(curve, g, given_point_order - m1)
    pt = montgomery_find_points(curve, g_m1_inverse)[0]
    # y' = alice_public * g^{-n} --> -n = residue mod r -> also known
    # (n might be r - n in which case y' = alice_public * g^{n})
    y_ = curve.add_points(alice_keypair.public, pt)

    m = kangaroo_attack(curve, point, given_point_order, y_,
                        a=0, b=(given_point_order - 1) // r)
    print(f'Kangaroo attacked returned value {m}')
    assert m * r + m1 == alice_keypair.secret

    # So goal now is to find m such that y_ = (g_)^m
    # This will be done using the kangaroo-style attack (Problem 58).
    # Once we have m, secret = m * r + n
