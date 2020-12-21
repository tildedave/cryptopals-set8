from itertools import combinations
from operator import itemgetter
import random
from random import randint
import time
from typing import Callable, Optional, Sequence, Tuple, List, Union

import pytest

from diffie_helman import DiffieHelman, ECDHKeypair
from numtheory import crt_inductive, mod_sqrt, small_factors
from elliptic_curve import (
    EllipticCurvePoint,
    MontgomeryCurve, WeierstrassCurve,
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
               r: int,
               twist_order: int,
               order_indices: Optional[Sequence[int]] = None) -> int:
    bogus_u = find_twist_point_with_order(curve, r, twist_order)
    confused_response = montgomery_ladder(curve, bogus_u, keypair.secret)
    if not order_indices:
        order_indices = range(1, r)

    for x in order_indices:
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
    for r in small_factors(twist_curve_order, max_factor=2**24):
        start = time.time()
        if r == 2:
            continue

        # Either alice_secret == x mod r OR alice_secret == -x mod r
        # Need to clarify.  (Clarification will be done later through CRT.)
        x = find_index(curve, keypair, r, twist_curve_order)
        print(f'Secret ≡ {x} mod {r} (duration {time.time() - start})')
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
    for r1, r2 in sorted(combinations(residues, 2), key=lambda x: x[0] * x[1]):
        order = r1 * r2
        m1 = next(filter(lambda x: x[1] == r1, residue_list))[0]
        m2 = next(filter(lambda x: x[1] == r2, residue_list))[0]

        def test_func(x):
            """
            Only test indices that match our known CRT residues
            """
            return x % r1 in [m1, r1 - m1] and x % r2 in [m2, r2 - m2]

        x = find_index(curve, keypair, order, twist_order,
                       filter(test_func, range(1, order)))
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

        print(f'{x} mod {order} OR {order - x} mod {order} '
              f'(now {len(new_possibilities)} possibilities)')
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

    # u is the first parameter of (u, v)
    def pseudorandom_map(u: Union[int, EllipticCurvePoint]):
        if not isinstance(u, int):
            u = u[0]
        return 2 ** (u % k)

    f = pseudorandom_map
    precomputed_map = {f(x): curve.scalar_mult(g, f(x)) for x in range(0, k)}
    for x in range(0, k):
        assert montgomery_ladder(curve, g[0], f(x)) == precomputed_map[f(x)][0]

    N = 0
    for i in range(0, k):
        N += pseudorandom_map(i)
    N = 4 * (N // k)

    # "Tame Kangaroo"
    tame_start = time.time()
    print(f'Tame kangaroo start; N={N}')
    xT = 0  # Scalar
    yT = curve.scalar_mult(g, b)  # montgomery_ladder(curve, g[0], b)  # Group element

    yExp = b

    last = time.time()
    for i in range(0, N):
        if i % 10_000 == 0:
            print(i, time.time() - last)
            last = time.time()

        print(i, xT, yT)

        old_xT = xT
        xT += f(yT)
        yExp = (yExp + f(yT)) % point_order
        # yT = montgomery_ladder(curve, g[0], yExp)
        # Up until now yT has order b + old_xT
        assert montgomery_ladder(curve, g[0], b + old_xT) == yT[0]
        assert montgomery_ladder(curve, g[0], f(yT)) == precomputed_map[f(yT)][0]

        yT = curve.add_points(yT, precomputed_map[f(yT)])
        print(f'xT {xT} yT {yT} {f(yT)} ',
              f'expected: {montgomery_ladder(curve, g[0], b + xT)} ',
              f'actual: {montgomery_ladder(curve, g[0], yExp)}')
        assert yT[0] == montgomery_ladder(curve, g[0], b + xT), \
            'Tame kangaroo did not have expected value'
    print(f'Time kangaroo complete; time={time.time() - tame_start}')
    # Tame kangaroo in place, now we run the wild kangaroo

    print('Wild kangaroo start')
    xW = 0  # Again a scalar
    yW = y  # [0]  # Again a group element
    iterations = 0

    last_time = time.time()
    while xW < b - a + xT:
        if iterations % 10_000 == 0:
            print(iterations, xW, b - a + xT, time.time() - last_time)
            last_time = time.time()

        iterations += 1
        xW += f(yW)
        yW = curve.add_points(yW, precomputed_map[f(yW)])
        # yExp = (yExp + f(yW)) % point_order
        # yW = montgomery_ladder(curve, y[0], yExp)

        if yW == yT:
            # Boom
            print(f'Finished in {iterations} iterations')
            return b + xT - xW

    print('All done!')


if __name__ == "__main__":
    random.seed(0)

    calculate_residues = False  # ~3 minutes or so

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

    if calculate_residues:
        residues = subgroup_confinement_residues(curve, alice_keypair,
                                                 twist_order)
        residues_list = list(residues)
    else:
        residues_list = [
            (4, 11), (46, 107), (15, 197), (721, 1621),
            (36413, 105143), (140928, 405373), (3842, 2323367)]

    print(f'Residues: {residues_list}')
    option1, option2 = filter_moduli(curve, alice_keypair, twist_order,
                                     residues_list)

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
