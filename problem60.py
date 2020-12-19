from random import randint

import pytest

from numtheory import mod_sqrt, small_factors
from elliptic_curve import (
    EllipticCurvePoint,
    MontgomeryCurve,
    montgomery_ladder,
    montgomery_point_inverse, montgomery_point_test,
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


if __name__ == "__main__":
    assert montgomery_ladder(curve, 4, given_point_order) == 0
    given_group_order = 233970423115425145498902418297807005944
    total_points = 2 * p + 2
    twist_order = total_points - given_group_order
    given_group_order = 233970423115425145498902418297807005944
    # assert montgomery_ladder(curve, )
    print(f'{twist_order} (twist order)')
    print(f'{given_group_order} (curve group order)')
    print(small_factors(twist_order))
    # Sage factorization:
    # 2^2 * 11 * 107 * 197 * 1621 * 105143 * 405373 * 2323367 * 1571528514013

    twist_u = find_twist_point_with_order(curve, 107, twist_order)
    result = montgomery_ladder(curve, twist_u, 107)
    assert result == 0, f'{result} not 0'
