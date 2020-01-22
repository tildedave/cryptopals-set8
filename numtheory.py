from random import randint
from typing import Tuple, List

import pytest


def small_factors(j):
    """
    Return all factors of a passed in integer < 2 ** 16
    """
    factors = []
    for i in range(2, 2**16):
        if j % i == 0:
            factors.append(i)
            while j % i == 0:
                j = j // i
        if i > j:
            break

    return factors


def mod_exp(m: int, n: int, p: int):
    """
    Return m^n % p
    """
    y = 1
    if n == 0:
        return y

    if n < 0:
        raise ValueError('Not implemented yet')

    z = m
    while n > 0:
        if n % 2 == 1:
            y = (z * y) % p
        n = n // 2
        z = (z * z) % p

    return y % p


def mod_sqrt(a: int, p: int):
    """
    Tonelli/Shanks Algorithm for modular square root
    """
    e = 0
    q = p - 1
    while q % 2 == 0:
        e += 1
        q = q // 2
    assert q % 2 == 1, 'Reduced value was not odd'

    # Step 1 - find generator
    while True:
        n = randint(2, p)
        # Dumb way to make sure this isn't a quadratic residue
        # (Could implement this better)
        if mod_exp(n, (p - 1) // 2, p) != 1:
            break

    # Step 2 - initialize
    z = mod_exp(n, q, p)
    y = z
    r = e
    x = mod_exp(a, (q - 1) // 2, p)
    b = (a * x * x) % p
    x = (a * x) % p
    while True:
        # Step 3 - Find exponent
        if b % p == 1:
            return x

        # find smallest m such that b^2^m = 1 (p)
        # if m = r output that a is a non-residue
        m = 1
        while mod_exp(b, 2 ** m, p) != 1:
            m += 1

        if m == r:
            raise ValueError(f'{a} was not a quadratic residue mod {p}')

        # Step 4 - reduce exponent
        t = mod_exp(y, 2 ** (r - m - 1), p)
        y = t * t % p
        r = m
        x = x * t % p
        b = b * y % p


def test_mod_sqrt():
    assert mod_sqrt(4, 7) in [2, 5]
    assert mod_sqrt(2, 7) in [3, 4]
    assert mod_sqrt(58, 101) in [82, 19]
    with pytest.raises(ValueError) as excinfo:
        mod_sqrt(5, 7)
    assert "not a quadratic residue" in str(excinfo.value)


def mod_inverse(m: int, p: int):
    # This works for p = 3 but maybe not p = 2
    return mod_exp(m, p - 2, p)


def test_mod_inverse():
    assert mod_inverse(5, 7) == 3


def mod_divide(m: int, n: int, p: int):
    """
    Return m / n mod p - probably won't work for p = 2
    """
    if n == 1:
        return m

    return (m * mod_inverse(n, p)) % p


def euclid_extended(a: int, b: int):
    """
    Return (u, v, d) such that ua + vb = d and d = (a, b)
    """
    u = 1
    d = a
    if b == 0:
        v = 0
        return (u, v, d)
    v1 = 0
    v3 = b
    while True:
        if v3 == 0:
            v = (d - a*u) // b
            return (u, v, d)
        q = d // v3
        t3 = d % v3
        t1 = u - q * v1
        u = v1
        d = v3
        v1 = t1
        v3 = t3


def crt_inductive(residue_list: List[Tuple[int, int]]):
    assert len(residue_list) > 0, 'List must be non-empty'
    x, m = residue_list[0]
    for x_, m_ in residue_list[1:]:
        u, v, d = euclid_extended(m, m_)
        assert d == 1, 'Residues were not co-prime'
        x = u * m * x_ + v * m_ * x
        m = m * m_
        x = x % m

    return x, m
