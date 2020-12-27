from random import randint, randrange
from typing import Generator, Tuple, List

import gmpy2
import pytest

MAX_TRIES = 1_000


def small_factors(j, max_factor=2**16) -> Generator[int, None, None]:
    """
    Return all factors of a passed in integer < 2 ** 16
    """
    for i in range(2, max_factor):
        if j % i == 0:
            yield i
            while j % i == 0:
                j = j // i
        if i > j:
            break


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
        if pow(n, (p - 1) // 2, p) != 1:
            break

    # Step 2 - initialize
    z = pow(n, q, p)
    y = z
    r = e
    x = pow(a, (q - 1) // 2, p)
    b = (a * x * x) % p
    x = (a * x) % p
    while True:
        # Step 3 - Find exponent
        if b % p == 1:
            return x

        # find smallest m such that b^2^m = 1 (p)
        # if m = r output that a is a non-residue
        m = 1
        while pow(b, 2 ** m, p) != 1:
            m += 1

        if m == r:
            raise ValueError(f'{a} was not a quadratic residue mod {p}')

        # Step 4 - reduce exponent
        t = pow(y, 2 ** (r - m - 1), p)
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


def mod_inverse(a, m):
    return pow(a, -1, m)


def test_mod_inverse():
    assert mod_inverse(5, 7) == 3
    assert mod_inverse(17, 3120) == 2753


def mod_divide(m: int, n: int, p: int):
    """
    Return m / n mod p - probably won't work for p = 2
    """
    if n == 1:
        return m

    return (m * pow(n, p - 2, p)) % p


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


def kronecker_symbol(a: int, b: int):
    tab2 = [0, 1, 0, -1, 0, -1, 0, 1]  # (-1)^((n^2 - 1) / 8)
    # Step 1
    if b == 0:
        if a != 1:
            return 0
        return 1

    # Step 2
    # if a and b are both even, output 0 and terminate
    if a % 2 == b % 2 == 0:
        return 0

    v = 0
    while b % 2 == 0:
        v += 1
        b = b // 2

    if v % 2 == 0:
        k = 1
    else:
        k = tab2[a & 7]  # (-1)^((a^2 - 1)/8)

    if b < 0:
        b = -b
        if a < 0:
            k = -k

    # Step 3
    while True:
        assert b % 2 == 1, 'b should be odd'
        assert b > 0, 'b > 3'

        if a == 0:
            if b > 1:
                return 0
            if b == 1:
                return k

        v = 0
        while a % 2 == 0:
            v += 1
            a = a // 2

        if v % 2 == 1:
            k = k * tab2[b & 7]  # (-1)^((a^2 - 1)/8)

        # Step 4
        if a & b & 2:
            k = -k

        r = abs(a)
        a = b % r
        b = r


def test_kronecker_symbol():
    assert -1 == kronecker_symbol(-1, 7)
    assert 1 == kronecker_symbol(-1, 13)
    assert -1 == kronecker_symbol(2, 3)
    assert -1 == kronecker_symbol(2, 5)
    assert -1 == kronecker_symbol(2, 11)
    assert -1 == kronecker_symbol(2, 13)
    assert 1 == kronecker_symbol(2, 7)
    assert 1 == kronecker_symbol(2, 17)
    assert 1 == kronecker_symbol(2, 23)
    assert 1 == kronecker_symbol(36881633580730759730498811283302146613,
                                 233970423115425145524320034830162017933)


def random_prime(e, range_start=2**120, range_end=2**121) -> int:
    for _ in range(0, MAX_TRIES):
        p = randrange(range_start, range_end)
        if not gmpy2.is_prime(p):
            continue

        if (p - 1) % e != 0:
            return p

    raise ValueError(f'Could not find random prime with totient prime to {e}')
