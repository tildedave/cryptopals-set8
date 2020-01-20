from typing import Tuple, List


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


def mod_exp(m, n, p):
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

    return x
