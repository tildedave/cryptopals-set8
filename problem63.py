from typing import Optional, Tuple

GF2Polynomial = int

def polynomial_add(p1: GF2Polynomial, p2: GF2Polynomial):
    return p1 ^ p2


# GF2 is weird
polynomial_subtract = polynomial_add


def polynomial_degree(p1: GF2Polynomial):
    return p1.bit_length() - 1


def polynomial_string(p: GF2Polynomial):
    if p == 0:
        return '0'

    s = []
    for i in range(0, polynomial_degree(p) + 1):
        s.append(p & 1)
        p >>= 1

    parts = []
    for i, b in enumerate(s):
        if b:
            if i == 0:
                parts.insert(0, '1')
            elif i == 1:
                parts.insert(0, 'x')
            else:
                parts.insert(0, f'x^{i}')

    return ' + '.join(parts)


def polynomial_mult(a: GF2Polynomial,
                    b: GF2Polynomial,
                    mod: Optional[GF2Polynomial] = None,
                    ) -> GF2Polynomial:
    p = 0
    while a > 0:
        if a & 1:
            p ^= b
        a >>= 1
        b <<= 1

        if mod is not None and polynomial_degree(b) == polynomial_degree(mod):
            b ^= mod

    return p


def polynomial_divmod(a: GF2Polynomial,
                      b: GF2Polynomial,
                      ) -> Tuple[GF2Polynomial, GF2Polynomial]:
    """
    Returns (a // b, a % b)
    """
    q, r = 0, a

    while polynomial_degree(r) >= polynomial_degree(b):
        d = polynomial_degree(r) - polynomial_degree(b)
        q = q ^ (1 << d)
        r = r ^ (b << d)

    return q, r


def polynomial_egcd(a: GF2Polynomial,
                    b: GF2Polynomial,
                    mod: Optional[GF2Polynomial] = None):
    """
    Return (d, u, v) so that a * u + b * v = d
    """
    if a == 0:
        return (b, 0, 1)
    else:
        q, r = polynomial_divmod(b, a)
        # so now q * b + r == a
        assert polynomial_add(polynomial_mult(q, a), r) == b

        g, x, y = polynomial_egcd(r, a, mod)
        assert polynomial_add(
            polynomial_mult(x, r),
            polynomial_mult(y, a)
        ) == g, f'WRONG: gcd({polynomial_string(r)}, {polynomial_string(a)}) = ({polynomial_string(g)}, {polynomial_string(x)}, {polynomial_string(y)})'

        return (g, polynomial_subtract(y, polynomial_mult(q, x, mod)), x)


def polynomial_inverse(a: GF2Polynomial,
                       m: GF2Polynomial,
                       ) -> GF2Polynomial:
    g, x, _ = polynomial_egcd(a, m)
    if g != 1:
        raise ValueError(f'{polynomial_string(a)} was not invertible')

    return divmod(x, m)[1]


def polynomial_mod_exp(a: GF2Polynomial,
                       n: int,
                       m: GF2Polynomial):
    p = 1
    while n > 0:
        if n % 2 == 1:
            p = polynomial_mult(p, a, m)
        a = polynomial_mult(a, a, m)
        n = n // 2

    return p


def test_polynomial_mult():
    # (x^2 + x + 1) * (x + 1) == x^3 + 1
    assert polynomial_mult(7, 3, 2**16) == 9
    assert polynomial_mult(7, 3, 8) == 1


def test_polynomial_degree():
    # 1 = 1
    assert polynomial_degree(1) == 0
    # 2 = x
    assert polynomial_degree(2) == 1
    # 3 = x + 1
    assert polynomial_degree(3) == 1


def test_polynomial_divmod():
    assert polynomial_divmod(9, 7) == (3, 0)
    assert polynomial_divmod(7, 15) == (0, 7)


def test_polynomial_gcd():
    # xgcd(x^3 + x + 1, x) = (1, 1, x^2 + 1)
    assert polynomial_egcd(11, 2) == (1, 1, 5)

    # xgcd(x, x^3 + x + 1) = (1, x^2 + 1, 1)
    assert polynomial_egcd(2, 11) == (1, 5, 1)

    # xgcd(x^3 + x + 1, x^2 + x + 1) == (1, x + 1, x^2)
    assert polynomial_egcd(11, 7) == (1, 3, 4)

    # gcd(x^3 + 1, x^2 + x + 1) = (x^2 + x + 1, 0, 1)
    assert polynomial_egcd(9, 7) == (7, 0, 1)


def test_polynomial_inverse():
    # Irreducible: x^4 + x + 1
    mod = 2**4 + 2**1 + 2**0

    p = 2**3 + 1
    inv = polynomial_inverse(p, mod)
    assert polynomial_mult(p, inv, mod) == 1


def test_polynomial_modexp():
    p = 2**3 + 1
    mod = 2**6 + 2**1 + 1
    # Sage - ((x^3 + 1)^3) % (x^6 + x + 1)
    assert polynomial_mod_exp(p, 3, mod)) == 2**4 + 2**1
