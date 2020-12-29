from math import isqrt, prod, gcd
from random import randint, randrange
from typing import Callable, Generator, List, Optional, Tuple

import gmpy2
import pytest

MAX_TRIES = 1_000


# g is some polynomial, we'll just hardcode it to x^2 + 1 mod n
def pollard_rho(n, g: Optional[Callable[[int], int]]=None) -> Optional[int]:
    if not g:
        g = lambda x: (x * x + 1) % n

    x = 2
    y = 2
    d = 1
    while d == 1:
        x = g(x)
        y = g(g(y))
        d = gcd(abs(x - y), n)

    if d == n:
        return None

    return d


def test_pollard_rho():
    assert pollard_rho(53) == None
    assert pollard_rho(26) in [2, 13]


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
        if x % d != x_ % d:
            raise ValueError('No unique solution')
        x = (x * v * m_ + x_ * u * m) // d
        m = m * m_
        x = x % m

    return x, m


def test_crt_inductive():
    assert crt_inductive([(0, 3), (3, 4), (4, 5)]) == (39, 60)
    with pytest.raises(ValueError):
        crt_inductive([(2, 4), (1, 6)])
    assert crt_inductive([(2, 4), (0, 6)]) == (6, 24)


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


def random_prime(e: Optional[int] = None,
                 range_start=2**120,
                 range_end=2**121,
                 ) -> int:
    for _ in range(0, MAX_TRIES):
        p = randrange(range_start, range_end)
        if not gmpy2.is_prime(p):
            continue

        if not e or (p - 1) % e != 0:
                return p

    raise ValueError(f'Could not find random prime with totient prime to {e}')


def random_smooth_prime(b=2**16, range_start=None, range_end=None):
    for _ in range(0, MAX_TRIES):
        kwargs = {'range_start': b, 'range_end': b * 2}
        if range_start:
            kwargs['range_start'] = range_start
        if range_end:
            kwargs['range_end'] = range_end

        p = random_prime(**kwargs)

        # This is pretty inefficient but should be fine since we don't need p
        # to be that large
        start = p - 1
        factors = list(small_factors(p - 1, max_factor=range_end or b * 2))
        for f in factors:
            while start % f == 0:
                start //= f

        if start == 1:
            return p

    raise ValueError(f'Could not find smooth prime')


def is_primitive_root(g: int, p: int) -> bool:
    factors = list(small_factors(p - 1, max_factor=p))
    # Ensure factor list is comprehensive
    start = p - 1
    for f in factors:
        while start % f == 0:
            start //= f
    assert start == 1

    for f in factors:
        if pow(g, (p - 1) // f, p) == 1:
            return False

    return True


def test_random_smooth_prime():
    assert random_smooth_prime() > 0


def test_is_primitive_root():
    assert is_primitive_root(5, 23)
    assert is_primitive_root(3, 34)
    assert is_primitive_root(6, 23) == False
    assert is_primitive_root(2, 83477)


def prime_power_factor(n: int, p: int) -> int:
    x = 0
    while n % p == 0:
        x += 1
        n = n // p

    return x


def test_prime_power_factor():
    assert prime_power_factor(8, 2) == 3


def discrete_log(g: int, b: int, n: int, order_g=None):
    """
    Return a so that b = g^a mod n

    This is Shanks "baby step, giant step" algorithm
    """
    if not order_g:
        order_g = n
    m = isqrt(order_g)
    if m * m < order_g:
        m += 1

    table = {pow(g, j, n): j for j in range(m - 1, -1, -1)}
    inverse_g = pow(g, -m, n)
    x = b

    for i in range(0, m):
        if x in table:
            return i * m + table[x]
        x = (x * inverse_g) % n
    else:
        raise ValueError('Discrete logarithm unsolvable')


def test_discrete_log():
    assert discrete_log(5, 20, 47) == 37
    assert discrete_log(7, 8458730, 18989249) == 8912894


def pohlig_hellman_subgroup(g: int, h: int, r: int, e: int, p: int):
    """
    Solve the discrete log problem g^x = h mod p

    g is the generator of a subgroup of prime power order r^e
    """
    def powers(ele, order):
        for i in range(order):
            yield pow(ele, i, p)

    x_ = 0
    gam = pow(g, pow(r, e - 1), p)
    assert pow(gam, r, p) == 1  # Has order r (prime)

    for k in range(0, e):
        h_ = pow(pow(g, -x_, p) * h, pow(r, e - 1 - k), p)
        d_ = discrete_log(gam, h_, p)
        assert pow(gam, d_, p) == h_, 'Discrete log incorrect'
        x_ = (x_ + pow(r, k, p) * d_) % p

    return x_


def pohlig_hellman(g: int, h: int, p: int):
    residue_list = []
    n = p - 1
    assert pow(g, n, p) == 1

    for r in small_factors(n):
        e = prime_power_factor(n, r)
        p_ = pow(r, e)

        g_ = pow(g, n // p_, p)
        assert pow(g_, p_, p) == 1
        h_ = pow(h, n // p_, p)

        # Subgroup of order p_ exists by Sylow theorems
        x_ = pohlig_hellman_subgroup(g_, h_, r, e, p)
        assert pow(g_, x_, p) == h_

        residue_list.append((x_, p_))

    x = crt_inductive(residue_list)[0]
    assert pow(g, x, p) == h % p, 'Did not solve discrete logarithm correctly'
    return x


def test_pohlig_hellman():
    assert pohlig_hellman_subgroup(5, 3, 2, 4, 2**4 + 1) == 13
    assert pohlig_hellman_subgroup(5, 3, 2, 16, 2**16 + 1) == 27659
    assert pohlig_hellman(6, 7531, 8101) == 6689
