from copy import copy
from fractions import Fraction
from operator import itemgetter
import random
import string
from typing import Iterable, List, Tuple, Union

from diffie_hellman import DiffieHellman
from ecdsa import ECDHKeypair, ECDSASignature, hash_msg
from elliptic_curve import WeierstrassCurve
from numtheory import mod_divide, mod_inverse


def create_dh() -> DiffieHellman:
    p = 233970423115425145524320034830162017933
    curve = WeierstrassCurve(p, -95051, 11279326)
    point = (182, 85518893674295321206118380980485522083)
    point_order = 29246302889428143187362802287225875743

    return DiffieHellman(curve, point, point_order)


def ecdsa_sign_biased(msg: str, keypair: ECDHKeypair):
    n = keypair.config.n
    k = keypair.config.rand_scalar()
    k = (k >> 8) << 8  # Mask off the last 8 bits as suggested
    r = keypair.config.scalar_mult_point(k)[0]
    s = mod_divide(hash_msg(msg) + keypair.secret * r, k, n)

    return ECDSASignature(r, s)


Scalar = Union[int, Fraction, float]
Vector = List[Scalar]


def scalar_product(u: Vector, a: Scalar) -> Vector:
    return [a * u[i] for i in range(0, len(u))]


def inner_product(u: Vector, v: Vector) -> Scalar:
    assert len(u) == len(v)

    return sum(u[i] * v[i] for i in range(0, len(u)))


def test_inner_product():
    assert inner_product([1, 2], [2, 3]) == 8


def test_scalar_product():
    assert scalar_product([1, 2], 3) == [3, 6]


def proj(u: Vector, v: Vector) -> Vector:
    if all(x == 0 for x in u):
        return [Fraction(0)] * len(u)

    return scalar_product(u, inner_product(v, u) / inner_product(u, u))


def vec_sum(vecs: Iterable[Vector]):
    assert len(vecs) > 0, 'Cannot sum over empty vectors'
    q = vecs[0]
    for vec in vecs[1:]:
        assert len(vec) == len(q)
        q = [q[i] + vec[i] for i in range(0, len(q))]

    return q


def vec_subtract(vec: Vector, *vecs: Vector):
    q = copy(vec)
    for v in vecs:
        assert len(v) == len(vec)
        q = [q[i] - v[i] for i in range(0, len(vec))]

    return q


def graham_schmidt(b: List[Vector]) -> List[Vector]:
    q = [Fraction(0)] * len(b)
    for i, v in enumerate(b):
        sum_list = [[Fraction(0)] * len(v)] + [proj(u, v) for u in q[:i]]
        q[i] = vec_subtract(v, vec_sum(sum_list))

    return q

def test_graham_schmidt():
    # https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Example
    result = graham_schmidt([[3, 1], [2, 2]])
    assert result == [[3, 1], [Fraction(-2, 5), Fraction(6, 5)]]
    u1, u2 = result
    assert inner_product(u1, u2) == 0


def lll_reduction(b: List[Vector], delta: Fraction=Fraction(99, 100)):
    b = copy(b)
    q = graham_schmidt(b)

    def mu(i: int, j: int) -> Scalar:
        v = b[i]
        u = q[j]
        ret = Fraction(inner_product(v, u), inner_product(u, u))
        return ret

    n = len(b)
    k = 1

    while k < n:
        print('k', k, n)
        for j in range(k - 1, -1, -1):
            m = mu(k, j)
            if abs(m) > Fraction(1, 2):
                b[k] = vec_subtract(b[k], scalar_product(b[j], round(m)))
                q = graham_schmidt(b)

        threshold = (delta - mu(k, k - 1)**2) * inner_product(q[k -1], q[k - 1])
        if inner_product(q[k], q[k]) >= threshold:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            q = graham_schmidt(b)
            k = max(k - 1, 1)

    return b


def test_lll_reduction():
    test_basis = [
        [  -2,    0,    2,    0],
        [ Fraction(1, 2),   -1,    0,    0],
        [  -1,    0,   -2,  Fraction(1, 2)],
        [  -1,    1,    1,    2],
    ]
    expected_result = [
        [ 1/2, -1,  0,    0],
        [  -1,  0, -2,  1/2],
        [-1/2,  0,  1,    2],
        [-3/2, -1,  2,    0],
    ]

    assert lll_reduction(test_basis) == expected_result


def test_lll_reduction_minimum_polynomial():
    # this is sqrt(2) + sqrt(3)
    # approximations in the ith row is precision * alpha^{i}
    test_basis = [
        [1, 0, 0, 0, 0, 10000],
        [0, 1, 0, 0, 0, 31462],
        [0, 0, 1, 0, 0, 98989],
        [0, 0, 0, 1, 0, 311448],
        [0, 0, 0, 0, 1, 979897]
    ]
    # corresponds with sqrt(2) + sqrt(3) minpoly being x^4 - 10x^2 + 1
    assert lll_reduction(test_basis)[0][0:5] == [1, 0, -10, 0, 1]


def test_lll_against_biased_nonce_attack():
    dh = create_dh()
    chars = string.ascii_lowercase + string.digits
    alice_keypair = dh.generate_keypair()

    num_signatures = 30
    message_length = 40
    l = 8  # Number of biased bits
    q = dh.config.n

    attacker_pairs : List[Tuple[int, int]] = []
    for _ in range(num_signatures):
        msg = ''.join(random.choices(chars, k=message_length))
        sig = ecdsa_sign_biased(msg, alice_keypair)

        r, s = sig.r, sig.s

        t = mod_divide(r, s * 2**l, q)
        u = mod_divide(hash_msg(msg), (-s * 2**l), q)
        attacker_pairs.append((u, t))

    ct = Fraction(1, 2**l)
    cu = Fraction(q, 2**l)
    dim = len(attacker_pairs) + 2
    basis = []
    for i in range(len(attacker_pairs)):
        vec = [0] * dim
        vec[i] = q
        basis.append(vec)

    basis.append(list(map(itemgetter(1), attacker_pairs)) + [ct, 0])
    basis.append(list(map(itemgetter(0), attacker_pairs)) + [0, cu])

    print(lll_reduction(basis))
    print('secret key', alice_keypair)
    print('looking for', cu)
