from copy import copy
from fractions import Fraction
from typing import List, Union, Iterable

from ecdsa import ECDHKeypair, ECDSASignature


def ecdsa_sign_biased(msg: str, keypair: ECDHKeypair):
    n = keypair.config.n
    k = keypair.config.rand_scalar()
    k = (k >> 8) << 8  # Mask off the last 8 bits as suggested
    r = keypair.config.scalar_mult_point(k)[0]
    s = mod_divide(hash_msg(msg) + keypair.secret * r, k, n)

    return ECDSASignature(r, hash=s)


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
        sum_list = [[Fraction(0)] * len(b)] + [proj(u, v) for u in q[:i]]
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
