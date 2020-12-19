from typing import Tuple, Union
from random import randint

from numtheory import mod_divide, mod_sqrt


EllipticCurvePoint = Tuple[int, int]

"""
This is a point at infinity.

(It's assumed that (0, 1) isn't on the given curve.)
"""
EllipticCurveIdentity: EllipticCurvePoint = (0, 1)


class EllipticCurve:
    """
    Abstract class for typing
    """
    def __contains__(self, p1: EllipticCurvePoint):
        raise NotImplementedError

    def scalar_mult(self, p1: EllipticCurvePoint, n: int):
        raise NotImplementedError


class WeierstrassCurve(EllipticCurve):
    """
    Curve defined by y^2 = x^3 + a*x + b
    """

    def __init__(self, p, a, b):
        self.prime = p
        self.a = a % p
        self.b = b % p
        assert b % p != 1, '(0, 1) should not have been on curve'

    def __contains__(self, p1: EllipticCurvePoint):
        if p1 == EllipticCurveIdentity:
            return True

        x, y = p1
        lhs = (y * y) % self.prime
        rhs = (x * x * x + self.a * x + self.b) % self.prime
        return lhs == rhs

    def __str__(self):
        return (
            f'EllipticCurve y^2 = x^3 + {self.a}x + {self.b} '
            f'mod {self.prime}'
        )

    def scalar_mult(self, p1: EllipticCurvePoint, n: int):
        """
        Return a point added to itself n times
        """
        y = EllipticCurveIdentity
        if n == 0:
            return y

        if n < 0:
            raise ValueError('Not implemented yet')

        z = p1
        while n > 0:
            if n % 2 == 1:
                y = add_point(z, y, self)
            n = n // 2
            z = add_point(z, z, self)

        return y


def invert_point(p1: EllipticCurvePoint,
                 curve: WeierstrassCurve,
                 ) -> EllipticCurvePoint:
    """
    Return the inverse of the given elliptic curve point
    """
    return (p1[0], curve.prime - p1[1])


def add_point(p1: EllipticCurvePoint,
              p2: EllipticCurvePoint,
              curve: WeierstrassCurve,
              ) -> EllipticCurvePoint:
    if p1 == EllipticCurveIdentity:
        return p2

    if p2 == EllipticCurveIdentity:
        return p1

    if p1 == invert_point(p2, curve):
        return EllipticCurveIdentity

    p = curve.prime
    x1, y1 = p1
    x2, y2 = p2
    if p1 == p2:
        m = mod_divide(3 * x1 * x1 + curve.a, 2 * y1, p)
    else:
        m = mod_divide(y2 - y1, x2 - x1, p)

    x3 = (m * m - x1 - x2) % p
    y3 = (m * (x1 - x3) - y1) % p

    return (x3, y3)


def find_point_on_curve(curve: WeierstrassCurve):
    iterations = 0
    MAX_ITERATIONS = 5000
    while iterations < MAX_ITERATIONS:
        iterations += 1
        # x could be a quadratic residue so we need to assume
        x = randint(2, curve.prime)
        try:
            y_sq = (x * x * x + curve.a * x + curve.b) % curve.prime
            y = mod_sqrt(y_sq, curve.prime)
            assert (x, y) in curve, 'Chosen point should be on curve'

            return (x, y)
        except ValueError:
            # this is acceptable
            continue

    raise ValueError('Unable to find point on curve')


def find_point_order(pt: EllipticCurvePoint, curve: WeierstrassCurve):
    """
    Debugging method to find the order of a given point.  Uses repeated adding
    """
    order = 1
    while pt != EllipticCurveIdentity:
        pt = add_point(pt, pt, curve)
        order += 1

    return order


class MontgomeryCurve(EllipticCurve):
    """
    Curve defined by B*v^2 = u^3 + A*u^2 + u
    """

    def __init__(self, p, a, b):
        self.prime = p
        self.a = a % p
        self.b = b % p
        assert (0, 1) not in self, '(0, 1) should not have been on curve'

    def __contains__(self, p1: EllipticCurvePoint):
        # p1 is (u, v)
        u, v = p1[0], p1[1]
        p = self.prime

        return (self.b * v * v) % p == (u * u * u + self.a * u * u + u) % p

    def scalar_mult(self, p1: EllipticCurvePoint, n: int):
        # Implement ladder
        u, v = p1[0], p1[1]
        p = self.prime

        u2, w2 = (1, 0)
        u3, w3 = (u, 1)



def test_montgomery_contains():
    p = 233970423115425145524320034830162017933
    curve = MontgomeryCurve(p, 534, 1)
    assert (4, 85518893674295321206118380980485522083) in curve
