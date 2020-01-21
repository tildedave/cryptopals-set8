from typing import Tuple, Union
from numtheory import mod_divide


# "Point at infinity".
O = (0, 1)
EllipticCurvePoint = Tuple[int, int]


class EllipticCurve:
    """
    Curve defined by y^2 = x^3 + a*x + b
    """

    def __init__(self, p, a, b):
        self.prime = p
        self.a = a % p
        self.b = b % p

    def __contains__(self, p1: EllipticCurvePoint):
        if p1 == O:
            return True

        x, y = p1
        lhs = (y * y) % self.prime
        rhs = (x * x * x + self.a * x + self.b) % self.prime
        return lhs == rhs

    @property
    def identity(self):
        return O


def invert_point(p1: EllipticCurvePoint, curve: EllipticCurve) -> EllipticCurvePoint:
    """
    Return the inverse of the given elliptic curve point
    """
    return (p1[0], curve.prime - p1[1])


def add_point(p1: EllipticCurvePoint, p2: EllipticCurvePoint, curve: EllipticCurve) -> EllipticCurvePoint:
    if p1 == curve.identity:
        return p2

    if p2 == curve.identity:
        return p1

    if p1 == invert_point(p2, curve):
        return curve.identity

    p = curve.prime
    x1, y1 = p1
    x2, y2 = p2
    if p1 == p2:
        m = mod_divide(3 * x1 * x1  + curve.a, 2 * y1, p)
    else:
        m = mod_divide(y2 - y1, x2 - x1, p)

    x3 = (m * m - x1 - x2) % p
    y3 = (m * (x1 - x3) - y1) % p

    return (x3, y3)


def scalar_mult_point(p1: EllipticCurvePoint, n: int, curve: EllipticCurve):
    """
    Return a point added to itself n times
    """
    y = curve.identity
    if n == 0:
        return y

    if n < 0:
        raise ValueError('Not implemented yet')

    z = p1
    while n > 0:
        if n % 2 == 1:
            y = add_point(z, y, curve)
        n = n // 2
        z = add_point(z, z, curve)

    return y
