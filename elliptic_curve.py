from typing import Tuple, Union
from random import randint

from numtheory import kronecker_symbol, mod_divide, mod_sqrt


EllipticCurvePoint = Tuple[int, int]

"""
This is a point at infinity.

(It's assumed that (0, 1) isn't on the given curve.)

TODO - this is Weierstrass only
"""
EllipticCurveIdentity: EllipticCurvePoint = (0, 1)


class EllipticCurve:
    """
    Abstract class for typing
    """
    def __contains__(self, p1: EllipticCurvePoint):
        raise NotImplementedError

    def scalar_mult(self,
                    p1: EllipticCurvePoint,
                    n: int) -> EllipticCurvePoint:
        raise NotImplementedError

    def add_points(self,
                   p1: EllipticCurvePoint,
                   p2: EllipticCurvePoint,
                   ) -> EllipticCurvePoint:
        raise NotImplementedError

    def double_point(self, p1: EllipticCurvePoint) -> EllipticCurvePoint:
        return self.add_points(p1, p1)

    def find_point_on_curve(self) -> EllipticCurvePoint:
        raise NotImplementedError

    def is_identity(self, point: EllipticCurvePoint) -> bool:
        raise NotImplementedError

    def find_point_of_order(self,
                            r: int,
                            curve_order: int,
                            ) -> EllipticCurvePoint:
        for _ in range(0, 10):
            pt = self.find_point_on_curve()
            assert pt in self, 'find_point_on_curve returned invalid value'

            candidate_point = self.scalar_mult(pt, curve_order // r)
            if self.is_identity(candidate_point):
                continue

            assert self.is_identity(self.scalar_mult(candidate_point, r))
            return candidate_point

        raise ValueError(f'Unable to find point of order {r} on curve {self}')


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

    def scalar_mult(self,
                    p1: EllipticCurvePoint,
                    n: int) -> EllipticCurvePoint:
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
                y = self.add_points(z, y)
            n = n // 2
            z = self.double_point(z)

        return y

    def invert_point(self,
                     p1: EllipticCurvePoint,
                     ) -> EllipticCurvePoint:
        """
        Return the inverse of the given elliptic curve point
        """
        return (p1[0], self.prime - p1[1])

    def add_points(self,
                   p1: EllipticCurvePoint,
                   p2: EllipticCurvePoint,
                   ) -> EllipticCurvePoint:
        if p1 == EllipticCurveIdentity:
            return p2

        if p2 == EllipticCurveIdentity:
            return p1

        if p1 == self.invert_point(p2):
            return EllipticCurveIdentity

        p = self.prime
        x1, y1 = p1
        x2, y2 = p2
        if p1 == p2:
            m = mod_divide(3 * x1 * x1 + self.a, 2 * y1, p)
        else:
            m = mod_divide(y2 - y1, x2 - x1, p)

        x3 = (m * m - x1 - x2) % p
        y3 = (m * (x1 - x3) - y1) % p

        return (x3, y3)

    def find_point_on_curve(self) -> EllipticCurvePoint:
        MAX_ITERATIONS = 5000
        for _ in range(0, MAX_ITERATIONS):
            # x could be a quadratic residue so we need to assume
            x = randint(2, self.prime)
            try:
                y_sq = (x * x * x + self.a * x + self.b) % self.prime
                y = mod_sqrt(y_sq, self.prime)
                assert (x, y) in self, 'Chosen point should be on curve'

                return (x, y)
            except ValueError:
                # this is acceptable
                continue

        raise ValueError('Unable to find point on curve')

    def is_identity(self, point: EllipticCurvePoint) -> bool:
        return point is EllipticCurveIdentity


def find_point_order(pt: EllipticCurvePoint, curve: WeierstrassCurve):
    """
    Debugging method to find the order of a given point.  Uses repeated adding
    """
    order = 1
    original_point = pt
    while pt != EllipticCurveIdentity:
        pt = curve.add_points(pt, original_point)
        order += 1

    return order


def cswap(n1, n2, z):
    d = int(z != 0)
    return (n1 * (1 - d) + n2 * d, n2 * (1 - d) + n1 * d)


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

    def scalar_mult(self,
                    p1: EllipticCurvePoint,
                    k: int) -> EllipticCurvePoint:
        u_ = montgomery_ladder(self, p1[0], k)
        # (u, v) and (u, -v) are both valid
        return (u_, montgomery_find_point(self, u_)[0])

    def find_point_on_curve(self) -> EllipticCurvePoint:
        MAX_ITERATIONS = 5000
        for _ in range(0, MAX_ITERATIONS):
            u = randint(2, self.prime)
            if not montgomery_point_test(self, u):
                continue

            return (u, montgomery_find_point(self, u)[0])

        raise ValueError('Unable to find point on curve')

    def is_identity(self, point: EllipticCurvePoint) -> bool:
        # Not exactly correct
        return point[0] == 0 and point[1] == 0


def mod_point(pt: EllipticCurvePoint, p: int) -> EllipticCurvePoint:
    return (pt[0] % p, pt[1] % p)


def montgomery_ladder(curve: MontgomeryCurve, u: int, k: int) -> int:
    p = curve.prime
    a = curve.a

    u2, w2 = (1, 0)
    u3, w3 = (u, 1)

    for i in reversed(range(0, p.bit_length())):
        b = 1 & (k >> i)
        u2, u3 = cswap(u2, u3, b)
        w2, w3 = cswap(w2, w3, b)
        u3, w3 = mod_point(((u2*u3 - w2*w3) ** 2, u * (u2*w3 - w2*u3) ** 2), p)
        u2, w2 = mod_point(((u2 ** 2 - w2 ** 2) ** 2,
                            4*u2*w2 * (u2 ** 2 + a * u2 * w2 + w2 ** 2)), p)
        u2, u3 = cswap(u2, u3, b)
        w2, w3 = cswap(w2, w3, b)

    return (u2 * pow(w2, p-2, p)) % p


def montgomery_find_point(curve: MontgomeryCurve,
                          u: int,
                          ) -> Tuple[int, int]:
    p = curve.prime
    a = curve.a
    b = curve.b

    rhs = (u ** 3 + a * (u ** 2) + u) % p
    if rhs == 0:
        return (0, 0)

    v = mod_sqrt(mod_divide(rhs, b, p), p)

    if p - v < v:
        return (p - v, v)

    return (v, p - v)


def montgomery_point_test(curve: MontgomeryCurve, u: int) -> bool:
    p = curve.prime
    a = curve.a
    b = curve.b

    rhs = (u ** 3 + a * (u ** 2) + u) % p
    v = mod_divide(rhs, b, p)

    return kronecker_symbol(v, p) == 1


def test_cswap():
    assert (4, 5) == cswap(4, 5, 0)
    assert (5, 4) == cswap(4, 5, 3)


def test_weierstrass_contains():
    p = 233970423115425145524320034830162017933
    curve = WeierstrassCurve(p, -95051, 11279326)
    point = (182, 85518893674295321206118380980485522083)
    assert point in curve, 'Point was not on curve (somehow)'

    given_order = 29246302889428143187362802287225875743
    pt = curve.scalar_mult(point, given_order)
    assert pt == EllipticCurveIdentity, 'Point did not have expected order'


def test_montgomery_contains():
    p = 233970423115425145524320034830162017933
    curve = MontgomeryCurve(p, 534, 1)
    assert (4, 85518893674295321206118380980485522083) in curve


def test_montgomery_ladder():
    p = 233970423115425145524320034830162017933
    curve = MontgomeryCurve(p, 534, 1)
    given_order = 29246302889428143187362802287225875743

    assert montgomery_ladder(curve, 4, given_order) == 0
    assert montgomery_ladder(curve, 76600469441198017145391791613091732004, 11) == 0


def test_montgomery_find_point_on_curve():
    p = 233970423115425145524320034830162017933
    given_order = 233970423115425145498902418297807005944
    curve = MontgomeryCurve(p, 534, 1)
    point = curve.find_point_of_order(4, given_order)

    assert curve.scalar_mult(point, 4) == (0, 0)

