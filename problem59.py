from elliptic_curve import EllipticCurve, add_point, invert_point, scalar_mult_point

p = 233970423115425145524320034830162017933
curve = EllipticCurve(p, -95051, 11279326)
point = (182, 85518893674295321206118380980485522083)
assert point in curve, 'Point was not on curve (somehow)'

given_order = 29246302889428143187362802287225875743
pt = scalar_mult_point(point, given_order, curve)
assert pt == curve.identity, 'Point did not have expected order'

