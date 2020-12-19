from elliptic_curve import MontgomeryCurve, montgomery_ladder

p = 233970423115425145524320034830162017933
curve = MontgomeryCurve(p, 534, 1)

given_order = 29246302889428143187362802287225875743


if __name__ == "__main__":
    assert montgomery_ladder(curve, 4, given_order) == 0
    # Ladder attack
    bogus_u = 76600469441198017145391791613091732004
    assert montgomery_ladder(curve, bogus_u, 11) == 0

