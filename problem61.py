import random

from diffie_hellman import DiffieHellman, ECDHConfig, ECDHKeypair
from elliptic_curve import WeierstrassCurve
from numtheory import (
    crt_inductive,
    is_primitive_root,
    mod_divide,
    mod_inverse,
    pohlig_hellman,
    random_prime,
    random_smooth_prime,
    small_factors,
)
from rsa import RSAKeypair, rsa_sign, hash_msg, rsa_verify
from ecdsa import ecdsa_sign, ecdsa_verify


SIGN_MSG = """Call me Ishmael. Some years ago - never mind how long precisely -
having little or no money in my purse, and nothing particular to interest me on
shore, I thought I would sail about a little and see the watery part of the
world."""


def create_dh() -> DiffieHellman:
    p = 233970423115425145524320034830162017933
    curve = WeierstrassCurve(p, -95051, 11279326)
    point = (182, 85518893674295321206118380980485522083)
    point_order = 29246302889428143187362802287225875743

    return DiffieHellman(curve, point, point_order)


def test_sign_and_verify():
    dh = create_dh()
    keypair = dh.generate_keypair()

    sig = ecdsa_sign(SIGN_MSG, keypair)
    assert ecdsa_verify(SIGN_MSG, sig, keypair), 'Should have verified'


def test_eve_attack():
    dh = create_dh()
    alice_keypair = dh.generate_keypair()
    assert alice_keypair.is_valid()

    config = alice_keypair.config
    sig = ecdsa_sign(SIGN_MSG, alice_keypair)
    assert ecdsa_verify(SIGN_MSG, sig, alice_keypair)

    curve = config.curve
    eve_secret = random.randint(1, config.n)
    u1 = mod_divide(hash_msg(SIGN_MSG), sig.hash, config.n)
    u2 = mod_divide(sig.point_x, sig.hash, config.n)
    R = curve.add_points(
        config.scalar_mult_point(u1),
        curve.scalar_mult(alice_keypair.public, u2))

    assert curve.is_identity(curve.scalar_mult(R, config.n))

    t = mod_inverse(u1 + u2 * eve_secret, config.n)
    G_ = curve.scalar_mult(R, t)
    Q_ = curve.scalar_mult(G_, eve_secret)

    # Why does this work?
    # G_ = (u1 + u2 * d')^{-1} * R
    # Q_ = d' * (u1 + u2 * d')^{-1} * R
    # u1 * G_ + u2 * Q_ = (u1 + d' * u2) * G_ = R

    attacker_config = ECDHConfig(curve, G_, config.n)
    assert attacker_config.is_valid()  # Validates point order of G'
    eve_keypair = ECDHKeypair(attacker_config, eve_secret, public=Q_)
    assert eve_keypair.is_valid()  # Validates the public key matches secret

    assert ecdsa_verify(SIGN_MSG, sig, eve_keypair), 'Should have verified'


def test_eve_attack_rsa():
    kwargs = dict(range_start=2**120, range_end=2**121)
    e = random_prime(**kwargs)
    alice_keypair = RSAKeypair.create(e, **kwargs)
    sig = rsa_sign(SIGN_MSG, alice_keypair)

    assert rsa_verify(sig, alice_keypair)
    # We will now create a keypair that also verifies this signature

    # 1) find p
    #   a) p - 1 needs to be smooth (all small factors).
    #   b) s and pad(m) need to be in the same subgroups.  Text suggests
    #      ensuring both are primitive roots.

    MAX_TRIES = 1_000
    h = hash_msg(SIGN_MSG)
    p = None
    for _ in range(0, MAX_TRIES):
        p = random_smooth_prime(**kwargs)
        if is_primitive_root(sig.signature, p) and is_primitive_root(h, p):
            break
    else:
        assert False, f'Unable to find suitable prime {p}'

    p_group_factors = filter(lambda x: x != 2, list(small_factors(p - 1)))
    q = None

    # We now need a random prime so that p * q > exponent
    for _ in range(0, MAX_TRIES):
        q = random_smooth_prime(range_start=kwargs['range_start'] * 16,
                                range_end=kwargs['range_end'] * 32)

        if not is_primitive_root(sig.signature, q):
            continue

        if not is_primitive_root(h, q):
            continue

        # Then just ensure that q - 1 shares no factors with p
        for f in p_group_factors:
            if q - 1 % f  == 0:
                continue

        break

    assert p * q > alice_keypair.exponent

    # Now determine ep = e' mod p and eq = e' mod q using Pohlig-Hellman
    # s^e = pad(m) mod N

    ep = pohlig_hellman(sig.signature, h, p)
    eq = pohlig_hellman(sig.signature, h, q)
    assert pow(sig.signature, ep, p) == h % p, 'Did not solve correctly'
    assert pow(sig.signature, eq, q) == h % q, 'Did not solve correctly'

    e_, _ = crt_inductive([(ep, p - 1), (eq, q - 1)])
    assert pow(sig.signature, e_, (p * q)) == h % (p * q)

    d = mod_inverse(e_, p * q)
    eve_keypair = RSAKeypair(e_, d, p * q)
    assert rsa_verify(sig, eve_keypair)


if __name__ == '__main__':
    test_eve_attack()
