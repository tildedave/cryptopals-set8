from binascii import unhexlify
from numtheory import crt_inductive, mod_inverse
from rsa import RSAKeypair, rsa_encrypt

from gmpy2 import iroot, mpz

def test_crack_shared_cipher():
    ciphertext = 'Call me Ishmael.'

    keypair0 = RSAKeypair.create(e=3)
    keypair1 = RSAKeypair.create(e=3)
    keypair2 = RSAKeypair.create(e=3)

    c0 = rsa_encrypt(ciphertext, keypair0)
    c1 = rsa_encrypt(ciphertext, keypair1)
    c2 = rsa_encrypt(ciphertext, keypair2)

    n0 = keypair0.exponent
    n1 = keypair1.exponent
    n2 = keypair2.exponent
    m_s_0 = n1 * n2
    m_s_1 = n0 * n2
    m_s_2 = n0 * n1

    result, _ = crt_inductive([
        (c0 * m_s_0 * mod_inverse(m_s_0, n0), keypair0.exponent),
        (c1 * m_s_1 * mod_inverse(m_s_1, n1), n1),
        (c2 * m_s_2 * mod_inverse(m_s_2, n2), n2),
    ])
    cube_root, is_exact = iroot(mpz(result), 3)
    assert is_exact, 'Cube root should have been exact'
    assert unhexlify('{:02x}'.format(cube_root)).decode() == ciphertext
