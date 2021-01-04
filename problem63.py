from copy import copy
import random
from random import randint
from typing import List, Optional, Tuple

from cryptography.hazmat.primitives.ciphers import algorithms, modes, Cipher
import pytest


STREAM_MSG = """Call me Ishmael. Some years ago - never mind how long precisely
- having little or no money in my purse, and nothing particular to interest me
on shore, I thought I would sail about a little and see the watery part of the
world."""

ASSOCIATED_MSG = """Gabriel felt humiliated by the failure of his irony and by
the evocation of this figure from the dead, a boy in the gasworks. While he had
been full of memories of their secret life together, full of tenderness and joy
and desire, she had been comparing him in her mind with another."""


FieldElement = int  # Element of GF(2^128)
FieldPolynomial = List[FieldElement]  # Smallest degree up to largest degree
ZeroPolynomial: FieldPolynomial = []
OnePolynomial: FieldPolynomial = [1]


def element_add(p1: FieldElement, p2: FieldElement):
    return p1 ^ p2


# GF2 is weird
element_subtract = element_add


def element_degree(p1: FieldElement):
    return p1.bit_length() - 1


def polynomial_degree(p1: FieldPolynomial):
    return len(p1) - 1


def element_string(p: FieldElement):
    if p == 0:
        return '0'

    s = []
    for i in range(0, element_degree(p) + 1):
        s.append(p & 1)
        p >>= 1

    parts: List[str] = []
    for i, b in enumerate(s):
        if b:
            if i == 0:
                parts.insert(0, '1')
            elif i == 1:
                parts.insert(0, 'a')
            else:
                parts.insert(0, f'a^{i}')

    return ' + '.join(parts)


def polynomial_string(p: FieldPolynomial):
    if p == ZeroPolynomial:
        return '0'

    parts: List[str] = []
    for i, b in enumerate(p):
        if b != 0:
            b_str = element_string(b)
            if i == 0:
                parts.insert(0, b_str)
            elif i == 1:
                if b == 1:
                    parts.insert(0, 'x')
                else:
                    parts.insert(0, f'({b_str})*x')
            else:
                if b == 1:
                    parts.insert(0, f'x^{i}')
                else:
                    parts.insert(0, f'({b_str})*x^{i}')

    return ' + '.join(parts)


def element_mult(a: FieldElement,
                 b: FieldElement,
                 mod: Optional[FieldElement] = None,
                 ) -> FieldElement:
    p = 0
    while a > 0:
        if a & 1:
            p ^= b
        a >>= 1
        b <<= 1

        if mod is not None and element_degree(b) == element_degree(mod):
            b ^= mod

    if mod is not None and p > mod:
        return element_divmod(p, mod)[1]

    return p


def polynomial_trim(a: FieldPolynomial):
    # Find last non-zero index
    for i in range(len(a) - 1, -1, -1):
        if a[i] != 0:
            return a[0:i + 1]

    return ZeroPolynomial


def polynomial_mult(a: FieldPolynomial,
                    b: FieldPolynomial,
                    mod: Optional[FieldPolynomial] = None,
                    ) -> FieldPolynomial:
    d1 = polynomial_degree(a)
    d2 = polynomial_degree(b)
    dim = d1 + d2
    p = [0] * (dim + 1)

    for n in range(0, dim + 1):
        # Degree 0 factors are just the i, j 0 multipliers
        # Degree 1 factors are i = 1, j = 0, etc
        for i in range(0, n + 1):
            j = n - i
            if i <= d1 and j <= d2:
                p[n] = element_add(p[n], element_mult(a[i], b[j], GCM_MODULUS))

    if mod is None:
        return polynomial_trim(p)

    return polynomial_divmod(p, mod)[1]


def polynomial_add(a: FieldPolynomial, b: FieldPolynomial):
    d1 = polynomial_degree(a)
    d2 = polynomial_degree(b)
    dim = max(d1, d2)
    p = [0] * (dim + 1)

    for n in range(0, dim + 1):
        if n > d1:
            p[n] = b[n]
        elif n > d2:
            p[n] = a[n]
        else:
            p[n] = element_add(a[n], b[n])

    return polynomial_trim(p)


polynomial_subtract = polynomial_add


def polynomial_scalar_mult(a: FieldPolynomial,
                           x: FieldElement,
                           mod: Optional[FieldElement] = None):
    return polynomial_trim([element_mult(x, y, mod) for y in a])


def polynomial_divmod(a: FieldPolynomial,
                      b: FieldPolynomial,
                      ) -> Tuple[FieldPolynomial, FieldPolynomial]:
    """
    Return (q = a // b, r = a % b) through "synthetic division"

    At the end, q * b + r == a
    """
    b = polynomial_trim(b)  # Makes determining the leading coefficient easier

    if b == ZeroPolynomial:
        raise ValueError('Division by zero')

    d1 = polynomial_degree(a)
    d2 = polynomial_degree(b)

    if d1 < d2:
        r = copy(a)
        return ZeroPolynomial, r

    out = copy(a)
    inv = element_inverse(b[-1], GCM_MODULUS)

    for i in range(len(a) - 1, len(b) - 2, -1):
        out[i] = element_mult(out[i], inv, GCM_MODULUS)
        x = out[i]
        for j in range(len(b) - 2, -1, -1):
            term = i - (len(b) - 1 - j)
            y = element_mult(x, b[j], GCM_MODULUS)
            out[term] = element_subtract(out[term], y)

    return polynomial_trim(out[len(b) - 1:]), polynomial_trim(out[:len(b) - 1])


def polynomial_gcd(a: FieldPolynomial, b: FieldPolynomial):
    if len(a) > len(b):
        return polynomial_gcd(b, a)

    if a == ZeroPolynomial:
        return b

    _, r = polynomial_divmod(b, a)
    if r == ZeroPolynomial:
        return polynomial_make_monic(a)

    return polynomial_gcd(r, a)


def polynomial_exp(a: FieldPolynomial,
                   n: int,
                   m: Optional[FieldPolynomial] = None) -> FieldPolynomial:
    p = OnePolynomial
    while n > 0:
        if n % 2 == 1:
            p = polynomial_mult(p, a, m)
        a = polynomial_mult(a, a, m)
        n = n // 2

    return p


def polynomial_make_monic(a: FieldPolynomial):
    a = polynomial_trim(a)
    if a == ZeroPolynomial:
        return a

    inv = element_inverse(a[-1], GCM_MODULUS)
    return polynomial_trim(polynomial_scalar_mult(a, inv, GCM_MODULUS))


def polynomial_derivative(a: FieldPolynomial):
    # Derivative of a_0 + a_1 x + a_2 x^2 + ... = a_1 + a_3 x^2 + ...
    d = [0] * polynomial_degree(a)
    for i in range(0, len(d)):
        if i % 2 == 1:
            continue
        d[i] = a[i + 1]

    return polynomial_trim(d)


def polynomial_remove_square_factors(a: FieldPolynomial):
    while True:
        d = polynomial_derivative(a)
        g = polynomial_gcd(a, d)
        if g == OnePolynomial or d == ZeroPolynomial:
            return a

        q, r = polynomial_divmod(a, g)
        assert r == ZeroPolynomial
        a = q


def polynomial_ddf(a: FieldPolynomial, q=2**128):
    """
    Takes a monic, squarefree polynomial and computes its distinct degree
    factors
    """
    i = 1
    factors: List[Tuple[FieldPolynomial, int]] = []
    a_ = copy(a)
    while polynomial_degree(a_) >= 2 * i:
        # x^{(2^128)^i} - x --> in our field x^((2^128)^i) + x
        poly = polynomial_subtract(polynomial_exp([0, 1], q**i, a_), [0, 1])
        g = polynomial_gcd(a_, poly)
        if g != OnePolynomial:
            factors.append((g, i))
            q0, r = polynomial_divmod(a_, g)
            assert r == ZeroPolynomial
            a_ = q0
        i += 1

    if a_ != OnePolynomial:
        factors.append((a_, polynomial_degree(a_)))

    if len(factors) == 0:
        return [(a, 1)]

    return factors


def polynomial_edf(a: FieldPolynomial, d: int, q: int = 2**128):
    """
    Factor polynomial using Cantor-Zassenhaus method
    """
    n = polynomial_degree(a)
    assert n % d == 0, 'Should have been equally divisible'
    r = n // d
    factors = [a]

    while len(factors) < r:
        # Generate a random polynomial between 1 and a
        # Field elements are between 0 and 2^128 (so a random 128 bit integer)
        h = polynomial_trim([randint(0, q) for _ in range(len(a))])
        h = polynomial_make_monic(h)
        assert h != ZeroPolynomial  # Incredibly unlikely

        g = polynomial_gcd(h, a)
        if g == OnePolynomial:
            assert (q**d - 1) % 3 == 0
            g = polynomial_subtract(
                polynomial_exp(h, (q**d - 1) // 3, a),
                OnePolynomial)

        new_factors = []
        for u in factors:
            if polynomial_degree(u) == d:
                new_factors.append(u)
                continue

            # Relies on u being trimmed
            u_gcd = polynomial_gcd(g, u)
            if u_gcd not in [OnePolynomial, u]:
                # Non-trivial factor existed - remove u from factors and add
                # a new one
                new_factors.append(u_gcd)
                q0, r0 = polynomial_divmod(u, u_gcd)
                assert r0 == ZeroPolynomial
                new_factors.append(q0)
            else:
                new_factors.append(u)

        factors = new_factors

    return factors


def test_polynomial_add():
    p2 = [1, 0, 1]
    p1 = [0, 1, 0, 1]
    assert polynomial_add(p1, p2) == [1, 1, 1, 1]
    assert polynomial_add(p1, p1) == ZeroPolynomial


def test_polynomial_mult():
    p1 = [1, 0, 1]
    assert polynomial_mult(p1, p1) == [1, 0, 0, 0, 1]


def test_polynomial_divmod():
    assert polynomial_divmod([1, 0, 1], [1, 0, 0, 0, 1]) == ([], [1, 0, 1])

    p1 = [0, 1, 1, 1]
    p2 = [1, 1]
    q, r = polynomial_divmod(p1, p2)
    assert q, r == ([1, 0, 1], [1])
    # This is true modulo trimming
    assert polynomial_add(polynomial_mult(q, p2), r) == p1


def test_polynomial_divmod_arbitrary_field_elements():
    a = [56]
    b = [26]
    q, r = polynomial_divmod(a, b)

    assert len(polynomial_mult(q, b)) == 1
    # Here q should be 0 and r should be the inverse of b
    x = polynomial_mult(q, b)[0]
    assert x < GCM_MODULUS
    assert x == a[0]
    # q * b + r == a (supposedly)
    assert polynomial_add(polynomial_mult(q, b), r) == a


def test_polynomial_exp():
    a = [1, 0, 1]  # x^2 + 1
    assert polynomial_exp(a, 5) == [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]

    m = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # x^12 + x^9 + x^6 + x^3 + 1
    assert polynomial_exp([0, 1], 16, m) == [0, 1]


def test_polynomial_gcd():
    a = [1, 0, 0, 1]  # x^3 + 1
    b = [1, 0, 0, 0, 1]  # x^4 + 1
    assert polynomial_gcd(a, b) == [1, 1]  # x + 1
    assert polynomial_gcd(a, ZeroPolynomial) == a
    assert polynomial_gcd(ZeroPolynomial, a) == a

    a = [1, 1, 0, 0, 0, 0, 1]
    b = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert polynomial_gcd(a, b) == OnePolynomial


def test_polynomial_derivative():
    # Derivative of x^4 + x^3 + x^2 + x + 1 = 3x^2 + 1 = x^2 + 1
    assert polynomial_derivative([1, 1, 1, 1, 1]) == [1, 0, 1]


def test_polynomial_remove_square_factors():
    p1 = [1, 1, 0, 0, 1, 1]  # x^5+x^4+x+1 = (x+1)^5
    assert polynomial_remove_square_factors(p1) == [1, 1]
    p2 = [1, 0, 0, 1, 1, 0, 0, 1]  # x^7+x^4+x^3+1 = (x^2+1)^2 * (x^3+1)
    assert polynomial_remove_square_factors(p2) == [1, 0, 0, 1]


def test_polynomial_ddf():
    ddf = polynomial_ddf([1, 0, 0, 1], q=2)
    assert len(ddf) == 2

    assert ([1, 1], 1) in ddf
    assert ([1, 1, 1], 2) in ddf

    # ((x^4+x^3+1) * (x^4+x^3+x^2+x+1) * (x^4+x+1)) = x^12+x^9+x^6+x^3+1
    # These are all irreducible so DDF won't break these down further.
    p = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    result = polynomial_ddf(p, q=2)
    assert result == [(p, 4)]

    # This is q = 2^128 which is the normal one we need
    assert polynomial_ddf([1, 1, 1]) == [([1, 1, 1], 1)]


def test_polynomial_edf():
    result = polynomial_edf([1, 1, 1], 1)
    # Every polynomial of degree < 128 splits in GF(2**128)
    assert len(result) == 2
    assert polynomial_mult(result[0], result[1]) == [1, 1, 1]

    # ((x^4+x^3+1) * (x^4+x^3+x^2+x+1) * (x^4+x+1)) = x^12+x^9+x^6+x^3+1
    # These are all irreducible so DDF won't break these down further.
    p = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    result = polynomial_edf(p, 4, q=2)
    assert len(result) == 3
    assert [1, 0, 0, 1, 1] in result
    assert [1, 1, 1, 1, 1] in result
    assert [1, 1, 0, 0, 1] in result


@pytest.mark.skip("fairly slow (30 seconds)")
def test_factor_split_twelve_degree_poly():
    # Should split into linear factors in GF(2**128)
    p = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    result = polynomial_edf(p, 1)
    assert len(result) == 12


def element_divmod(a: FieldElement,
                   b: FieldElement,
                   ) -> Tuple[FieldElement, FieldElement]:
    """
    Returns (a // b, a % b)
    """
    q, r = 0, a

    while element_degree(r) >= element_degree(b):
        d = element_degree(r) - element_degree(b)
        q = q ^ (1 << d)
        r = r ^ (b << d)

    return q, r


def element_egcd(a: FieldElement,
                 b: FieldElement):
    """
    Return (d, u, v) so that a * u + b * v = d
    """
    if a == 0:
        return (b, 0, 1)
    else:
        q, r = element_divmod(b, a)
        # so now q * b + r == a
        assert element_add(element_mult(q, a), r) == b

        g, x, y = element_egcd(r, a)
        return (g, element_subtract(y, element_mult(q, x)), x)


def element_inverse(a: FieldElement,
                    m: FieldElement,
                    ) -> FieldElement:
    g, x, _ = element_egcd(a, m)
    if g != 1:
        raise ValueError(f'{element_string(a)} was not invertible')

    # x * a + y * m = 1, therefore x inverts a
    return x


def element_exp(a: FieldElement,
                n: int,
                m: Optional[FieldElement] = None):
    p = 1
    while n > 0:
        if n % 2 == 1:
            p = element_mult(p, a, m)
        a = element_mult(a, a, m)
        n = n // 2

    return p


def test_element_mult():
    # (x^2 + x + 1) * (x + 1) == x^3 + 1
    assert element_mult(7, 3, 2**16) == 9
    assert element_mult(7, 3, 8) == 1


def test_element_degree():
    # 1 = 1
    assert element_degree(1) == 0
    # 2 = x
    assert element_degree(2) == 1
    # 3 = x + 1
    assert element_degree(3) == 1


def test_element_divmod():
    assert element_divmod(9, 7) == (3, 0)
    assert element_divmod(7, 15) == (0, 7)


def test_element_gcd():
    # xgcd(x^3 + x + 1, x) = (1, 1, x^2 + 1)
    assert element_egcd(11, 2) == (1, 1, 5)

    # xgcd(x, x^3 + x + 1) = (1, x^2 + 1, 1)
    assert element_egcd(2, 11) == (1, 5, 1)

    # xgcd(x^3 + x + 1, x^2 + x + 1) == (1, x + 1, x^2)
    assert element_egcd(11, 7) == (1, 3, 4)

    # gcd(x^3 + 1, x^2 + x + 1) = (x^2 + x + 1, 0, 1)
    assert element_egcd(9, 7) == (7, 0, 1)


def test_element_inverse():
    # Irreducible: x^4 + x + 1
    mod = 2**4 + 2**1 + 2**0

    p = 2**3 + 1
    inv = element_inverse(p, mod)
    assert element_mult(p, inv, mod) == 1
    assert element_inverse(1, mod) == 1


def test_element_exp():
    p = 2**3 + 1
    mod = 2**6 + 2**1 + 1
    # Sage - ((x^3 + 1)^3) % (x^6 + x + 1)
    assert element_exp(p, 3, mod) == 2**4 + 2**1
    assert element_exp(p, 0) == 1


def aes_encrypt(block: bytes, aes_key: str):
    aes_cipher = Cipher(algorithms.AES(aes_key.encode()), modes.ECB())
    encryptor = aes_cipher.encryptor()
    return encryptor.update(block) + encryptor.finalize()


def pad_bytes(b: bytes, block_length_bytes: int):
    if len(b) == 0:
        return bytes(128 // 8), 0

    missing_bytes = len(b) % block_length_bytes
    if missing_bytes == 0:
        return b, len(b)

    pad_bytes = block_length_bytes - missing_bytes
    return b + bytes(pad_bytes), len(b)


def int_from_bytes(b: bytes) -> int:
    return int.from_bytes(b, byteorder='big')


def get_nth_block(b: bytes, block_num: int):
    bytes_per_block = 128 // 8
    start = (block_num - 1) * bytes_per_block
    block = b[start:start + bytes_per_block]
    assert len(block) == bytes_per_block

    return block


GCM_MODULUS = 2**128 + 2**7 + 2**2 + 2 + 1  # x^128 + x^127 + x^2 + x + 1


def gcm_encrypt(plaintext: bytes,
                associated_data: bytes,
                aes_key: str,
                nonce: bytes,
                ) -> Tuple[bytes, int]:
    assert len(nonce) == 96 // 8, f'Nonce must be {96 // 8} bytes'

    bytes_per_block = 128 // 8
    plaintext, plaintext_length = pad_bytes(plaintext, 128 // 8)
    associated_data, associated_length = pad_bytes(associated_data, 128 // 8)
    assert len(plaintext) % bytes_per_block == 0
    assert len(associated_data) % bytes_per_block == 0

    # CTR Encryption
    ciphertext = ctr_aes(plaintext, plaintext_length, aes_key, nonce)
    associated_bitlen = (associated_length * 8).to_bytes(8, byteorder='big')
    cipher_bitlen = (plaintext_length * 8).to_bytes(8, byteorder='big')
    length_block = associated_bitlen + cipher_bitlen

    assert len(length_block) == bytes_per_block
    # MAC Calculation
    t = gcm_mac(ciphertext, associated_data, length_block, aes_key, nonce)

    # Must trim ciphertext to original bitlength
    return ciphertext[0: plaintext_length], t


def ctr_aes(plaintext: bytes,
            plaintext_length: int,
            aes_key: str,
            nonce: bytes) -> bytes:
    ciphertext = b''
    bytes_per_block = 128 // 8
    block_num = 1
    num_blocks = len(plaintext) // bytes_per_block

    while block_num <= num_blocks:
        cb = bytes(nonce) + block_num.to_bytes(4, byteorder='big')
        cb_block = int_from_bytes(aes_encrypt(cb, aes_key))
        b = int_from_bytes(get_nth_block(plaintext, block_num))
        cipherblock = element_add(b, cb_block).to_bytes(bytes_per_block,
                                                        byteorder='big')

        if block_num == num_blocks:
            # Last block.  Must zero out anything after the end.
            # This is sort of annoying; I had issues getting masking to work
            nonzero_bytes = plaintext_length % bytes_per_block
            cipherblock = bytearray(cipherblock)
            if nonzero_bytes != 0:
                for i in range(nonzero_bytes, bytes_per_block):
                    cipherblock[i] = 0

        ciphertext += cipherblock
        block_num += 1

    return ciphertext


def gcm_mac(ciphertext: bytes,
            associated_data: bytes,
            length_block: bytes,
            aes_key: str,
            nonce: bytes) -> int:
    bytes_per_block = 128 // 8
    block_num = 1
    total_bytes = associated_data + ciphertext + length_block

    h = int_from_bytes(aes_encrypt(bytes(bytes_per_block), aes_key))
    g = 0
    while block_num <= len(total_bytes) // bytes_per_block:
        # MAC: Convert block into FieldElement
        block = get_nth_block(total_bytes, block_num)
        b: FieldElement = int_from_bytes(block)
        g = element_add(g, b)
        g = element_mult(g, h, GCM_MODULUS)
        block_num += 1

    # '1' block is length (128 - 96) // 8 = 4
    j0 = bytes(nonce) + (1).to_bytes(4, byteorder='big')
    s = int.from_bytes(aes_encrypt(j0, aes_key), byteorder='big')
    t = element_add(g, s)

    return t


def gcm_decrypt(ciphertext: bytes,
                associated_data: bytes,
                aes_key: str,
                nonce: bytes,
                t: int):
    assert len(nonce) == 96 // 8, f'Nonce must be {96 // 8} bytes'

    bytes_per_block = 128 // 8
    ciphertext, cipher_length = pad_bytes(ciphertext, 128 // 8)

    associated_data, associated_length = pad_bytes(associated_data, 128 // 8)
    assert len(ciphertext) % bytes_per_block == 0
    assert len(associated_data) % bytes_per_block == 0

    # CTR Encryption
    plaintext = ctr_aes(ciphertext, cipher_length, aes_key, nonce)

    # Length block
    associated_bitlen = (associated_length * 8).to_bytes(8, byteorder='big')
    cipher_bitlen = (cipher_length * 8).to_bytes(8, byteorder='big')
    length_block = associated_bitlen + cipher_bitlen
    assert len(length_block) == bytes_per_block

    t0 = gcm_mac(ciphertext, associated_data, length_block, aes_key, nonce)

    return plaintext[0: cipher_length], t0 == t


def ciphertext_to_field_polynomial(ciphertext: bytes,
                                   associated_data: bytes):
    ciphertext, cipher_length = pad_bytes(ciphertext, 128 // 8)
    associated_data, associated_length = pad_bytes(associated_data, 128 // 8)

    associated_bitlen = (associated_length * 8).to_bytes(8, byteorder='big')
    cipher_bitlen = (cipher_length * 8).to_bytes(8, byteorder='big')
    length_block = associated_bitlen + cipher_bitlen
    total_bytes = associated_data + ciphertext + length_block
    bytes_per_block = 128 // 8
    block_num = 1

    poly = []
    while block_num <= len(total_bytes) // bytes_per_block:
        # MAC: Convert block into FieldElement
        block = get_nth_block(total_bytes, block_num)
        poly.append(int_from_bytes(block))
        block_num += 1

    # s is the constant term but we don't know that
    poly.append(0)
    poly.reverse()

    return poly


def polynomial_evaluate(poly: FieldPolynomial,
                        x: FieldElement,
                        ) -> FieldElement:
    v = 0
    for a, i in enumerate(poly):
        i_coeff = element_mult(a, element_exp(x, i, GCM_MODULUS))
        v = element_add(v, i_coeff)

    return v


def test_gcm_encryption_with_associated_data():
    aes_key = ''.join('s' for _ in range(32))
    nonce = b'\0' * 12
    plaintext = STREAM_MSG.encode()
    associated_data = ASSOCIATED_MSG.encode()
    ct, t = gcm_encrypt(plaintext, associated_data, aes_key, nonce)
    result, valid = gcm_decrypt(ct, associated_data, aes_key, nonce, t)

    assert result == plaintext
    assert valid


def test_gcm_encryption_single_block():
    aes_key = 's' * 32
    nonce = b'\0' * 12
    plaintext = b'a' * (128 // 8)
    ct, t = gcm_encrypt(plaintext, bytes(0), aes_key, nonce)
    result, valid = gcm_decrypt(ct, bytes(0), aes_key, nonce, t)
    assert result == plaintext
    assert valid


def get_auth_key_candidates(l: List[Tuple[bytes, bytes, int]]):
    p = ZeroPolynomial
    for ct, ad, t in l:
        poly = ciphertext_to_field_polynomial(ct, ad)
        poly[0] = t
        p = polynomial_add(p, poly)

    p = polynomial_remove_square_factors(p)
    p = polynomial_make_monic(p)

    ddf_factors = polynomial_ddf(p)
    candidates = []
    for factor, d in ddf_factors:
        edf_factors = polynomial_edf(factor, d)
        for f in edf_factors:
            if polynomial_degree(f) == 1:
                # Polynomial has form x - c
                candidates.append(f[0])

    return candidates


def test_gcm_encryption_attack():
    random.seed(0)
    aes_key = 's' * 32
    nonce = b'\0' * 11 + b'\1'
    plaintext = b'a' * (128 // 8) * 4
    ct, t1 = gcm_encrypt(plaintext, bytes(0), aes_key, nonce)
    ct2, t2 = gcm_encrypt(b'b' * (128 // 8), bytes(0), aes_key, nonce)
    h = int_from_bytes(aes_encrypt(bytes(128 // 8), aes_key))

    candidates = get_auth_key_candidates([
        (ct, bytes(0), t1),
        (ct2, bytes(0), t2),
    ])
    assert h in candidates, 'Should have retrieved authentication key'
