from copy import copy
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


GF2Polynomial = int


def polynomial_add(p1: GF2Polynomial, p2: GF2Polynomial):
    return p1 ^ p2


# GF2 is weird
polynomial_subtract = polynomial_add


def polynomial_degree(p1: GF2Polynomial):
    return p1.bit_length() - 1


def polynomial_string(p: GF2Polynomial):
    if p == 0:
        return '0'

    s = []
    for i in range(0, polynomial_degree(p) + 1):
        s.append(p & 1)
        p >>= 1

    parts: List[str] = []
    for i, b in enumerate(s):
        if b:
            if i == 0:
                parts.insert(0, '1')
            elif i == 1:
                parts.insert(0, 'x')
            else:
                parts.insert(0, f'x^{i}')

    return ' + '.join(parts)


def polynomial_mult(a: GF2Polynomial,
                    b: GF2Polynomial,
                    mod: Optional[GF2Polynomial] = None,
                    ) -> GF2Polynomial:
    p = 0
    while a > 0:
        if a & 1:
            p ^= b
        a >>= 1
        b <<= 1

        if mod is not None and polynomial_degree(b) == polynomial_degree(mod):
            b ^= mod

    return p


def polynomial_divmod(a: GF2Polynomial,
                      b: GF2Polynomial,
                      ) -> Tuple[GF2Polynomial, GF2Polynomial]:
    """
    Returns (a // b, a % b)
    """
    q, r = 0, a

    while polynomial_degree(r) >= polynomial_degree(b):
        d = polynomial_degree(r) - polynomial_degree(b)
        q = q ^ (1 << d)
        r = r ^ (b << d)

    return q, r


def polynomial_egcd(a: GF2Polynomial,
                    b: GF2Polynomial,
                    mod: Optional[GF2Polynomial] = None):
    """
    Return (d, u, v) so that a * u + b * v = d
    """
    if a == 0:
        return (b, 0, 1)
    else:
        q, r = polynomial_divmod(b, a)
        # so now q * b + r == a
        assert polynomial_add(polynomial_mult(q, a), r) == b

        g, x, y = polynomial_egcd(r, a, mod)
        return (g, polynomial_subtract(y, polynomial_mult(q, x, mod)), x)


def polynomial_inverse(a: GF2Polynomial,
                       m: GF2Polynomial,
                       ) -> GF2Polynomial:
    g, x, _ = polynomial_egcd(a, m)
    if g != 1:
        raise ValueError(f'{polynomial_string(a)} was not invertible')

    return divmod(x, m)[1]


def polynomial_mod_exp(a: GF2Polynomial,
                       n: int,
                       m: GF2Polynomial):
    p = 1
    while n > 0:
        if n % 2 == 1:
            p = polynomial_mult(p, a, m)
        a = polynomial_mult(a, a, m)
        n = n // 2

    return p


def test_polynomial_mult():
    # (x^2 + x + 1) * (x + 1) == x^3 + 1
    assert polynomial_mult(7, 3, 2**16) == 9
    assert polynomial_mult(7, 3, 8) == 1


def test_polynomial_degree():
    # 1 = 1
    assert polynomial_degree(1) == 0
    # 2 = x
    assert polynomial_degree(2) == 1
    # 3 = x + 1
    assert polynomial_degree(3) == 1


def test_polynomial_divmod():
    assert polynomial_divmod(9, 7) == (3, 0)
    assert polynomial_divmod(7, 15) == (0, 7)


def test_polynomial_gcd():
    # xgcd(x^3 + x + 1, x) = (1, 1, x^2 + 1)
    assert polynomial_egcd(11, 2) == (1, 1, 5)

    # xgcd(x, x^3 + x + 1) = (1, x^2 + 1, 1)
    assert polynomial_egcd(2, 11) == (1, 5, 1)

    # xgcd(x^3 + x + 1, x^2 + x + 1) == (1, x + 1, x^2)
    assert polynomial_egcd(11, 7) == (1, 3, 4)

    # gcd(x^3 + 1, x^2 + x + 1) = (x^2 + x + 1, 0, 1)
    assert polynomial_egcd(9, 7) == (7, 0, 1)


def test_polynomial_inverse():
    # Irreducible: x^4 + x + 1
    mod = 2**4 + 2**1 + 2**0

    p = 2**3 + 1
    inv = polynomial_inverse(p, mod)
    assert polynomial_mult(p, inv, mod) == 1


def test_polynomial_modexp():
    p = 2**3 + 1
    mod = 2**6 + 2**1 + 1
    # Sage - ((x^3 + 1)^3) % (x^6 + x + 1)
    assert polynomial_mod_exp(p, 3, mod) == 2**4 + 2**1


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
    print('length', length_block, plaintext_length)
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
        cipherblock = polynomial_add(b, cb_block).to_bytes(bytes_per_block,
                                                           byteorder='big')

        if block_num == num_blocks:
            # Last block.  Must zero out anything after the end.
            # This is sort of annoying but I had issues getting masking to work
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
    print(f'{ciphertext=} {associated_data=} {length_block=} {aes_key=} {nonce=} {len(ciphertext)}')
    bytes_per_block = 128 // 8
    block_num = 1
    total_bytes = associated_data + ciphertext + length_block

    h = int_from_bytes(aes_encrypt(bytes(bytes_per_block), aes_key))
    g = 0
    while block_num <= len(total_bytes) // bytes_per_block:
        # MAC: Convert block into GF2Polynomial
        block = get_nth_block(total_bytes, block_num)
        b: GF2Polynomial = int_from_bytes(block)
        g = polynomial_add(g, b)
        g = polynomial_mult(g, h, GCM_MODULUS)
        block_num += 1

    # '1' block is length (128 - 96) // 8 = 4
    j0 = bytes(nonce) + (1).to_bytes(4, byteorder='big')
    s = int.from_bytes(aes_encrypt(j0, aes_key), byteorder='big')
    t = polynomial_add(g, s)

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


def test_gcm_encryption_with_associated_data():
    aes_key = ''.join('s' for _ in range(32))
    nonce = b'\0' * 12
    plaintext = STREAM_MSG.encode()
    associated_data = ASSOCIATED_MSG.encode()
    ct, t = gcm_encrypt(plaintext, associated_data, aes_key, nonce)
    result, valid = gcm_decrypt(ct, associated_data, aes_key, nonce, t)
    print(result)
    assert valid


def test_gcm_encryption_single_block():
    aes_key = 's' * 32
    nonce = b'\0' * 12
    ct, t = gcm_encrypt(b'a' * (128 // 8), bytes(0), aes_key, nonce)
    result, valid = gcm_decrypt(ct, bytes(0), aes_key, nonce, t)
    assert valid
