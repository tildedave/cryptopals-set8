from functools import lru_cache
from math import log2
import galois
from galois import GF2
import numpy as np
from numpy.testing import assert_array_equal
import os
import pickle
import random
import string
from typing import Callable, List, TypeVar

from problem63 import (
    GCM_MODULUS,
    FieldElement,
    aes_encrypt,
    element_add,
    element_mult,
    gcm_encrypt,
    gcm_mac,
    gcm_mac_compute_g,
    get_nth_block,
    int_from_bytes,
)

# Length = 128
Scalar = int
SV = TypeVar('SV', bound=Scalar)
ScalarFunc = Callable[[SV, SV], SV]

# Length of both of these = 128
Vec = List[Scalar]
Matrix = List[Vec]
MatrixSize = 128

ir_poly = [0] * 129
ir_poly[0] = ir_poly[1] = ir_poly[2] = ir_poly[7] = ir_poly[128] = 1
field = galois.GF(2**128, galois.Poly(ir_poly, field=GF2))


def matrix_null_space(a: galois.FieldArray):
    # https://math.stackexchange.com/a/1612735/585559
    rows, cols = a.shape

    aug_array = a.Zeros((cols, rows + cols))
    aug_array[:, 0:rows] = a.transpose()
    aug_array[:, rows:] = a.Identity(cols)
    reduced = aug_array.row_reduce(ncols=rows)
    zero_rows, = np.where(~reduced[0:cols, 0:rows].any(axis=1))
    results = []
    for row in zero_rows:
        results.append(reduced[row, rows:])

    return results


def test_matrix_null_space():
    GF13 = galois.GF(13)
    knuth_example = GF13([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 7, 11, 10, 12, 5, 11],
        [3, 6, 3, 3, 0, 4, 7, 2],
        [4, 3, 6, 4, 1, 6, 2, 3],
        [2, 11, 8, 8, 2, 1, 3, 11],
        [6, 11, 8, 6, 2, 6, 10, 9],
        [5, 11, 7, 10, 0, 11, 6, 12],
        [3, 3, 12, 5, 0, 11, 9, 11],
    ])
    results = matrix_null_space(knuth_example.copy())
    assert len(results) == 3
    for result in results:
        assert np.matmul(knuth_example, result).all() == GF13.Zeros((8)).all()

    linearly_independent = GF13([
        [1, 0, 5],
        [0, 1, 2],
        [0, 0, 1],
    ])
    results = matrix_null_space(linearly_independent)
    assert len(results) == 0

    GF6199 = galois.GF(6199)
    so_example = GF6199([
        [1, 0, 2, 6199-3],
        [0, 1, 6199-1, 2],
        [0, 0, 0, 0],
    ])
    results = matrix_null_space(so_example)
    assert len(results) == 2
    for result in results:
        assert np.matmul(so_example, result).all() == GF6199.Zeros((4)).all()


def element_to_gf2_element(c: FieldElement) -> Vec:
    v = [0] * 128
    i = 0
    while c > 0:
        if c % 2 == 1:
            v[i] = 1
        c = c // 2
        i += 1

    return field.Vector(list(reversed(v)))


def vec_trunc(v: Vec):
    last_non_zero = None
    for i in range(0, 128):
        if v[i] != 0:
            last_non_zero = i + 1

    return v[0:last_non_zero]


def test_element_to_gf2_element():
    assert element_to_gf2_element(9) == field(9)


def matrix_transpose(a: Matrix) -> Matrix:
    rows = len(a)
    cols = len(a[0])
    b: Matrix = [[]] * cols
    for i in range(0, rows):
        b[i] = [0] * rows

    for i in range(0, len(a)):
        for j in range(0, len(a[i])):
            b[j][i] = a[i][j]

    return b


@lru_cache
def get_basis_elems() -> List[galois.FieldArray]:
    elems: List[galois.FieldArray] = []
    a = field.primitive_element
    x = 1
    for _ in range(0, 128):
        elems.append(x)
        x = x * a

    return elems


def gf2_scalar_matrix(c: FieldElement) -> Matrix:
    rows: List[np.ndarray] = []
    c_elem = field(c)
    basis_els = get_basis_elems()
    for i in range(0, 128):
        rows.insert(0, (c_elem * basis_els[i]).vector())

    return GF2(np.vstack(rows).transpose())


def vec_to_matrix(v: Vec) -> Matrix:
    return [[x] for x in v]


def matrix_to_vec(m: Matrix) -> Vec:
    v = [0] * len(m)
    for i in range(0, len(m)):
        assert len(m[i]) == 1, \
            'Assertion error: matrix_to_vec run on have multi-column matrix'
        v[i] = m[i][0]

    return v


def vec_to_element(v: Vec) -> FieldElement:
    e = 0
    for i in range(0, 128):
        if v[i] == 1:
            e += 2**i

    return e


def test_scalar_multiplication_vector():
    # Let's experiment with finding a matrix
    # Using test from problem63:
    # (x^2 + x + 1) * (x + 1) == x^3 + 1
    x_plus_one_matrix = gf2_scalar_matrix(3)
    e2 = field(7)
    result = np.matmul(x_plus_one_matrix, e2.vector())
    assert field.Vector(result) == field(9)

    e3 = field(3)
    result = np.matmul(x_plus_one_matrix, e3.vector())
    assert field.Vector(result) == field(5)


def gf2_square_matrix() -> GF2:
    rows: List[np.ndarray] = []
    a = field.primitive_element
    for i in range(0, 128):
        basis_elem = a ** i
        rows.insert(0, (basis_elem * basis_elem).vector())

    return GF2(np.vstack(rows).transpose())


def test_squaring_as_matrix():
    sq_m = gf2_square_matrix()
    # (a + 1)^2 = a^2 + 1
    result = np.matmul(sq_m, field(3).vector())
    assert field.Vector(result) == field(5)
    # (a^2 + a + 1)^2 = a^4 + a^2 + 1
    result = np.matmul(sq_m, field(7).vector())
    assert field.Vector(result) == field(21)
    # (a^5 + a + 1)^2 = a^10 + a^2 + 1
    result = np.matmul(sq_m, field(35).vector())
    assert field.Vector(result) == field(1029)


def generate_plaintext(num_blocks):
    block_size = 128 // 8
    ciphertext = ''
    for _ in range(num_blocks):
        block = ''.join(random.choice(string.ascii_letters)
                        for _ in range(block_size))
        assert len(block.encode()) == block_size
        ciphertext += block

    return ciphertext


@lru_cache
def get_matrix_pows(n):
    matrix_pows = [[[None]]] * (n + 1)
    sq_matrix = m = gf2_square_matrix()  # Msq (y) = y * y
    matrix_pows[0] = GF2.Identity(128)
    for i in range(1, n + 1):
        matrix_pows[i] = m
        m = np.matmul(sq_matrix, m)

    return matrix_pows


def test_matrix_pows():
    matrix_pows = get_matrix_pows(17)
    for _ in range(0, 10):
        x = field.Random()
        x_vec = x.vector()
        for i in range(0, len(matrix_pows)):
            matrix_result = field.Vector(np.matmul(matrix_pows[i], x_vec))
            assert matrix_result == (x ** (2 ** i))


def test_scalar_matrix():
    pass



@lru_cache
def get_gf2_scalar_matrices():
    filename = './problem64_gf2_scalar_matrices.p'
    if os.path.exists(filename):
        data = pickle.load(open(filename, 'rb'))
        return data['gf2_scalar_matrices']

    matrices = []
    basis_elems = get_basis_elems()
    for i in range(0, 128):
        matrices.append(gf2_scalar_matrix(basis_elems[i]))

    pickle.dump({'gf2_scalar_matrices': matrices}, open(filename, 'wb'))

    return matrices


def calculate_ad(n,
                 coeffs: List[FieldElement],
                 block_num: int,
                 bit_flip_position: int,
                 ) -> Matrix:
    matrix_pows = get_matrix_pows(n)
    scalar_matrices = get_gf2_scalar_matrices()
    ad_matrix = GF2.Zeros((128, 128))
    for i in range(1, len(coeffs)):
        if i != block_num:
            continue
        ad_factor = np.matmul(scalar_matrices[bit_flip_position],
                              matrix_pows[i])
        ad_matrix += ad_factor

    return ad_matrix


def test_squaring_as_matrix_with_precomputed_powers():
    n = 17
    matrix_pows = get_matrix_pows(n)
    e1 = field(3)
    expected = field(3)
    for i in range(1, 17):
        result = np.matmul(matrix_pows[i], e1.vector())
        expected = expected * expected
        assert field.Vector(result) == expected


def apply_bitflips(ciphertext, flip_vector):
    forged_text = bytearray(ciphertext)
    for i, should_flip in enumerate(flip_vector):
        if should_flip:
            block_num, bit_pos = divmod(i, 128)
            modify_block_num = 2 ** (block_num + 1)
            # Must figure out how to flip bit_pos within the block
            # Block starts at "start", ends at "start + bytes_per_block"
            byte_pos, bit_shift_num = divmod(bit_pos, 8)
            bytes_per_block = 128 // 8
            start = (modify_block_num - 1) * bytes_per_block
            forged_text[start + byte_pos] ^= (1 << (7 - bit_shift_num))

    return bytes(forged_text)


def test_gcm_encrypt_truncated_mac_attack():
    random.seed(0)

    tag_bits = 16
    num_blocks = 2 ** (tag_bits // 2 + 1)
    n = int(log2(num_blocks))
    aes_key = ''.join(random.choice(string.ascii_letters) for _ in range(32))
    nonce = ''.join(random.choice(string.ascii_letters) for _ in range(12))
    assert len(nonce.encode()) * 8 == 96

    plaintext1 = generate_plaintext(num_blocks).encode()
    ciphertext, t = gcm_encrypt(plaintext1, b'', aes_key, nonce.encode(),
                                tag_bits=tag_bits)

    coeffs: List = [0] * (n + 1)
    for i in range(1, n + 1):
        # Extract block 2**i from ciphertext
        coeffs[i] = field(int_from_bytes(get_nth_block(ciphertext, 2**i)))

    associated_bitlen = (0).to_bytes(8, byteorder='big')
    cipher_bitlen = (len(plaintext1) * 8).to_bytes(8, byteorder='big')
    length_block = associated_bitlen + cipher_bitlen

    # The problem (and Ferguson's paper) assume the first block contains just
    # metadata.  It doesn't look like that's the case in the NIST document (I
    # probably just haven't found the appropriate section) or the Problem 63
    # specifications.  In any case we'll just ignore the first block so we can
    # be consistent with the instructions.
    num_rows = (n - 1) * 128
    num_columns = n * 128

    # Let's validate that AD works the way we think it does
    print('validate ad time')
    bytes_per_block = 128 // 8
    h = int_from_bytes(aes_encrypt(bytes(bytes_per_block), aes_key))

    for i in range(0, 128):
        flip_vector = GF2.Zeros(num_columns)
        flip_vector[i] = 1
        flipped_ciphertext = apply_bitflips(ciphertext, flip_vector)

        ad = calculate_ad(n, coeffs, block_num=1, bit_flip_position=i)
        matrix_result = np.matmul(ad, field(h).vector())

        block_num = 1
        total_bytes = ciphertext + length_block
        total_flipped_ciphertext = flipped_ciphertext + length_block

        original_result = gcm_mac_compute_g(total_bytes, aes_key)
        flipped_result = gcm_mac_compute_g(total_flipped_ciphertext, aes_key)

        # Original value is 73870349645317180386927515316401291484
        assert field.Vector(matrix_result) == \
            field(flipped_result ^ original_result), \
            'AD calculation did not match as expected'

    # this is the dependency matrix in the problem description
    t_matrix = GF2.Zeros((num_rows, num_columns))

    print('t_matrix generation time')
    for j in range(num_columns):
        # Create the matrix AD that results from flipping the jth bit of the
        # ciphertext.
        block_num = (j // 128) + 1
        ad_matrix = calculate_ad(n, coeffs, block_num, j % 128)
        for i in range(num_rows // 128):
            # Copy row i of ad_matrix into column j of t_matrix
            t_matrix[(i * 128):(i + 1) * 128, j] = ad_matrix[i, :]
    results = matrix_null_space(t_matrix)
    assert len(results) == 128  # sure

    # all 2^i combinations
    done = False
    count = 0

    while not done:
        v = GF2.Zeros(num_columns)
        vector_list = random.sample(results, random.randint(1, len(results)))
        for vector in vector_list:
            v += vector

        forged_ciphertext = apply_bitflips(ciphertext, v.vector())
        forged_text_tag = gcm_mac(forged_ciphertext, b'', length_block,
                                  aes_key, nonce.encode(),
                                  tag_bits=tag_bits)

        if forged_text_tag == t:
            print('success!!!!!!!!!', v)
            print(forged_ciphertext, ciphertext)
            done = True

        count += 1
        if count > 2 ** (tag_bits // 2):
            assert False, 'did not find candidate in time'
