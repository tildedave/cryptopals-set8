from functools import lru_cache
from math import log2
from copy import deepcopy
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
    gcm_encrypt,
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

def make_matrix(size=MatrixSize) -> Matrix:
    m: Matrix = [[]] * size
    for i in range(0, size):
        m[i] = [0] * size

    return m


Zero = make_matrix()
Identity = make_matrix()
for i in range(0, MatrixSize):
    Identity[i][i] = 1


def split_square_matrix(a: Matrix):
    a_size = len(a)
    top = a[0:a_size // 2]
    bottom = a[a_size // 2:]

    a_11 = [r[0:a_size // 2] for r in top]
    a_12 = [r[a_size // 2:] for r in top]
    a_21 = [r[0:a_size // 2] for r in bottom]
    a_22 = [r[a_size // 2:] for r in bottom]

    return a_11, a_12, a_21, a_22


def combine_matrices(a_11: Matrix,
                     a_12: Matrix,
                     a_21: Matrix,
                     a_22: Matrix,
                     ) -> Matrix:
    c = make_matrix(size=len(a_11) * 2)
    idx = len(a_11)
    for i in range(0, len(a_11)):
        for j in range(0, len(a_11)):
            c[i][j] = a_11[i][j]
            c[i][j + idx] = a_12[i][j]
            c[i + idx][j] = a_21[i][j]
            c[i + idx][j + idx] = a_22[i][j]

    return c


class FieldContext:
    def __init__(self,
                 element_add: Callable[[Scalar, Scalar], Scalar],
                 element_mult: Callable[[Scalar, Scalar], Scalar],
                 element_inverse: Callable[[Scalar], Scalar],
                 minus_one: Scalar) -> None:
        self.element_add = element_add
        self.element_mult = element_mult
        self.element_inverse = element_inverse
        self.minus_one = minus_one


def matrix_multiply(ctx: FieldContext, a: Matrix, b: Matrix) -> Matrix:
    # Naive matrix multiplication.
    # https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Iterative_algorithm
    c: Matrix = [[]] * len(a)
    for i in range(0, len(c)):
        c[i] = [0] * len(b[0])

    # a -> m * n
    # b -> n * p
    # result is size m * p

    for i in range(0, len(a)):
        for j in range(0, len(b[0])):
            # putting c[i][j] in place
            # c[i][j] = dot product of row i of a * column j of b
            sum = 0
            for n in range(0, len(a[i])):
                m = ctx.element_mult(a[i][n], b[n][j])
                sum = ctx.element_add(sum, m)
            c[i][j] = sum

    return c


def vec_scalar_mult(ctx, v: Vec, c: Scalar) -> Vec:
    return [ctx.element_mult(x, c) for x in v]


def matrix_add(ctx: FieldContext, a: Matrix, b: Matrix) -> Matrix:
    assert len(a) == len(b)
    c = deepcopy(a)
    for i, a_i in enumerate(a):
        for j, x in enumerate(a_i):
            c[i][j] = ctx.element_add(x, b[i][j])

    return c


def matrix_multiply_divide_and_conquer(ctx: FieldContext,
                                       a: Matrix,
                                       b: Matrix,
                                       ) -> Matrix:
    """
    Multiply matrices via divide and conquer technique

    Only works for matrices which are a power of 2 (can fix later if
    needed)
    """
    a_size = len(a)
    b_size = len(b)
    assert a_size == b_size

    if a_size == 1:
        return [[ctx.element_mult(a[0][0], b[0][0])]]

    a_11, a_12, a_21, a_22 = split_square_matrix(a)
    b_11, b_12, b_21, b_22 = split_square_matrix(b)

    c_11 = matrix_add(
        ctx,
        matrix_multiply_divide_and_conquer(ctx, a_11, b_11),
        matrix_multiply_divide_and_conquer(ctx, a_12, b_21))
    c_12 = matrix_add(
        ctx,
        matrix_multiply_divide_and_conquer(ctx, a_11, b_12),
        matrix_multiply_divide_and_conquer(ctx, a_12, b_22))
    c_21 = matrix_add(
        ctx,
        matrix_multiply_divide_and_conquer(ctx, a_21, b_11),
        matrix_multiply_divide_and_conquer(ctx, a_22, b_21))
    c_22 = matrix_add(
        ctx,
        matrix_multiply_divide_and_conquer(ctx, a_21, b_12),
        matrix_multiply_divide_and_conquer(ctx, a_22, b_22))

    return combine_matrices(c_11, c_12, c_21, c_22)


def assert_square_matrix(a: Matrix):
    for i, col in enumerate(a):
        assert len(col) == len(a), \
            f'Row {i} did not have the expected number of columns ({len(a)})'


def matrix_null_space(ctx: FieldContext, a: Matrix):
    """
    Adapted from Algorithm N of TAoCP section 4.6.2

    This finds the linearly independency vectors v_i, ..., v_i such that
    A v_i = 0
    """
    assert_square_matrix(a)
    n = len(a)
    cols = [-1] * n
    r = 0
    vecs = []

    for k in range(0, n):
        row = a[k]
        found = False
        for j in range(0, n):
            if row[j] != 0 and cols[j] < 0:
                found = True
                break

        if found:
            inv = ctx.element_mult(ctx.minus_one, ctx.element_inverse(row[j]))
            # multiply column j of A by -1 / a_kj
            for i in range(0, n):
                a[i][j] = ctx.element_mult(a[i][j], inv)

            # now add a_{ki} times column j to column i for all i != j
            for i in range(0, n):
                if i == j:
                    continue
                fact = a[k][i]
                for l in range(0, n):
                    a[l][i] = ctx.element_add(a[l][i],
                                              ctx.element_mult(a[l][j], fact))
            cols[j] = k
        else:
            # no j
            r += 1
            # output the vector
            null_vec: Vec = [0] * n
            for j in range(0, n):
                for s, c_s in enumerate(cols):
                    if c_s == j and c_s >= 0:
                        null_vec[j] = a[k][s]
                        break

                if j == k:
                    null_vec[j] = 1

            vecs.append(null_vec)

    return vecs


def PrimeContext(p):
    return FieldContext(lambda x, y: (x + y) % p,
                        lambda x, y: (x * y) % p,
                        lambda x: pow(x, p - 2, p),
                        p - 1)


def test_matrix_multiply():
    a = np.array([[0, 1], [1, 0]])
    b = np.array([[2, 3], [3, 4]])
    ctx = PrimeContext(127)
    assert matrix_multiply(ctx, b, a) == [[3, 2], [4, 3]]
    assert_array_equal(np.matmul(b, a), np.array([[3, 2], [4, 3]]))
    assert matrix_multiply_divide_and_conquer(ctx, b, a) == \
        [[3, 2], [4, 3]]

    c = np.array([[2, 7, 3], [1, 5, 8], [0, 4, 1]])
    d = np.array([[3, 0, 1], [2, 1, 0], [1, 2, 4]])

    assert matrix_multiply(ctx, c, d) == \
        [[23, 13, 14], [21, 21, 33], [9, 6, 4]]

    e = [[1, 2, 3], [4, 5, 6]]
    f = [[7, 8], [9, 10], [11, 12]]

    assert matrix_multiply(PrimeContext(241), e, f) == [[58, 64], [139, 154]]


def test_matrix_null_space():
    knuth_example = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 7, 11, 10, 12, 5, 11],
        [3, 6, 3, 3, 0, 4, 7, 2],
        [4, 3, 6, 4, 1, 6, 2, 3],
        [2, 11, 8, 8, 2, 1, 3, 11],
        [6, 11, 8, 6, 2, 6, 10, 9],
        [5, 11, 7, 10, 0, 11, 6, 12],
        [3, 3, 12, 5, 0, 11, 9, 11],
    ]
    results = matrix_null_space(PrimeContext(13), knuth_example)
    assert results == [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 5, 0, 9, 5, 1, 0],
        [0, 9, 11, 9, 10, 12, 0, 1],
    ]

    linearly_independent = [
        [1, 0, 5],
        [0, 1, 2],
        [0, 0, 1],
    ]
    results = matrix_null_space(PrimeContext(13), linearly_independent)
    assert len(results) == 0


GF2Context = FieldContext(lambda x, y: (x + y) % 2,
                          lambda x, y: 0 if x == 0 or y == 0 else 1,
                          lambda x: (x % 2),
                          1)


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
    for i in range(1, n + 1):
        matrix_pows[i] = m
        m = np.matmul(sq_matrix, m)

    return matrix_pows


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
        ad_matrix = ad_matrix + ad_factor

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


def test_gcm_encrypt_truncated_mac_attack():
    random.seed(0)

    num_blocks = 2 ** 2  # 17 one day after we have this actually working
    n = int(log2(num_blocks))
    aes_key = ''.join(random.choice(string.ascii_letters) for _ in range(32))
    nonce = ''.join(random.choice(string.ascii_letters) for _ in range(12))
    assert len(nonce.encode()) * 8 == 96

    plaintext1 = generate_plaintext(num_blocks).encode()
    ciphertext, t = gcm_encrypt(plaintext1, b'', aes_key, nonce.encode(),
                                tag_bits=64)

    # The problem (and Ferguson's paper) assume the first block contains just
    # metadata.  It doesn't look like that's the case in the NIST document (I
    # probably just haven't found the appropriate section) or the Problem 63
    # specifications.  In any case we'll just ignore the first block so we can
    # be consistent with the instructions.
    num_rows = (n - 1) * 128
    num_columns = n * 128

    dependency_matrix = GF2.Zeros((num_rows, num_columns))

    coeffs: List = [0] * (n + 1)
    for i in range(1, n + 1):
        # Extract block 2**i from ciphertext
        coeffs[i] = field(int_from_bytes(get_nth_block(ciphertext, 2**i)))

    for j in range(num_columns):
        # Create the matrix AD that results from flipping the jth bit of the
        # ciphertext.
        block_num = (j // 128) + 1
        ad_matrix = calculate_ad(n, coeffs, block_num, j % 128)
