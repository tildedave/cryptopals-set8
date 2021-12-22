from math import log2
from copy import deepcopy
import os.path
import pickle
import random
import string
from typing import Callable, Dict, List, Tuple, TypeVar

from problem63 import (
    GCM_MODULUS,
    FieldElement,
    element_add,
    element_mult,
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
    a = [[0, 1], [1, 0]]
    b = [[2, 3], [3, 4]]

    ctx = PrimeContext(127)
    assert matrix_multiply(ctx, b, a) == [[3, 2], [4, 3]]
    assert matrix_multiply_divide_and_conquer(ctx, b, a) == \
        [[3, 2], [4, 3]]

    c = [[2, 7, 3], [1, 5, 8], [0, 4, 1]]
    d = [[3, 0, 1], [2, 1, 0], [1, 2, 4]]

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


def element_to_vec(c: FieldElement) -> Vec:
    v = [0] * 128
    i = 0
    while c > 0:
        if c % 2 == 1:
            v[i] = 1
        c = c // 2
        i += 1

    return v


def vec_trunc(v: Vec):
    last_non_zero = None
    for i in range(0, 128):
        if v[i] != 0:
            last_non_zero = i + 1

    return v[0:last_non_zero]


def test_element_to_vec():
    assert(vec_trunc(element_to_vec(9)) == [1, 0, 0, 1])


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


def gf2_scalar_matrix(c: FieldElement) -> Matrix:
    cols = []
    for i in range(0, 128):
        basis_elem = 1 << i
        cols.append(element_to_vec(element_mult(c, basis_elem, GCM_MODULUS)))

    return matrix_transpose(cols)


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
    e2 = element_to_vec(7)
    result = matrix_multiply(GF2Context, x_plus_one_matrix, vec_to_matrix(e2))
    assert vec_to_element(matrix_to_vec(result)) == 9

    e3 = element_to_vec(3)
    result = matrix_multiply(GF2Context, x_plus_one_matrix, vec_to_matrix(e3))
    assert vec_to_element(matrix_to_vec(result)) == 5


def gf2_square_matrix() -> Matrix:
    cols = []
    for i in range(0, 128):
        basis_elem = 1 << i
        cols.append(element_to_vec(element_mult(
            basis_elem, basis_elem, GCM_MODULUS)))

    return matrix_transpose(cols)


def test_squaring_as_matrix():
    sq_m = gf2_square_matrix()
    # (a + 1)^2 = a^2 + 1
    e1 = element_to_vec(3)
    result = matrix_multiply(GF2Context, sq_m, vec_to_matrix(e1))
    assert vec_to_element(matrix_to_vec(result)) == 5
    # (a^2 + a + 1)^2 = a^4 + a^2 + 1
    e2 = element_to_vec(7)
    result = matrix_multiply(GF2Context, sq_m, vec_to_matrix(e2))
    assert vec_to_element(matrix_to_vec(result)) == 21
    # (a^5 + a + 1)^2 = a^10 + a^2 + 1
    e3 = element_to_vec(35)
    result = matrix_multiply(GF2Context, sq_m, vec_to_matrix(e3))
    assert vec_to_element(matrix_to_vec(result)) == 1029


def generate_plaintext(num_blocks):
    block_size = 128 // 8
    ciphertext = ''
    for _ in range(num_blocks):
        block = ''.join(random.choice(string.ascii_letters)
                        for _ in range(block_size))
        assert len(block.encode()) == block_size
        ciphertext += block

    return ciphertext


def calculate_ad(matrix_pows: List[Matrix],
                 coeffs: List[FieldElement],
                 new_coeffs: List[FieldElement],
                 ) -> Matrix:
    ad_matrix = make_matrix(128)
    for i in range(1, len(coeffs)):
        if coeffs[i] == new_coeffs[i]:
            assert element_add(coeffs[i], new_coeffs[i]) == 0
            continue
        # adding is subtraction in GF2
        flipped_scalar = element_add(coeffs[i], new_coeffs[i])
        ad_factor = matrix_multiply(
            GF2Context,
            gf2_scalar_matrix(flipped_scalar),
            matrix_pows[i],
        )
        ad_matrix = matrix_add(GF2Context, ad_matrix, ad_factor)

    return ad_matrix


def get_precomputed_structures() -> Tuple[bytes, int, List[Matrix]]:
    aes_key = ''.join(random.choice(string.ascii_letters) for _ in range(32))
    nonce = ''.join(random.choice(string.ascii_letters) for _ in range(12))
    assert len(nonce.encode()) * 8 == 96

    num_blocks = 2 ** 17
    n = int(log2(num_blocks))
    matrix_pows: List[Matrix]

    filename = './problem64_structures.p'
    if os.path.exists(filename):
        data = pickle.load(open(filename, 'rb'))
        ciphertext, t = data['ciphertext'], data['t']
        matrix_pows = data['matrix_pows']
        plaintext1 = data['plaintext1']
    else:
        # This takes around 20 seconds.  Pickling it saves some time while
        # fiddling with the system of simultaneous equations.
        # Longer-term we will need to generate multiple ciphertexts, so we
        # may just write these into the file as multiple ciphertexts as well.
        plaintext1 = generate_plaintext(num_blocks).encode()
        ciphertext, t = gcm_encrypt(plaintext1, b'', aes_key, nonce.encode(),
                                    tag_bits=64)

        # This takes a bunch of time too
        matrix_pows = [[[0]]] * (n + 1)
        sq_matrix = m = gf2_square_matrix()  # Msq (y) = y * y
        for i in range(1, n + 1):
            matrix_pows[i] = m
            m = matrix_multiply(GF2Context, sq_matrix, m)

        pickle.dump({
            'ciphertext': ciphertext,
            't': t,
            'matrix_pows': matrix_pows,
            'plaintext1': plaintext1,
        }, open(filename, 'wb'))

    return ciphertext, t, matrix_pows


def test_squaring_as_matrix_with_precomputed_powers():
    _, _, matrix_pows = get_precomputed_structures()

    e1 = element_to_vec(3)
    expected = 3
    for i in range(1, 17):
        result = matrix_multiply(GF2Context, matrix_pows[i], vec_to_matrix(e1))
        expected = element_mult(expected, expected, GCM_MODULUS)
        assert vec_to_element(matrix_to_vec(result)) == expected


def test_gcm_encrypt_truncated_mac_attack():
    random.seed(0)

    num_blocks = 2 ** 17
    n = int(log2(num_blocks))

    ciphertext, t, matrix_pows = get_precomputed_structures()

    # The problem (and Ferguson's paper) assume the first block contains just
    # metadata.  It doesn't look like that's the case in the NIST document (I
    # probably just haven't found the appropriate section) or the Problem 63
    # specifications.  In any case we'll just ignore the first block so we can
    # be consistent with the instructions.
    num_rows = (n - 1) * 128
    num_columns = n * 128

    dependency_matrix = [[]] * num_rows
    for i in range(num_rows):
        dependency_matrix[i] = [0] * num_columns

    coeffs: List[FieldElement] = [0] * (n + 1)
    for i in range(1, n + 1):
        # Extract block 2**i from ciphertext
        coeffs[i] = int_from_bytes(get_nth_block(ciphertext, 2**i))

    for j in range(num_columns):
        # Create the matrix AD that results from flipping the jth bit of the
        # ciphertext.
        block_num = (j // 128) + 1
        new_coeffs = coeffs.copy()
        new_coeffs[block_num] = coeffs[block_num] ^ (1 << j)
        print('about to compute')
        ad_matrix = calculate_ad(matrix_pows, coeffs, new_coeffs)
        print('done computing ad_matrix')
