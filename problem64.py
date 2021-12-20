from typing import Callable, List, TypeVar

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


def vec_add(ctx, v1: Vec, v2: Vec) -> Vec:
    assert len(v1) == len(v2)
    v = [0] * len(v1)
    for i in range(0, len(v1)):
        v[i] = ctx.element_add(v1[i], v2[i])

    return v


def vec_scalar_mult(ctx, v: Vec, c: Scalar) -> Vec:
    return [ctx.element_mult(x, c) for x in v]


def matrix_add(ctx: FieldContext, a: Matrix, b: Matrix) -> Matrix:
    assert len(a) == len(b)
    c = make_matrix(size=len(a))
    for i in range(0, len(a)):
        c[i] = vec_add(ctx, a[i], b[i])

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
    print(n)
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
