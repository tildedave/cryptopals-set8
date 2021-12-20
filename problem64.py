from typing import Callable, List

# Length = 128
FieldElement = List[int]  # Element of GF(2^128) - each element is 0 or 1
FieldElementFunc = Callable[[int, int], int]

# Length of both of these = 128
Matrix = List[List[int]]
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


class RingContext:
    def __init__(self,
                 element_add: FieldElementFunc,
                 element_mult: FieldElementFunc) -> None:
        self.element_add = element_add
        self.element_mult = element_mult


def matrix_multiply_naive(ctx: RingContext, a: Matrix, b: Matrix) -> Matrix:
    # Naive matrix multiplication.  Assumes a and b are square matrices of
    # the same size.
    # https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Iterative_algorithm
    c = make_matrix(size=len(a))

    for i in range(0, len(a)):
        for j in range(0, len(b)):
            sum = 0
            for n in range(0, len(a)):
                m = ctx.element_mult(a[i][n], b[n][j])
                sum = ctx.element_add(sum, m)
            c[i][j] = sum

    return c


def matrix_add(ctx: RingContext, a: Matrix, b: Matrix) -> Matrix:
    assert len(a) == len(b)
    c = make_matrix(size=len(a))
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            c[i][j] = ctx.element_add(a[i][j], b[i][j])

    return c


def matrix_multiply_divide_and_conquer(ctx: RingContext,
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
        return [[a[0][0] * b[0][0]]]

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


IntegerContext = RingContext(lambda x, y: x + y, lambda x, y: x * y)
GF2Context = RingContext(lambda x, y: x ^ y, lambda x, y: x & y)


def test_matrix_multiply():
    a = [[0, 1], [1, 0]]
    b = [[2, 3], [3, 4]]

    assert matrix_multiply_naive(IntegerContext, b, a) == [[3, 2], [4, 3]]
    assert matrix_multiply_divide_and_conquer(IntegerContext, b, a) == \
        [[3, 2], [4, 3]]

    c = [[2, 7, 3], [1, 5, 8], [0, 4, 1]]
    d = [[3, 0, 1], [2, 1, 0], [1, 2, 4]]

    assert matrix_multiply_naive(IntegerContext, c, d) == \
        [[23, 13, 14], [21, 21, 33], [9, 6, 4]]
