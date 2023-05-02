import numpy as np


def num_stagings(l: int):
    return l**3 + 1 if l != 2 else 8


def num_cstrees(n: int):
    return np.fromiter(map(num_stagings, range(n)), int).prod()
    # return reduce(mul, map(num_stagings, range(n)))


def a_1_to_n(n: int):
    return np.fromiter(map(num_cstrees, range(1, n + 1)), int)


def enumerate_subdivisions(n, max_codim):
    pass
