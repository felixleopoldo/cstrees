import numpy as np


def num_stagings(l: int):
    return l**3 + 1 if l != 2 else 8


def num_cstrees(n: int):
    return np.fromiter(map(num_stagings, range(n)), np.uint).prod()
    # replace with .cumprod() to get sequence from 1 to n


def enumerate_subdivisions(n, max_codim):
    pass
