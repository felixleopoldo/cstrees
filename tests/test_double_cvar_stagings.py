from itertools import zip_longest
from copy import deepcopy

import src.cstrees.double_cvar_stagings as dcs


def test_enumerate_binary_stagings():
    for n in range(10):
        stagings = dcs.enumerate_binary_stagings(n)
        assert sum(1 for staging in stagings) == dcs.num_binary_stagings(n)


def test_enumerate_binary_cstrees():
    for n in range(2, 6):
        cstrees = dcs.enumerate_binary_cstrees(n)
        assert sum(1 for cstree in cstrees) == dcs.num_binary_cstrees(n)


def test_max2_cvars_stagings():
    for n in range(4, 10):
        new_stagings = list(dcs.max2_cvars_stagings(n))
        old_stagings = list(dcs.max2_cvars_stagings_old(n))
        assert len(new_stagings) == len(old_stagings)
        non_unique = deepcopy(old_stagings)
        for idx, staging in enumerate(old_stagings):
            non_unique.remove(staging)
            try:
                non_unique.remove(staging)
                print(n, idx, staging)
                assert False
            except ValueError:
                continue
