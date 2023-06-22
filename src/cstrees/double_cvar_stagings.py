"""Enumerate CStrees and stagings with up to 2 context variables."""
from copy import deepcopy
from itertools import chain

import numpy as np


def num_stagings(lvl: int):
    """Use formula to compute number of stagings for binary CStree at given level."""
    return lvl**3 + 1 if lvl != 2 else 8


def num_cstrees(num_lvls: int):
    """Use formula to compute number of binary CStrees for given number of levels."""
    return np.fromiter(map(num_stagings, range(num_lvls)), np.uint).prod()
    # replace with .cumprod() to get sequence from 1 to n


def enumerate_stagings(lvl: int):
    """Enumerate stagings for binary CStree at given level."""
    n_cube = [{0, 1} for _ in range(lvl - 1)]
    cd0 = [n_cube]
    cd1s = codim_1_subdivs(n_cube)
    cd12s = []
    cd2s = []
    for idx, subdiv in enumerate(cd1s):
        sub0_cd1s = codim_1_subdivs(subdiv[0], idx)
        sub1_cd1s = codim_1_subdivs(subdiv[1], idx)
        cd12s += [[subdiv[1]] + sub0_cd1 for sub0_cd1 in sub0_cd1s]
        cd12s += [[subdiv[0]] + sub1_cd1 for sub1_cd1 in sub1_cd1s]
        cd2s += [
            sub0_cd1 + sub1_cd1 for sub0_cd1 in sub0_cd1s for sub1_cd1 in sub1_cd1s
        ]

    if lvl - 1 == 2:
        _ = cd2s.pop(0)

    return chain([cd0], cd1s, cd12s, cd2s)


def max2_cvars_stagings(lvl):
    """Convert to iter."""
    tmp = enumerate_stagings(lvl)
    for staging in tmp:
        yield staging


def codim_1_subdivs(cube, fixed_dims=None):
    """Divide a cube into codimension-1 cubes, without dividing in fixed dims."""
    num_lvls = len(cube)
    subdivisions = []
    for idx in range(num_lvls):
        if idx == fixed_dims:
            continue
        subcube_0, subcube_1 = deepcopy(cube), deepcopy(cube)
        subcube_0[idx] = 0
        subcube_1[idx] = 1
        subdivision = [subcube_0, subcube_1]

        # add codim-1 subdivisions
        subdivisions.append(subdivision)
    return subdivisions


# test stage enumeration
# for n in range(10):
#     stagings = enumerate_stagings(n)
#     assert len(stagings) == num_stagings(n)

# for staging in enumerate_stagings(2):
#     print(staging)


def enumerate_cstrees(num_lvls: int):
    """Enumerate binary CStrees for given number of levels."""
    if num_lvls < 2:
        raise ValueError("CStrees with fewer than 2 variables aren't interesting.")
    if num_lvls == 2:
        return ({1: staging} for staging in enumerate_stagings(1))
    return (
        {**cstree, **{num_lvls - 1: staging}}
        for staging in enumerate_stagings(num_lvls - 1)
        for cstree in enumerate_cstrees(num_lvls - 1)
    )


# test stage enumeration
# for n in range(2, 6):
#     cstrees = enumerate_cstrees(n)
#     assert sum(1 for cstree in cstrees) == num_cstrees(n)
#
# for staging in enumerate_cstrees(3):
#     print(staging)
