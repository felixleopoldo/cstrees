from copy import deepcopy

import numpy as np


def num_stagings(lvl: int):
    return lvl**3 + 1 if lvl != 2 else 8


def num_cstrees(num_lvls: int):
    return np.fromiter(map(num_stagings, range(num_lvls)), np.uint).prod()
    # replace with .cumprod() to get sequence from 1 to n


def enumerate_stagings(num_lvls: int):
    n_cube = [{0, 1} for _ in range(num_lvls)]
    cd0 = n_cube
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

    if num_lvls == 2:
        _ = cd2s.pop(0)

    return [cd0] + cd1s + cd12s + cd2s


def codim_1_subdivs(cube, fixed_dims=None):
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


def enumerate_cstrees(num_vars: int):
    if num_vars < 2:
        raise ValueError("CStrees with fewer than 2 variables aren't interesting.")
    elif num_vars == 2:
        return ({1: staging} for staging in enumerate_stagings(1))
    return (
        {**cstree, **{num_vars - 1: staging}}
        for staging in enumerate_stagings(num_vars - 1)
        for cstree in enumerate_cstrees(num_vars - 1)
    )


# test stage enumeration
# for n in range(2, 6):
#     cstrees = enumerate_cstrees(n)
#     assert sum(1 for cstree in cstrees) == num_cstrees(n)

# for staging in enumerate_cstrees(3):
#     print(staging)
