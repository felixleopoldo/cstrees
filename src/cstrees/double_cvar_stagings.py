"""Enumerate CStrees and stagings with up to 2 context variables."""
from copy import deepcopy
from itertools import chain

import numpy as np


def num_binary_stagings(lvl: int):
    """Use formula to compute number of stagings at given level of binary CStree ."""
    return lvl**3 + 1 if lvl != 2 else 8


def num_binary_cstrees(num_lvls: int):
    """Use formula to compute number of binary CStrees for given number of levels."""
    return np.fromiter(map(num_binary_stagings, range(num_lvls)), np.uint).prod()
    # replace with .cumprod() to get sequence from 1 to n


def max2_cvars_stagings(lvl):
    """Convert to iter."""
    tmp = enumerate_binary_stagings(lvl - 1)
    for staging in tmp:
        yield staging


def max2_cvars_stagings_general(var_outcomes: list, restricted_to_cvars: tuple = None):
    """Enumerate stagings at given level of CStree."""
    stage_no_context = [var_outcomes]
    yield stage_no_context


def enumerate_binary_stagings(lvl: int):
    """Enumerate stagings at given level of binary CStree."""
    n_cube = [{0, 1} for _ in range(lvl)]
    cd0 = [n_cube]
    cd1s = codim_1_subdivs_binary(n_cube)
    cd12s = []
    cd2s = []
    for idx, subdiv in enumerate(cd1s):
        sub0_cd1s = codim_1_subdivs_binary(subdiv[0], idx)
        sub1_cd1s = codim_1_subdivs_binary(subdiv[1], idx)
        cd12s += [[subdiv[1]] + sub0_cd1 for sub0_cd1 in sub0_cd1s]
        cd12s += [[subdiv[0]] + sub1_cd1 for sub1_cd1 in sub1_cd1s]
        cd2s += [
            sub0_cd1 + sub1_cd1 for sub0_cd1 in sub0_cd1s for sub1_cd1 in sub1_cd1s
        ]

    if lvl == 2:
        _ = cd2s.pop(0)

    return chain([cd0], cd1s, cd12s, cd2s)


def codim_1_subdivs_binary(cube, fixed_dims=None):
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


def max1_cvar_stagings(staging: list, fixed_vars: tuple = None):
    """Enumerate new stagings resulting from one additional cvar to given staging."""
    pass


def enumerate_binary_cstrees(num_lvls: int):
    """Enumerate binary CStrees for given number of levels."""
    if num_lvls < 2:
        raise ValueError("CStrees with fewer than 2 variables aren't interesting.")
    if num_lvls == 2:
        return ({1: staging} for staging in enumerate_binary_stagings(1))
    return (
        {**cstree, **{num_lvls - 1: staging}}
        for staging in enumerate_binary_stagings(num_lvls - 1)
        for cstree in enumerate_binary_cstrees(num_lvls - 1)
    )
