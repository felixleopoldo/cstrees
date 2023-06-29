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


def max2_cvars_stagings_new(var_outcomes: list, restricted_to_cvars: tuple = None):
    """Enumerate stagings at given level of CStree."""
    var_outcomes = [{0, 1}] * (var_outcomes - 1)  # remove this eventually
    degen = False
    staging_no_context = [var_outcomes]
    yield staging_no_context
    for var, staging in enumerate(one_cvar_stagings(var_outcomes)):
        yield staging
        for substaging_0 in one_cvar_stagings(staging[0], var):
            yield [staging[1]] + substaging_0
            first = True
            for substaging_1 in one_cvar_stagings(staging[1], var):
                if first:
                    # makes sure each substaging_1 gets recorded once,
                    # despite being inside of the loop over
                    # substaging_0
                    yield [staging[0]] + substaging_1
                    first = False
                    if degen:
                        break
                    elif len(var_outcomes) == 2:
                        degen = True
                yield substaging_0 + substaging_1


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


def one_cvar_stagings(staging: list, fixed_var: int = None):
    """Enumerate new stagings resulting from adding one cvar to given staging."""
    for var, stage in enumerate(staging):
        if var == fixed_var:
            continue
        substage_0, substage_1 = deepcopy(staging), deepcopy(staging)
        substage_0[var] = 0
        substage_1[var] = 1
        yield [substage_0, substage_1]


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


"""notes for finishing generalization beyond binary vars with unrestricted cvar set:
- change from excluding fixed_var to only including unfixed_vars
  - should be input list of (lists or sets), or dict?
  - account for this inside inner loops somehow
- prod and (maybe) zip still useful
- need to add loop within zip (like for max1_cvar case)
"""
