"""Enumerate CStrees and stagings with up to 2 context variables."""
from copy import deepcopy
from itertools import product


def num_stagings(lvl: int):
    """Use formula to compute number of stagings at given level of binary CStree ."""
    return lvl**3 + 1 if lvl != 2 else 8


def max2_cvars_stagings(var_outcomes: list, restricted_to_cvars: tuple = None):
    """Enumerate stagings at given level of CStree."""
    degen = False
    staging_no_context = [var_outcomes]
    yield staging_no_context
    for var, staging in enumerate(one_cvar_stagings(var_outcomes)):
        yield staging
        zipped = zip(
            one_cvar_stagings(staging[0], var), one_cvar_stagings(staging[1], var)
        )
        for substaging_0, substaging_1 in zipped:
            yield [staging[0]] + substaging_1
            yield [staging[1]] + substaging_0
        prod = product(
            one_cvar_stagings(staging[0], var), one_cvar_stagings(staging[1], var)
        )
        for substaging_0, substaging_1 in prod:
            if degen:
                break
            yield substaging_0 + substaging_1
            if len(var_outcomes) == 2:
                degen = True


def one_cvar_stagings(staging: list, fixed_var: int = None):
    """Enumerate new stagings resulting from adding one cvar to given staging."""
    for var, stage in enumerate(staging):
        if var == fixed_var:
            continue
        substage_0, substage_1 = deepcopy(staging), deepcopy(staging)
        substage_0[var] = 0
        substage_1[var] = 1
        yield [substage_0, substage_1]


"""notes for finishing generalization beyond binary vars with unrestricted cvar set:
- change from excluding fixed_var to only including unfixed_vars
  - should be input list of (lists or sets), or dict?
  - account for this inside inner loops somehow
- prod and (maybe) zip still useful
- need to add loop within zip (like for max1_cvar case)
"""
