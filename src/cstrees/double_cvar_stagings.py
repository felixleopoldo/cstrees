"""Enumerate CStrees and stagings with up to 2 context variables."""
from itertools import combination, product, compress


def num_stagings(lvl: int):
    """Use formula to compute number of stagings at given level of binary CStree ."""
    return lvl**3 + 1 if lvl != 2 else 8


def max2_cvars_stagings(var_outcomes: list, possible_cvars: tuple = None):
    """Enumerate stagings at given level of CStree."""
    if restricted_to_cvars is None:
        num_vars = len(var_outcomes)
        restricted_to_cvars = reversed(tuple(combination(num_vars, num_vars - 1)))
    degen = False
    staging_no_context = [var_outcomes]
    yield staging_no_context
    for var, staging in enumerate(one_cvar_stagings(var_outcomes, possible_cvars)):
        yield staging
        for stage in staging:
            yield [stage] + None
            pass
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


def one_cvar_stagings(staging: list, possible_cvars: tuple):
    """Enumerate new stagings resulting from adding one cvar to given staging."""
    num_vars = len(staging[0])
    pcv_mask = (var in possible_cvars for var in range(num_vars))
    for var, stage in zip(possible_cvars, compress(staging, pcv_mask)):
        yield [staging[:var] + [outcome] + staging[var + 1 :] for outcome in stage]


"""notes for finishing generalization beyond binary vars with unrestricted cvar set:
- change from excluding fixed_var to only including unfixed_vars
  - should be input list of (lists or sets), or dict?
  - account for this inside inner loops somehow
- prod and (maybe) zip still useful
- need to add loop within zip (like for max1_cvar case)
"""
