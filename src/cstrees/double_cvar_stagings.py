"""Enumerate CStrees and stagings with up to 2 context variables."""
from itertools import combination, product, compress


def num_stagings(lvl: int):
    """Use formula to compute number of stagings at given level of binary CStree ."""
    return lvl**3 + 1 if lvl != 2 else 8


def max2_cvars_stagings(var_outcomes: list, possible_cvars: tuple = None):
    """Enumerate stagings at given level of CStree."""
    staging_no_context = [var_outcomes]
    yield staging_no_context
    degen = False
    num_vars = len(var_outcomes)
    sub_possible_cvs = reversed(tuple(combination(num_vars, num_vars - 1)))
    zipped = zip(sub_possible_cvs, one_cvar_stagings(var_outcomes, possible_cvars))
    for possible_cvs, staging in zipped:
        yield staging
        zipped = zip(
            one_cvar_stagings(staging[0], var), one_cvar_stagings(staging[1], var)
        )  # need to replace var here with possible_cvs
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
