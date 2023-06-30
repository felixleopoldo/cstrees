"""Enumerate CStrees and stagings with up to 2 context variables."""
from itertools import combinations, product, compress
from copy import deepcopy


def num_stagings(lvl: int):
    """Use formula to compute number of stagings at given level of binary CStree ."""
    return lvl**3 + 1 if lvl != 2 else 8


def max2_cvars_stagings(var_outcomes: list, possible_cvars: tuple = None):
    """Enumerate stagings at given level of CStree."""
    staging_no_context = [var_outcomes]
    yield staging_no_context

    degen = False

    num_vars = len(var_outcomes)
    sub_possible_cvs = reversed(tuple(combinations(range(num_vars), num_vars - 1)))
    z_stagings = zip(
        sub_possible_cvs, one_cvar_stagings(var_outcomes, possible_cvars, num_vars)
    )
    for possible_cvs, staging in z_stagings:
        yield staging

        num_stages = len(staging)
        for subset_size in range(1, num_stages):  # proper, nonempty subsets
            subsets = combinations(range(subset_size), subset_size)
            for subset in subsets:
                substaging = [staging[i] for i in subset]
                subposs_cvs = [possible_cvs[i] for i in subset]
                substagings = list(one_cvar_stagings(substaging, subposs_cvs, num_vars))
                print(f"\n{subset}")
                yield [
                    staging[i] if i in subset else substagings[i]
                    for i in range(num_stages)
                ]
        # z_substagings = zip(
        #     *(one_cvar_stagings(stage, possible_cvs, num_vars) for stage in staging)
        # )
        # for substagings in z_substagings:
        #     for stage in staging:
        #         for substaging in substagings:
        #             yield [stage] + substaging

        prod = product(
            one_cvar_stagings(staging[0], possible_cvs, num_vars),
            one_cvar_stagings(staging[1], possible_cvs, num_vars),
        )  # need to replace var here with possible_cvs
        for substaging_0, substaging_1 in prod:
            if degen:
                break
            yield substaging_0 + substaging_1
            if len(var_outcomes) == 2:
                degen = True


def one_cvar_stagings(staging: list, possible_cvars: tuple, num_vars: int):
    """Enumerate new stagings resulting from adding one cvar to given staging."""
    pcv_mask = (var in possible_cvars for var in range(num_vars))
    for var, stage in zip(possible_cvars, compress(staging, pcv_mask)):
        yield [staging[:var] + [outcome] + staging[var + 1 :] for outcome in stage]
