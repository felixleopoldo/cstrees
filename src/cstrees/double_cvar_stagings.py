"""Enumerate CStrees and stagings with up to 2 context variables."""
from itertools import combinations, product, compress
from copy import deepcopy


def num_stagings(lvl: int):
    """Use formula to compute number of stagings at given level of binary CStree ."""
    return lvl**3 + 1 if lvl != 2 else 8


# def max2_cvars_stagings(var_outcomes: list, possible_cvars: tuple = None):
def codim_max2_boxes(box: list, splittable_dims: tuple = None):
    """Enumerate stagings at given level of CStree."""
    codim_0_box = [box]
    yield codim_0_box

    degen = False

    num_dims = len(box)
    sub_splittable_dims = reversed(tuple(combinations(range(num_dims), num_dims - 1)))
    z_cd1_boxes = zip(
        sub_splittable_dims, codim_1_boxes(box, splittable_dims, num_dims)
    )
    for possible_cvs, staging in z_cd1_boxes:
        yield staging

        num_stages = len(staging)
        for subset_size in range(1, num_stages):  # proper, nonempty subsets
            subsets = combinations(range(subset_size), subset_size)
            for subset in subsets:
                substaging = [staging[i] for i in subset]
                subposs_cvs = [possible_cvs[i] for i in subset]
                substagings = list(codim_1_boxes(substaging, subposs_cvs, num_dims))
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
            codim_1_boxes(staging[0], possible_cvs, num_dims),
            codim_1_boxes(staging[1], possible_cvs, num_dims),
        )  # need to replace var here with possible_cvs
        for substaging_0, substaging_1 in prod:
            if degen:
                break
            yield substaging_0 + substaging_1
            if len(box) == 2:
                degen = True


def codim_1_boxes(box: list, splittable_dims: tuple, num_dims: int):
    """Enumerate new stagings resulting from adding one cvar to given staging."""
    # assert num_dims == len(box)
    splittable_mask = (dim in splittable_dims for dim in range(num_dims))
    for dim, subbox in zip(splittable_dims, compress(box, splittable_mask)):
        yield [box[:dim] + [point] + box[dim + 1 :] for point in subbox]
