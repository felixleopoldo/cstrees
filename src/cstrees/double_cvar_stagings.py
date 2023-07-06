"""Enumerate CStrees and stagings with up to 2 context variables."""
from itertools import combinations, product


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
    if splittable_dims is None:
        splittable_dims = range(num_dims)
    sub_splittable_dims = reversed(tuple(combinations(splittable_dims, num_dims - 1)))
    z_cd1_subdivs = zip(
        sub_splittable_dims, codim_1_subdivs(codim_0_box, splittable_dims)
    )
    for poss_split_dims, cd1_subdiv in z_cd1_subdivs:
        yield cd1_subdiv

        num_cd1_boxes = len(cd1_subdiv)
        for subset_size in range(1, num_cd1_boxes):
            subsets = combinations(range(num_cd1_boxes), subset_size)
            for subset in subsets:
                for cd12_subdiv in codim_1_subdivs(cd1_subdiv, poss_split_dims, subset):
                    yield cd12_subdiv
        if degen:
            break
        for cd2_subdiv in codim_1_subdivs(cd1_subdiv, poss_split_dims):
            yield cd2_subdiv
        if len(box) == 2:
            degen = True


def codim_1_subdivs(
    box: list, splittable_dims: tuple, splittable_subboxes: tuple = None
):
    """Enumerate codimension-1 subdivisions of the given (subdivision of a) box."""
    if splittable_subboxes is None:
        splittable_subboxes = range(len(box))
    for dims_to_split in product(*(splittable_dims for _ in splittable_subboxes)):
        cd1_subdiv = []
        for subbox_idx, subbox in enumerate(box):
            if subbox_idx in splittable_subboxes:
                dim = dims_to_split[splittable_subboxes.index(subbox_idx)]
                points = box[0][dim]
                for point in points:
                    cd1_subdiv += [subbox[:dim] + [point] + subbox[dim + 1 :]]
            else:
                cd1_subdiv += [subbox]
        yield cd1_subdiv
