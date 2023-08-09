"""Enumerate stagings with up to 2 context variables ."""
from typing import Generator, Iterable
from itertools import combinations, product


def num_stagings(lvl: int) -> int:
    """Use formula to compute number of stagings at given level of binary CStree ."""
    return lvl**3 + 1 if lvl != 2 else 8


# def max2_cvars_stagings(var_outcomes: list, possible_cvars: tuple = None):
def codim_max2_boxes(cards: list, splittable_dims: Iterable[int] = []) -> Generator:
    """Enumerate stagings at given level of CStree."""
    box = [set(range(card)) for card in cards]

    codim_0_box = [box]
    yield codim_0_box

    degen = False

    num_dims = len(box)
    if len(splittable_dims) == 0:
        splittable_dims = range(num_dims)
    sub_split_len = len(splittable_dims) - 1
    sub_splittable_dims = reversed(tuple(combinations(splittable_dims, sub_split_len)))
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
    box: list, splittable_dims: Iterable[int], splittable_subboxes: list = []
) -> Generator:
    """Enumerate codimension-1 subdivisions of the given (subdivision of a) box."""
    if len(splittable_subboxes) == 0:
        splittable_subboxes = list(range(len(box)))
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
