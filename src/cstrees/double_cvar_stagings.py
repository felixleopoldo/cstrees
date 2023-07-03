"""Enumerate CStrees and stagings with up to 2 context variables."""
from itertools import combinations, product, compress, zip_longest
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
                # the following loop splits all dims in the subset in
                # the same way, but could be that we need to loop
                # again and split it in all ways?
                for cd12_subdiv in codim_1_subdivs(cd1_subdiv, poss_split_dims, subset):
                    yield cd12_subdiv
                    # should be able to fit another for loop here
                # for fixed_dim, fixed_cd1_box in enumerate(cd1_box):
                #     to_split = [cd1_box[i] for i in sub_poss_split_dims]
                # # if len(sub_cd1_box) == 1:
                # #     sub_cd1_box = sub_cd1_box[0]
                # cd2_boxes = codim_1_subdivs(sub_cd1_box, sub_poss_split_dims)
                # for cd2_box in cd2_boxes:
                #     mask = (i in sub_poss_split_dims for i in range(num_dims))
                #     zipped = zip_longest(mask, cd2_box, cd1_box)
                #     yield [
                #         sub_cd1_box if sub_mask else cd2_box
                #         for (sub_mask, cd2_box, sub_cd1_box) in zipped
                #     ]
        if degen:
            break
        for cd2_subdiv in codim_1_subdivs(cd1_subdiv, poss_split_dims):
            yield cd2_subdiv
        if len(box) == 2:
            degen = True

        # prod = product(
        #     codim_1_subdivs(cd1_box[0], poss_split_dims),
        #     codim_1_subdivs(cd1_box[1], poss_split_dims),
        # )  # need to replace var here with possible_cvs
        # for substaging_0, substaging_1 in prod:
        #     if degen:
        #         break
        #     yield substaging_0 + substaging_1
        #     if len(box) == 2:
        #         degen = True


def codim_1_subdivs(
    box: list, splittable_dims: tuple, splittable_subboxes: tuple = None
):
    """Enumerate codimension-1 subdivisions of the given (subdivision of a) box."""
    if splittable_subboxes is None:
        splittable_subboxes = range(len(box))
    for dim in splittable_dims:
        points = box[0][dim]
        cd1_subdiv = []
        for subbox_idx, subbox in enumerate(box):
            if subbox_idx in splittable_subboxes:
                for point in points:
                    cd1_subdiv += [subbox[:dim] + [point] + subbox[dim + 1 :]]
            else:
                cd1_subdiv += [subbox]
        yield cd1_subdiv
        # yield [
        #     subbox[:dim] + [point] + subbox[dim + 1 :]
        #     if subbox_idx in splittable_subboxes
        #     else subbox
        #     for subbox_idx, subbox in enumerate(box)
        #     for point in points
        # ]
