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
    sub_splittable_dims = reversed(tuple(combinations(range(num_dims), num_dims - 1)))
    z_cd1_boxes = zip(
        sub_splittable_dims, codim_1_boxes(box, splittable_dims, num_dims)
    )
    for poss_split_dims, cd1_box in z_cd1_boxes:
        yield cd1_box

        num_poss_split_dims = len(poss_split_dims)
        for subset_size in range(1, num_poss_split_dims + 1):
            subsets = combinations(poss_split_dims, subset_size)
            for sub_poss_split_dims in subsets:
                sub_cd1_box = [cd1_box[i] for i in sub_poss_split_dims]
                if len(sub_cd1_box) == 1:
                    sub_cd1_box = sub_cd1_box[0]
                cd2_boxes = codim_1_boxes(sub_cd1_box, sub_poss_split_dims, num_dims)
                print("test", list(cd2_boxes))
                for cd2_box in cd2_boxes:
                    mask = (i in sub_poss_split_dims for i in range(num_dims))
                    zipped = zip_longest(mask, cd2_box, cd1_box)
                    yield [
                        sub_cd1_box if sub_mask else cd2_box
                        for (sub_mask, cd2_box, sub_cd1_box) in zipped
                    ]

        prod = product(
            codim_1_boxes(cd1_box[0], poss_split_dims, num_dims),
            codim_1_boxes(cd1_box[1], poss_split_dims, num_dims),
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
