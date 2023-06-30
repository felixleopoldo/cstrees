import src.cstrees.double_cvar_stagings as dcs


def test_max2_cvars_stagings_binary():
    for lvl in range(1, 10):
        num_binary_stagings = lvl**3 + 1 if lvl != 2 else 8
        var_outcomes = [{0, 1}] * lvl
        stagings = list(dcs.codim_max2_boxes(var_outcomes))
        assert num_binary_stagings == len(stagings)
        for idx, staging in enumerate(stagings):
            assert staging not in stagings[idx + 1 :]


def writing_test(n):
    n = 2
    box = [{0, 1}] * n
    splittable_dims = tuple(range(n))
    for subbox in dcs.codim_max2_boxes(box, splittable_dims):
        print(subbox)
