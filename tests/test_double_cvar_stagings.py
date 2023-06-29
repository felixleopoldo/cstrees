import src.cstrees.double_cvar_stagings as dcs


def test_enumerate_binary_stagings():
    for n in range(10):
        stagings = dcs.enumerate_binary_stagings(n)
        assert sum(1 for staging in stagings) == dcs.num_binary_stagings(n)


def test_enumerate_binary_cstrees():
    for n in range(2, 6):
        cstrees = dcs.enumerate_binary_cstrees(n)
        assert sum(1 for cstree in cstrees) == dcs.num_binary_cstrees(n)


def test_max2_cvars_stagings():
    for n in range(4, 10):
        old_stagings = list(dcs.max2_cvars_stagings_old(n))
        new_stagings = list(dcs.max2_cvars_stagings(n))
        assert len(new_stagings) == len(old_stagings)
        zipped = zip(old_stagings, new_stagings)
        for idx, (old_staging, new_staging) in enumerate(zipped):
            assert old_staging in new_stagings
            assert new_staging in old_stagings
            assert new_staging not in new_stagings[idx + 1 :]
            # try:
            #     assert old_staging in new_stagings
            # except AssertionError:
            #     print(f"{old_staging} not in `new_stagings`")
            # try:
            #     assert new_staging in old_stagings
            # except AssertionError:
            #     print(f"{new_staging} not in `old_stagings`")
            # try:
            #     assert new_staging not in new_stagings[idx + 1 :]
            # except AssertionError:
            #     print(f"{new_staging} repeated in `new_stagings`")
