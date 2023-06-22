import src.cstrees.double_cvar_stagings as dcs


def test_enumerate_binary_stagings():
    for n in range(10):
        stagings = dcs.enumerate_binary_stagings(n)
        assert sum(1 for staging in stagings) == dcs.num_binary_stagings(n)


def test_enumerate_binary_cstrees():
    for n in range(2, 6):
        cstrees = dcs.enumerate_binary_cstrees(n)
        assert sum(1 for cstree in cstrees) == dcs.num_binary_cstrees(n)
