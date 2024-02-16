# import pytest
import random

import numpy as np

import cstrees
from cstrees import cstree as ct


def test_project_defines_author_and_version():
    assert hasattr(cstrees, "__author__")
    assert hasattr(cstrees, "__version__")


def test_predict():
    np.random.seed(22)
    random.seed(22)

    cards = [3, 2, 2, 3]

    t = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=2)

    t.sample(100)

    # test empty
    empty_observation = {}
    assert t.predict(empty_observation) == (2, 0, 1, 1)
    assert t.predict(empty_observation, True) == ((2, 0, 1, 1), 0.11590809641032397)

    # test complete
    complete_observation = {0: 2, 1: 0, 2: 1, 3: 1}
    assert t.predict(complete_observation, True) == ((2, 0, 1, 1), 1.0)

    # partials
    partial_observation_1 = {0: 1}
    t.predict(partial_observation_1, True)
    assert t.predict(partial_observation_1, True) == ((1, 0, 1, 1), 0.26674411494959)

    partial_observation_2 = {0: 1, 3: 2}
    assert t.predict(partial_observation_2, True) == ((1, 1, 1, 2), 0.3829026812192928)

    partial_observation_3 = {0: 1, 2: 0, 3: 2}
    assert t.predict(partial_observation_3, True) == ((1, 0, 0, 2), 0.5913854582044948)

    # test conditional probs exist from small sample
    s = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    s.sample_stage_parameters(alpha=2)

    s.sample(35)

    s.predict({})  # shouldn't raise KeyError
