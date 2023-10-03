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
    empty_outcome = {}
    t.predict(empty_outcome)
    t.predict(empty_outcome, True)

    # test complete
    complete_outcome = {idx: outcome for idx, outcome in enumerate((2, 0, 1, 1))}
    t.predict(complete_outcome, True)

    # partials
    partial_outcome_1 = {0: 1}
    t.predict(partial_outcome_1, True)

    partial_outcome_2 = {0: 1, 3: 2}
    t.predict(partial_outcome_2, True)

    partial_outcome_3 = {0: 1, 2: 0, 3: 2}
    t.predict(partial_outcome_3, True)
