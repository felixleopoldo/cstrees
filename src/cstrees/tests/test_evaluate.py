import random

import numpy as np

from cstrees import cstree as ct
from cstrees.evaluate import kl_divergence


def test_kl_divergence():
    np.random.seed(22)
    random.seed(22)

    cards = [3, 2, 2, 3]

    t = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=2)

    t.sample(1000)

    assert kl_divergence(t, t) == 0

    e = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    e.sample_stage_parameters(alpha=2)

    e.sample(1000)
    assert kl_divergence(e, t) > 0
