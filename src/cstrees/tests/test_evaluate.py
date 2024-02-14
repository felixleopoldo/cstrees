import random

import numpy as np

from cstrees import cstree as ct
from cstrees.evaluate import kl_divergence


def test_kl_divergence():
    np.random.seed(22)
    random.seed(22)

    cards = [3, 2, 2, 3]

    # KL-divergence is 0 when models are identical
    t = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=2)

    t.sample(1000)

    assert kl_divergence(t, t) == 0

    # KL-divergence is positive nonzero whet models are different
    e = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    e.sample_stage_parameters(alpha=2)

    e.sample(1000)
    assert kl_divergence(e, t) > 0

    # Conditional probabilities exist even when all outcomes haven't
    # been observed
    s = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    s.sample_stage_parameters(alpha=2)

    s.sample(35)
    kl_divergence(s, t)  # shouldn't raise KeyError
