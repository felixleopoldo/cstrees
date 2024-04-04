"""Evaluate estimated CStrees."""

from itertools import product, pairwise
from functools import reduce

import numpy as np
from scipy.special import rel_entr

from cstrees.cstree import CStree

def KL_divergence(df_distr1, df_distr2):
    """Calculate the KL divergence between two distributions using scipy rel_entr.
    df_distr2 is typically the true distribution and df_distr1 is the estimated distribution.
    """

    distr1 = df_distr1["prob"].values
    distr2 = df_distr2["prob"].values
    
    log_distr1 = df_distr1["log_prob"].values
    log_distr2 = df_distr2["log_prob"].values


    return np.sum(rel_entr(distr1, distr2))

def kl_divergence(estimated: CStree, true: CStree) -> float:
    """Quantify how distribution of estimated CStree differs from that of true CStree.

    Notes:
    See the `KL divergence
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.
    """
    factorized_outcomes = (range(card) for card in true.cards)
    outcomes = product(*factorized_outcomes)

    def _rel_entr_of_outcome(outcome):
        assert true.labels == sorted(true.labels)
        true_nodes = (outcome[:idx] for idx in range(true.p + 1))
        true_edges = pairwise(true_nodes)

        est_ordered_outcome = tuple(outcome[i] for i in estimated.labels)
        est_nodes = (est_ordered_outcome[:idx] for idx in range(true.p + 1))
        est_edges = pairwise(est_nodes)

        def _probs_map(zipped_edge):
            est_edge, true_edge = zipped_edge
            try:
                est = estimated.tree[est_edge[0]][est_edge[1]]["cond_prob"]
            except KeyError:
                stage = estimated.get_stage(est_edge[0])
                est = stage.probs[est_edge[1][-1]]
            try:
                tru = true.tree[true_edge[0]][true_edge[1]]["cond_prob"]
            except KeyError:
                stage = true.get_stage(true_edge[0])
                tru = stage.probs[true_edge[1][-1]]
            return est, tru

        zipped_edges = zip(est_edges, true_edges)
        zipped_probs = map(_probs_map, zipped_edges)

        def _probs_of_outcome(prev_pair, next_pair):
            return prev_pair[0] * next_pair[0], prev_pair[1] * next_pair[1]

        est_prob_outcome, true_prob_outcome = reduce(_probs_of_outcome, zipped_probs)
        return rel_entr(est_prob_outcome, true_prob_outcome)

    return sum(map(_rel_entr_of_outcome, outcomes))
