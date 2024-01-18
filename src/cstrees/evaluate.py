"""Evaluate estimated CStrees."""
from itertools import product, pairwise, tee
import operator

from scipy.special import rel_entr

from cstrees.cstree import CStree


def kl_divergence(estimated: CStree, true: CStree) -> float:
    """Quantify how distribution of estimated CStree differs from that of true CStree.

    Notes:
    See the `KL divergence
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.
    """
    factorized_outcomes = (range(card) for card in enumerate(true.cards))
    outcomes = product(*factorized_outcomes)

    def _rel_entr_of_outcome(outcome):
        nodes = (outcome[:idx] for idx in range(true.p + 1))
        edges = pairwise(nodes)

        def _prob_map(edge):
            est = estimated.tree[edge[0]][edge[1]]["cond_prob"]
            tru = true.tree[edge[0]][edge[1]]["cond_prob"]
            return est, tru

        zipped_probs = map(_prob_map, edges)
        estimated_probs, true_probs = tee(zipped_probs)

        estimated_prob_outcome = reduce(operator.mul, estimated_probs)
        true_prob_outcome = reduce(operator.mul, true_probs)
        return rel_entr(estimated_prob_outcome, true_prob_outcome)

    return sum(map(_rel_entr_of_outcome, outcomes)
