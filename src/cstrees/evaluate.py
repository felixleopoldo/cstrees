"""Evaluate estimated CStrees."""
from itertools import product, pairwise, tee
from functools import reduce
import operator

from scipy.special import rel_entr

from cstrees.cstree import CStree


def kl_divergence(estimated: CStree, true: CStree) -> float:
    """Quantify how distribution of estimated CStree differs from that of true CStree.

    Notes:
    See the `KL divergence
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.
    """
    factorized_outcomes = (range(card) for card in true.cards)
    outcomes = product(*factorized_outcomes)

    def _rel_entr_of_outcome(outcome):
        nodes = (outcome[:idx] for idx in range(true.p + 1))
        edges = pairwise(nodes)

        def _probs_map(edge):
            est = estimated.tree[edge[0]][edge[1]]["cond_prob"]
            tru = true.tree[edge[0]][edge[1]]["cond_prob"]
            return est, tru

        zipped_probs = map(_probs_map, edges)

        def _probs_of_outcome(prev_pair, next_pair):
            return prev_pair[0] * next_pair[0], prev_pair[1] * next_pair[1]

        est_prob_outcome, true_prob_outcome = reduce(_probs_of_outcome, zipped_probs)
        return rel_entr(est_prob_outcome, true_prob_outcome)

    return sum(map(_rel_entr_of_outcome, outcomes))


# because CStrees are created on the fly while sampling, computing KL
# divergence (or making prediction) may produce key error; can catch
# these errors and set prob of corresponding outcome to 0? or sample
# more? or some other way to generate full tree?
