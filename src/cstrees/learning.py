from cstrees.cstree import *
from cstrees.csi_relation import *
from cstrees.stage import *
from cstrees.scoring import *

def n_stagings(p, cards, l, max_cvars=1):
    """ Returns the number of stagings at a given level.
        p: number of variables
        cards: list of cardinalities of the variables
        l: level
        cvars: number of context variables
    """
    
    stagings = all_stagings(p, cards, l, max_cvars)
    return sum(len(staging) for staging in stagings)


def optimal_staging_at_level(order, cards, data, l, max_cvars=1, alpha_tot=None, method="BDeu"):
    p = len(order)
    #co = CausalOrder(cards)
    tree = CStree(cards)
    tree.labels = order
    # tree.set_cardinalities(cards)

    stagings = all_stagings(p, cards, l, max_cvars)  # at level l
    max_staging = None
    max_staging_score = -np.inf
    #logging.debug("level: {}".format(l))

    for stlist in stagings:
        #logging.debug("scoring staging: {}".format([str(ss) for ss in stlist]))
        tree.update_stages({l: stlist})
        # This needs the stages to be set at the line above.
        level_counts = sc.counts_at_level(tree, l+1, data)
        #logging.debug("all counts in the staging")
        # for k, v in level_counts.items():
        #    logging.debug("{}: {}".format(k, v))
        score = sc.score_level(tree, l+1, level_counts, alpha_tot, method)
        #logging.debug("score: {}".format(score))
        if score > max_staging_score:
            #logging.debug("{} is current max".format(score))
            max_staging_score = score
            max_staging = stlist

    return max_staging, max_staging_score


def optimal_cstree(order, data, max_cvars=1, alpha_tot=1.0, method="BDeu"):

    # Order should use the same labels as data?
    # Order is only intresting when data is involved. Then its the order of the data columns
    # Otherwise we only talk about levels.

    # BUG?: Maybe these have to be adapted to the order.
    cards = data.iloc[0].values
    p = len(order)

    stages = {}
    stages[-1] = [Stage([])]
    for level in range(-1, p-1):  # dont stage the last level
        max_staging, max_staging_score = optimal_staging_at_level(
            order, cards, data, level, max_cvars, alpha_tot, method)
        stages[level] = max_staging
        print("max staging: {}".format([str(s) for s in max_staging]))

    # Create CStree
    tree = CStree(cards)
    tree.labels = order
    tree.update_stages(stages)
    # maybe set level labels here?
    return tree


def find_optimal_order(data, strategy="max", max_cvars=1, alpha_tot=1, method="BDeu"):
    """ Find the optimal causal order for the data.
    """

    p = data.shape[1]
    perms = permutations(list(range(p)))
    labels = data.columns.values
    cards = data.iloc[0].values
    optimal_order = None
    max_score = -np.inf

    # iterate over all permutations
    for perm in list(perms):
        # dont stage the last variable. What do i mean by this? /Felix
        order = list(perm)
        # maybe it should be the indices instead...
        order = [labels[i] for i in order]

        #print("scoring order: {}".format(order))
        score = sc.score_order(order, data, strategy="max",
                               max_cvars=max_cvars, alpha_tot=alpha_tot, method=method)
        print("order: {}, score: {}".format(order, score))

        # if score == max_score:
        #    optimal_orders.append(order)
        if score > max_score:
            max_score = score
            optimal_orders = order

    return optimal_orders, max_score


def find_optimal_cstree(data, max_cvars=1, alpha_tot=1, method="BDeu"):
    opt_order = find_optimal_order(
        data, max_cvars=max_cvars, alpha_tot=alpha_tot, method=method)
    print("optimal order: {}".format(opt_order))

    optimal_staging = optimal_cstree(
        opt_order, data, max_cvars=max_cvars, alpha_tot=alpha_tot, method=method)
    print("optimal staging: {}".format(optimal_staging))
