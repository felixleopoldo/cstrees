from itertools import permutations
import random

import numpy as np
from tqdm import tqdm

import cstrees.cstree as ct
import cstrees.stage as stl
import cstrees.scoring as sc


def all_stagings(cards, level, max_cvars=1, poss_cvars=[]):
    """ Returns a generator over all stagings of a given level.

    Args:
        l (int): The level of the stage.
        cards (list): List of cardinalities of the variables. Should be at least of length l+1. E.g.: l=2, cards=[2,2,2,2]
        max_cvars (int, optional): The maximum number of context variables . Defaults to 1.
        poss_cvars (list, optional): The possible context variables. Defaults to None.
    Raises:
        NotImplementedError: Exception if max_cvars > 2.

    Yields:
        generator: generator over all stagings of a given level.

    Examples:
        A staging with 2 stages for a binary CStree at level 2
        (numbering levels from 0) could e.g. be:

        >>> import cstrees.learning as ctl
        >>> cards = [2]*4
        >>> stagings = ctl.all_stagings(cards, 2, max_cvars=2)
        >>> for i, staging in enumerate(stagings):
        >>>     print("staging: {}".format(i))
        >>>     for stage in staging:
        >>>         print(stage)
        staging: 0
        [{0, 1}]
        [{0, 1}]
        staging: 1
        [0, {0, 1}]
        [1, {0, 1}]
        staging: 2
        [{0, 1}, 0]
        [{0, 1}, 1]
        staging: 3
        [1, {0, 1}]
        [0, 0]
        [0, 1]
        staging: 4
        [0, {0, 1}]
        [1, 0]
        [1, 1]
        staging: 5
        [{0, 1}, 1]
        [0, 0]
        [1, 0]
        staging: 6
        [{0, 1}, 0]
        [0, 1]
        [1, 1]
        staging: 7
        [0, 0]
        [1, 0]
        [0, 1]
        [1, 1]

    """
    
    assert level < len(cards)
    #assert level < len(cards)-1
    # if max_cvars == 1: # I guess this is not needed as a special case anymore, Alex?
    #     if level == -1:  # This is an imaginary level -1, it has no stages.
    #         yield [stl.Stage([])]
    #         return

    #     # All possible values for each variable

    #     vals = [list(range(cards[lev])) for lev in range(len(cards))]
    #     for k in range(level+1):  # all variables up to l can be context variables
    #         # When we restrict to max_cvars = 1, we have two cases:
    #         # Either all are in 1 color or all are in 2 different colors.
    #         stlist = []  # The staging: list of stl.Stages.
    #         # Loop through the values of the context variables.
    #         for v in vals[k]:
    #             left = [set(vals[i]) for i in range(k)]
    #             right = [set(vals[j]) for j in range(k+1, level+1)]
    #             # For example: [{0,1}, {0,1}, 0, {0, 1}]
    #             stagelistrep = left + [v] + right
    #             st = stl.Stage(stagelistrep)
    #             stlist += [st]
    #         yield stlist

    #     # The staging with no context variables
    #     stagelistrep = [set(v) for v in vals][:level+1]

    #     st = stl.Stage(stagelistrep)
    #     yield [st]
    # elif max_cvars == 2:
    if max_cvars == 1:
        print("max_cvars: {}".format(max_cvars))
        print("poss_cvars: {}".format(poss_cvars))
        #raise NotImplementedError("max_cvars <= 1 not implemented yet")
        from cstrees.double_cvar_stagings import codim_1_subdivs
        if level == -1:  # This is an imaginary level -1, it has no stages.
            yield [stl.Stage([])]
            return
        # BUG: Should also provide max_cvars!!
        for staging_list in codim_1_subdivs(box=[set(range(c)) for c in cards[:level+1]], 
                                            splittable_dims=poss_cvars):

            staging = []
            for stage_list in staging_list:
                # Fix repr bug
                if isinstance(stage_list, set):
                    stage_list = [stage_list]

                # could set colors here cut that takes time maybe.
                stage = stl.Stage(stage_list)
                staging.append(stage)
            yield staging
        
    elif max_cvars == 2:
        from cstrees.double_cvar_stagings import codim_max2_boxes
        if level == -1:  # This is an imaginary level -1, it has no stages.
            yield [stl.Stage([])]
            return
        # BUG: Should also provide max_cvars!!
        for staging_list in codim_max2_boxes(cards[:level+1], splittable_coords=poss_cvars):

            staging = []
            for stage_list in staging_list:
                # Fix repr bug
                if isinstance(stage_list, set):
                    stage_list = [stage_list]

                # could set colors here cut that takes time maybe.
                stage = stl.Stage(stage_list)
                staging.append(stage)
            yield staging

    else:
        raise NotImplementedError("max_cvars > 2 not implemented yet")


def n_stagings(cards, level, max_cvars=1):
    """ Returns the number of stagings at a given level.

    Args:
        p (int): Number of variables.
        cards (list): List of cardinalities of the variables.
        level (int): The level in the CStree.
        cvars (int, optional): The maximum number of context variables. Defaults to 1.

    Examples:
        >>> import cstrees.learning as ctl
        >>> cards = [2]*4
        >>> ctl.n_stagings(cards, 2, max_cvars=2)
        8

    """

    stagings = all_stagings(cards, level, max_cvars)

    return sum(1 for _ in stagings)



def _optimal_staging_at_level(order, context_scores, level, max_cvars=2, poss_cvars=[]):
    """Find the optimal staging at a given level.

    Args:
        order (list): The order of the variables. data (pandas DataFrame): The
        data as a pandas DataFrame.
        level (int): The level of the CStree.
        max_cvars (int, optional): Max context variables. Defaults to 1.
        alpha_tot (float, optional): The Dirichlet hyper parameter total pseudo
        counts. Defaults to None.
        method (str, optional): Parameter prior type.
        Defaults to "BDeu".

    Returns:
        tuple: (optimal staging, optimal score)

    """
    cards = [context_scores["cards"][var] for var in order]
    
    var = order[level+1] 
   
    poss_cvars_inds = [i for i,j in enumerate(order) if j in poss_cvars and i<=level]
    #print("poss cvars inds: {}".format(poss_cvars_inds))
    stagings = all_stagings(cards, level, max_cvars, poss_cvars=poss_cvars_inds)
    max_staging = None
    max_staging_score = -np.inf
    
    for staging in stagings:        
        staging_score = 0
        for stage in staging:
            if stage.level == -1:
                staging_score = context_scores["scores"][var]["None"]
                continue
            print("stage: {}".format(stage))
            # here we (=I) anyway extract just the context, so the stage format is a bit redundant.
            stage_context = sc.stage_to_context_key(stage, order) # BUG: something wrong somewhere
            print(stage_context)
            score = context_scores["scores"][var][stage_context]
            staging_score += score
        
        # Update the max score and the max staging
        if staging_score > max_staging_score:
            max_staging_score = staging_score
            max_staging = staging

    return max_staging, max_staging_score


def _optimal_cstree_given_order(order, context_scores):
    """Find the optimal CStree for a given order.

    Args:
        order (list): The order of the variables.
        data (pandas DataFrame): The data as a pandas DataFrame.
        max_cvars (int, optional): Max context variables. Defaults to 1.
        alpha_tot (float, optional): The Dirichlet hyper parameter total pseudo
        counts. Defaults to 1.0.
        method (str, optional): Parameter prior type. Defaults to "BDeu".

    """

    # BUG?: Maybe these have to be adapted to the order.
    #cards = data.iloc[0].values
    #cards = data.loc[0, order]
    #cards_dict = {var: card for var, card in zip(order, cards)}
    p = len(order)
    #print("order: {}".format(order))
    stages = {}
    stages[-1] = [stl.Stage([], color="black")]
    for level in range(-1, p-1):  # dont stage the last level
        #print("\nstaging level: {}".format(level))
        #print("var {} (one level above)".format(order[level+1]))
        #print("potential cvars of {}: {}".format(order[level+1], context_scores["poss_cvars"][order[level+1]]) )
        max_staging, max_staging_score = _optimal_staging_at_level(
            order, context_scores, level, 
            max_cvars=context_scores["max_cvars"], 
            poss_cvars=context_scores["poss_cvars"][order[level+1]])
        stages[level] = max_staging
        #print("max staging: {}".format([str(s) for s in max_staging]))

    # Create CStree
    tree = ct.CStree([context_scores["cards"][var] for var in order])
    tree.labels = order

    # Color each stage in the optimal staging. Singletons are black.
    # This should be done somewhere else probably.
    colors = ['peru','blueviolet', 'orange', 'navy', 'rebeccapurple', 'darkseagreen',
              'darkslategray', 'lightslategray', 'aquamarine',
              'lightgoldenrodyellow', 'cornsilk', 'azure', 'chocolate',
              'red', 'darkolivegreen']

    for level, staging in stages.items():
        for i, stage in enumerate(staging):
            #print("level: {}, stage: {}".format(level, stage))
            if (level==-1) or ((level>0) and all([isinstance(i, int) for i in stage.list_repr])):
                stage.color = "black"
            else:
                stage.color = colors[i]
            #print("stage color: {}".format(stage.color))
    tree.update_stages(stages)

    return tree


def _find_optimal_order(score_table, poss_cvars=None):
    """ Find the optimal causal order for the data using exhaustive search of
        the optimal order then the CStree having that order.

    Args:
        data (pandas DataFrame): The data as a pandas DataFrame.
        strategy (str, optional): The scoring strategy to use. Defaults to "max" which mean that the score of an order is the score of the maximal scoring CStree it can contain.
        max_cvars (int, optional): Max context variables. Defaults to 1.
        alpha_tot (float, optional): The Dirichlet hyper
        parameter total pseudo counts. Defaults to 1.
        method (str, optional): Parameter prior type. Defaults to "BDeu".
    Examples:
        >>> import cstrees.learning as ctl
        >>> optord, score = ctl.find_optimal_order(
        >>> df, strategy="max", max_cvars=2, alpha_tot=1.0, method="BDeu")
        >>> print("optimal order: {}, score {}".format(optord, score))

    """
    labels = list(score_table["scores"].keys())
    #p = len(score_table.keys())
    #p = data.shape[1]
    #perms = permutations(list(range(p)))
    perms = permutations(labels)
    #labels = data.columns.values
    optimal_orders = None
    max_score = -np.inf

    # iterate over all permutations
    for perm in list(perms):
        # dont stage the last variable. What do i mean by this? /Felix
        order = list(perm)
        # maybe it should be the indices instead...
        #order = [labels[i] for i in order]

        #print("scoring order: {}".format(order))
        score = sc.score_order(order, score_table)
        #print("order: {}, score: {}".format(order, score))

        # if score == max_score:
        #    optimal_orders.append(order)
        if score > max_score:
            max_score = score
            optimal_orders = order

    return optimal_orders, max_score


def relocate_node_in_order(order, node, new_pos):
    """ Relocate a node in an order.

    Args:
        order (list): The order of the variables.
        node (str): The node to relocate.
        new_pos (int): The new position of the node.

    Returns:
        list: The new order.
    """

    order.remove(node)
    order.insert(new_pos, node)
    return order


def move_up(node_index,
             order,
             orderscore,
             node_scores,
             score_table):
    """ Move a node up in an order.
    """
    order.insert(node_index+1, order.pop(node_index))

    tmp1 = node_scores[node_index]
    tmp2 = node_scores[node_index+1]

    #print("order {}".format(order))
    node1 = order[node_index]
    node2 = order[node_index+1]
    #active_cvars1 = list(set(order[:node_index]) & set(score_table["poss_cvars"][node1]))
    active_cvars1 = [v for v in order[:node_index] if v in score_table["poss_cvars"][node1]]
    
    #active_cvars2 = list(order[:node_index+1] & set(score_table["poss_cvars"][node2]))
    active_cvars2 = [v for v in order[:node_index+1] if v in score_table["poss_cvars"][node2]]
    pred1 = sc.list_to_score_key(active_cvars1)
    pred2 = sc.list_to_score_key(active_cvars2)
    
    #pred1 = sc.list_to_score_key(order[:node_index])
    #pred2 = sc.list_to_score_key(order[:node_index+1])

    node_scores[node_index] = score_table["scores"][order[node_index]][pred1]
    node_scores[node_index+1] = score_table["scores"][order[node_index+1]][pred2]

    orderscore += node_scores[node_index] + \
        node_scores[node_index+1] - tmp1 - tmp2

    return orderscore


def move_down(node_index,
               order,
               orderscore,
               node_scores,
               score_table):

    # prev scores
    tmp1 = node_scores[node_index]
    tmp2 = node_scores[node_index-1]
    # move the node
    order.insert(node_index-1, order.pop(node_index))

    active_cvars1 = [v for v in order[:node_index] if v in score_table["poss_cvars"][order[node_index]]]
    active_cvars2 = [v for v in order[:node_index-1] if v in score_table["poss_cvars"][order[node_index-1]]]

    # pred1 = sc.list_to_score_key(order[:node_index])
    # pred2 = sc.list_to_score_key(order[:node_index-1])
    
    pred1 = sc.list_to_score_key(active_cvars1)
    pred2 = sc.list_to_score_key(active_cvars2)

    node_scores[node_index] = score_table["scores"][order[node_index]][pred1]
    node_scores[node_index-1] = score_table["scores"][order[node_index-1]][pred2]

    orderscore += node_scores[node_index] + \
        node_scores[node_index-1] - tmp1 - tmp2

    return orderscore


def move_node(node_index_from,
              node_index_to,
              order,
              orderscore,
              node_scores,
               score_table):
    """ Move a node up in an order and update the node scores.

    Args:
        node_index_from (int): The index of the node to move.
        node_index_to (int): The index to move the node to.
        order (list): The order of the variables.
        orderscore (float): The score of the order.
        node_scores (list): The scores of the nodes.
        data (pandas.DataFrame): The data.
        max_cvars (int, optional): The maximum number of context variables. Defaults to 2.
        alpha_tot (float, optional): The total alpha. Defaults to 1.
        method (str, optional): The scoring method. Defaults to "BDeu".

    """

    if node_index_from < node_index_to:
        for i in range(node_index_from, node_index_to):
            orderscore = move_up(i, order, orderscore, node_scores, score_table)
    else:
        for i in range(node_index_from, node_index_to, -1):
            orderscore = move_down(
                i, order, orderscore, node_scores, score_table)
    return orderscore


def swap_neigbors_in_order(order, node_index1, node_index2):
    """ Swap two neighbors in an order.

    Args:
        order (list): The order of the variables.
        node_index1 (int): The index of the first node.
        node_index2 (int): The index of the second node.

    """


def gibbs_order_sampler(iterations, score_table):
    """ Gibbs order sampler.
    """
    # Score table for all noded in all positions in the order
    # score_table = sc.order_score_tables(data, max_cvars=max_cvars,
    #                                         alpha_tot=alpha_tot,
    #                                         method=method)

    order_trajectory = []
    #p = data.shape[1]
    p = len(score_table["scores"])

    #order = list(data.columns.values)  # list(range(p))
    order = list(score_table["scores"].keys()) #list(data.columns.values)  # list(range(p))
    random.shuffle(order)
    print("initial order: {}".format(order))
    scores = []

    node_scores = [0]*p
    for i in range(p):
        # possible parents to string
        subset_str = sc.list_to_score_key(order[:i] )
        
        subset_str = sc.list_to_score_key(list(set(order[:i]) & set(score_table["poss_cvars"][order[i]])))

        print("node: {} ".format(order[i]))
        print("subset: {}".format(subset_str))
        print(score_table["scores"][order[i]])
        node_scores[i] = score_table["scores"][order[i]][subset_str]
        #print("node score: {}".format(node_scores[i]))
        #rint("check: {}".format(check))

    score = np.sum(node_scores)
    print("initial score: {}".format(score))

    scores.append(score)
    order_trajectory.append(order)

    for i in tqdm(range(1, iterations+1), desc="Gibbs order sampler"):
        #print("\niteration: {}".format(i))
        # pick a random node
        node_index = np.random.randint(0, p)
        node = order_trajectory[i-1][node_index]
        #print("order: {}".format(order_trajectory[i-1]))
        #print("moving node {}".format(node))
        # calculate the neighborhood scores
        prop_probs = get_relocation_neighborhood(order_trajectory[i-1],
                                                 node_index,
                                                 scores[i-1],
                                                 node_scores,
                                                 score_table)

        #print("proposal probs: {}".format(prop_probs))
        #print("sum: {}".format(np.sum(prop_probs)))

        # Select at random from the proposal distribution
        new_pos = np.random.choice(list(range(len(prop_probs))), p=prop_probs)
        #print("to new pos: {}".format(new_pos))

        neworder = order_trajectory[i-1].copy()
        orderscore = move_node(node_index, new_pos,
                  neworder,
                  scores[i-1],
                  node_scores,
                  score_table)
        #print("order: {}".format(neworder))
        #print("score: {}".format(orderscore))
        order_trajectory.append(neworder)
        scores.append(orderscore)  # O(p)

    return order_trajectory, scores


def get_relocation_neighborhood(order, node_index, orderscore, node_scores,
                                score_table):
    # Move to the right
    # [1,2,3,i,4,5] => [1,2,3,4,i,5]

    neig_log_scores = [None] * len(order)
    neig_log_scores[node_index] = orderscore
    # print(order)
    for i in range(node_index, len(order)-1):
        #print("moving {} to the right".format(order[i]))

        orderscore = move_up(i, order, orderscore, node_scores, score_table)

        neig_log_scores[i+1] = orderscore
        # print(order)

    # Move all the way back. But dont relocate the nodes that have already been
    # relocated.
    for i in range(len(order)-1, 0, -1):
        #print("moving {} to the left".format(order[i]))
        orderscore = move_down(i, order, orderscore, node_scores, score_table)

        neig_log_scores[i-1] = orderscore
        # print(order)

    # move back to where we started
    for i in range(0, node_index):
        #print("moving {} to the right".format(order[i]))
        orderscore = move_up(i, order, orderscore, node_scores, score_table)

        neig_log_scores[i+1] = orderscore
        # print(order)

    #print("neig scores: {}".format(neig_log_scores))
    log_tot_neigh_scores = sc.logsumexp(neig_log_scores)
    #print("log tot neigh scores: {}".format(log_tot_neigh_scores))
    log_probs = neig_log_scores - log_tot_neigh_scores
    #print("log probs: {}".format(log_probs))

    prop_probs = np.exp(log_probs)
    #print("proposal probs: {}".format(prop_probs))
    return prop_probs


def find_optimal_cstree(data, max_cvars=1, alpha_tot=1, method="BDeu"):
    """ Find the optimal CStree for the data.

    Args:
        data (pandas DataFrame): The data as a pandas DataFrame.
        max_cvars (int, optional): Max context variables. Defaults to 1.
        alpha_tot (float, optional): The Dirichlet hyper parameter total pseudo counts. Defaults to 1.
        method (str, optional): Parameter prior type. Defaults to "BDeu".

    Examples:
        >>> import cstrees.learning as ctl
        >>> opttree = ctl.find_optimal_cstree(df, max_cvars=2, alpha_tot=1.0, method="BDeu")
        >>> opttree.to_df()
                a	b	c
        0	2	2	2
        1	*	-	-
        2	0	0	-
        3	1	0	-
        4	0	1	-
        5	1	1	-

    """
    score_table, context_scores, context_counts = sc.order_score_tables(data, 
                                                                    max_cvars=max_cvars, 
                                                                    alpha_tot=alpha_tot, 
                                                                    method=method)
    
    opt_order, _ = _find_optimal_order(score_table)

    opttree = _optimal_cstree_given_order(opt_order, context_scores,
                                          max_cvars=max_cvars)

    return opttree
