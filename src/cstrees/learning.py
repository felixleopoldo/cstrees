import numpy as np

import cstrees.cstree as ct
from cstrees import csi_relation
import cstrees.stage as stl
import cstrees.scoring as sc
from itertools import permutations


def all_stagings(cards, l, max_cvars=1):
    """ Returns a generator over all stagings of a given level.

    Args:
        l (int): The level of the stage.
        cards (list): List of cardinalities of the variables. Should be at least of length l+1.
        max_cvars (int, optional): The maximum number of context variables . Defaults to 1.

    Raises:
        NotImplementedError: Exception if max_cvars > 2.

    Yields:
        generator: generator over all stagings of a given level.

    Examples:
        A staging with 2 stages for a binary CStree at level 2
        (numbering levels from 0) could e.g. be:

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

    assert(l < len(cards)-1)
    if max_cvars == 1:
        if l == -1:  # This is an imaginary level -1, it has no stages.
            yield [stl.Stage([])]
            return

        # All possible values for each variable

        vals = [list(range(cards[l])) for l in range(len(cards))]
        for k in range(l+1):  # all variables up to l can be context variables
            # When we restrict to max_cvars = 1, we have two cases:
            # Either all are in 1 color or all are in 2 different colors.
            stlist = []  # The staging: list of stl.Stages.
            # Loop through the values of the context variables.
            for v in vals[k]:
                left = [set(vals[i]) for i in range(k)]
                right = [set(vals[j]) for j in range(k+1, l+1)]
                # For example: [{0,1}, {0,1}, 0, {0, 1}]
                stagelistrep = left + [v] + right
                st = stl.Stage(stagelistrep)
                stlist += [st]
            yield stlist

        # The staging with no context variables
        stagelistrep = [set(v) for v in vals][:l+1]

        st = stl.Stage(stagelistrep)
        yield [st]
    elif max_cvars == 2:
        from cstrees.double_cvar_stagings import enumerate_stagings


        for staging_list in enumerate_stagings(l+1):

            staging = []
            for stage_list in staging_list:
                # Fix repr bug
                if isinstance(stage_list, set):
                    stage_list = [stage_list]

                stage = stl.Stage(stage_list) #could set colors here cut that takes time maybe.
                staging.append(stage)
            yield staging

    else:
        raise NotImplementedError("max_cvars > 1 not implemented yet")


def n_stagings(cards, level, max_cvars=1):
    """ Returns the number of stagings at a given level.

    Args:
        p (int): Number of variables.
        cards (list): List of cardinalities of the variables.
        level (int): The level in the CStree.
        cvars (int, optional): The maximum number of context variables. Defaults to 1.
    
    Examples:
        >>> cards = [2]*4
        >>> ctl.n_stagings(cards, 2, max_cvars=2)
        8

    """

    stagings = all_stagings(cards, level, max_cvars)

    return sum(1 for _ in stagings)


def _optimal_staging_at_level(order, data, level, max_cvars=1, alpha_tot=None, 
                              method="BDeu"):
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

    cards = data.iloc[0].values
    assert(level < len(cards)-1)
    tree = ct.CStree(cards)
    tree.labels = order

    stagings = all_stagings(cards, level, max_cvars)
    max_staging = None
    max_staging_score = -np.inf

    for stlist in stagings:

        tree.update_stages({level: stlist})
        # This needs the stages to be set at the line above.
        level_counts = sc.counts_at_level(tree, level+1, data)
        score = sc.score_level(tree, level+1, level_counts, alpha_tot, method)

        if score > max_staging_score:
            max_staging_score = score
            max_staging = stlist

    return max_staging, max_staging_score


def _optimal_cstree_given_order(order, data, max_cvars=1, alpha_tot=1.0, 
                                method="BDeu"):
    """Find the optimal CStree for a given order.

    Args:
        order (list): The order of the variables.
        data (pandas DataFrame): The data as a pandas DataFrame.
        max_cvars (int, optional): Max context variables. Defaults to 1.
        alpha_tot (float, optional): The Dirichlet hyper parameter total pseudo 
        counts. Defaults to 1.0.
        method (str, optional): Parameter prior type. Defaults to "BDeu".
    Examples:


    """

    # BUG?: Maybe these have to be adapted to the order.
    cards = data.iloc[0].values
    p = len(order)

    stages = {}
    stages[-1] = [stl.Stage([], color="black")]
    for level in range(-1, p-1):  # dont stage the last level
        max_staging, max_staging_score = _optimal_staging_at_level(
            order, data, level, max_cvars, alpha_tot, method)
        stages[level] = max_staging
        #print("max staging: {}".format([str(s) for s in max_staging]))

    # Create CStree
    tree = ct.CStree(cards)
    tree.labels = order

    # Color each stage in the optimal staging. Singletons are black.
    # This should be done somewhere else probably.
    colors = ['blueviolet', 'orange', 'navy', 'rebeccapurple', 'darkseagreen',
              'darkslategray', 'lightslategray', 'aquamarine',
              'lightgoldenrodyellow', 'cornsilk', 'azure', 'chocolate',
              'red', 'darkolivegreen', 'chartreuse', 'turquoise', 'olive',
              'crimson', 'goldenrod', 'orchid', 'firebrick', 'lawngreen', 
              'deeppink', 'wheat', 'teal', 'mediumseagreen', 'peru', 'salmon', 
              'palegreen', 'navajowhite', 'yellowgreen', 'mediumaquamarine', 
              'darkcyan', 'dodgerblue', 'brown', 'powderblue', 'mistyrose', 
              'violet', 'darkslategrey', 'midnightblue', 'aliceblue', 
              'dimgrey', 'palegoldenrod', 'black', 'darkgrey', 'olivedrab', 
              'linen',  'lightblue', 'thistle', 'greenyellow', 'indianred', 
              'khaki']
              
    for level, staging in stages.items():
        for i, stage in enumerate(staging):
            if all([isinstance(i, int) for i in stage.list_repr]):
                stage.color = "black"
            else:
                stage.color = colors[i]
    tree.update_stages(stages)

    return tree


def _find_optimal_order(data, strategy="max", max_cvars=1, alpha_tot=1, 
                        method="BDeu"):
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
        >>> optord, score = ctl.find_optimal_order(
        >>> df, strategy="max", max_cvars=2, alpha_tot=1.0, method="BDeu")
        >>> print("optimal order: {}, score {}".format(optord, score))

    """

    p = data.shape[1]
    perms = permutations(list(range(p)))
    labels = data.columns.values
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
                               max_cvars=max_cvars, alpha_tot=alpha_tot, 
                               method=method)
        #print("order: {}, score: {}".format(order, score))

        # if score == max_score:
        #    optimal_orders.append(order)
        if score > max_score:
            max_score = score
            optimal_orders = order

    return optimal_orders, max_score


def find_optimal_cstree(data, max_cvars=1, alpha_tot=1, method="BDeu"):
    """ Find the optimal CStree for the data.

    Args:
        data (pandas DataFrame): The data as a pandas DataFrame.
        max_cvars (int, optional): Max context variables. Defaults to 1.
        alpha_tot (float, optional): The Dirichlet hyper parameter total pseudo counts. Defaults to 1.
        method (str, optional): Parameter prior type. Defaults to "BDeu".
    
    Examples:
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
    opt_order, _ = _find_optimal_order(
        data, max_cvars=max_cvars, alpha_tot=alpha_tot, method=method)

    opttree = _optimal_cstree_given_order(opt_order, data,
                                          max_cvars=max_cvars,
                                          alpha_tot=alpha_tot, method=method)

    return opttree
