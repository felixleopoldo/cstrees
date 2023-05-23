from cstrees.cstree import *
from cstrees.csi_relation import *
from cstrees.stage import *
from cstrees.scoring import *


def all_stagings(p, cards, l, max_cvars=1):
    """ Generates an iterator over all stagings of a given level.
        
    Args:
        p (int): Number of variables.
        cards (list): List of cardinalities of the variables.
        l (int): The level of the stage.
        max_cvars (int, optional): The maximum number of context variables . Defaults to 1.

    Raises:
        NotImplementedError: Exception if max_cvars > 1.

    Yields:
        _type_: Iterator over all stagings of a given level.

    Examples:
        A staging with 2 stages for a binary CStree at level 2 
        (numbering levels from 0) could e.g. be: 
        
        >>> staging = [Stage([{0, 1}, 0, {0, 1}]), Stage([{0, 1}, 1, {0, 1}])]

    """
    assert(l < len(cards)-1)
    assert(l < p-1)
    if max_cvars == 1:
        if l == -1:  # This is an imaginary level -1, it has no stages.
            yield [Stage([])]
            return

        # All possible values for each variable
        
        vals = [list(range(cards[l])) for l in range(p)]
        for k in range(l+1):  # all variables up to l can be context variables
            # When we restrict to max_cvars = 1, we have two cases:
            # Either all are in 1 color or all are in 2 different colors.
            stlist = []  # The staging: list of Stages.
            # Loop through the values of the context variables.
            for v in vals[k]:
                left = [set(vals[i]) for i in range(k)]
                right = [set(vals[j]) for j in range(k+1, l+1)]
                # For example: [{0,1}, {0,1}, 0, {0, 1}]
                stagelistrep = left + [v] + right
                st = Stage(stagelistrep)
                stlist += [st]
            yield stlist

        # The staging with no context variables
        stagelistrep = [set(v) for v in vals][:l+1]

        st = Stage(stagelistrep)
        yield [st]
    elif max_cvars == 2:
        from cstrees.double_cvar_stagings import enumerate_stagings
        

        for staging_list in enumerate_stagings(l+1):
            
            staging = []
            for stage_list in staging_list:
                # Fix repr bug
                if isinstance(stage_list, set):
                    stage_list = [stage_list]
                
                stage = Stage(stage_list) #could set colors here cut that takes time maybe.
                staging.append(stage)
            yield staging
                
    else:
        raise NotImplementedError("max_cvars > 1 not implemented yet")


def n_stagings(p, cards, l, max_cvars=1):
    """ Returns the number of stagings at a given level.
    
    Args:
        p (int): Number of variables.            
        cards (list): List of cardinalities of the variables.
        l (int): The level of the stage.
        cvars (int, optional): The maximum number of context variables . Defaults to 1.

    """
    
    stagings = all_stagings(p, cards, l, max_cvars)
    for staging in stagings:
        print([str(s) for s in staging])
    
    return sum(len(staging) for staging in stagings)


def optimal_staging_at_level(order, data, l, max_cvars=1, alpha_tot=None, method="BDeu"):
    p = len(order)
    cards = data.iloc[0].values
    assert(l < len(cards)-1)
    tree = CStree(cards)
    tree.labels = order
    
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
    stages[-1] = [Stage([], color="black")]
    for level in range(-1, p-1):  # dont stage the last level
        max_staging, max_staging_score = optimal_staging_at_level(
            order, cards, data, level, max_cvars, alpha_tot, method)
        stages[level] = max_staging
        print("max staging: {}".format([str(s) for s in max_staging]))

    # Create CStree
    tree = CStree(cards)
    tree.labels = order
    
    # color each stage. Singletons are black. This should be done somewhere else probably.
    colors = ['blueviolet', 'orange', 'navy', 'rebeccapurple', 'darkseagreen', 'darkslategray', 'lightslategray', 'aquamarine', 'lightgoldenrodyellow', 'cornsilk', 'azure', 'chocolate', 'red', 'darkolivegreen', 'chartreuse', 'turquoise', 'olive', 'crimson', 'goldenrod', 'orchid', 'firebrick', 'lawngreen', 'deeppink', 'wheat', 'teal', 'mediumseagreen', 'peru', 'salmon', 'palegreen', 'navajowhite', 'yellowgreen', 'mediumaquamarine', 'darkcyan', 'dodgerblue', 'brown', 'powderblue', 'mistyrose', 'violet', 'darkslategrey', 'midnightblue', 'aliceblue', 'dimgrey', 'palegoldenrod', 'black', 'darkgrey', 'olivedrab', 'linen', 'lightblue', 'thistle', 'greenyellow', 'indianred', 'khaki', 'lightslategrey', 'slateblue', 'purple', 'deepskyblue', 'magenta', 'yellow', 'ivory', 'darkorchid', 'mediumpurple', 'snow', 'dimgray', 'palevioletred', 'darkslateblue', 'sandybrown', 'lightgray', 'lemonchiffon', 'gray', 'silver', 'aqua', 'tomato', 'lightyellow', 'seagreen', 'darkmagenta', 'beige', 'cornflowerblue', 'peachpuff', 'ghostwhite', 'cyan', 'lightcoral', 'hotpink', 'lightpink', 'lightskyblue', 'slategrey', 'tan', 'oldlace', 'steelblue', 'springgreen', 'fuchsia', 'lime', 'papayawhip', 'mediumblue', 'mediumspringgreen', 'darkorange', 'lightgreen', 'blue', 'slategray', 'white', 'saddlebrown', 'mediumturquoise', 'paleturquoise', 'darkblue', 'plum', 'lightseagreen', 'lightgrey', 'blanchedalmond', 'lavenderblush', 'darkkhaki', 'gainsboro', 'lightsalmon', 'darkturquoise', 'moccasin', 'darkgoldenrod', 'mediumorchid', 'honeydew', 'mediumslateblue', 'maroon', 'forestgreen', 'darkgray', 'floralwhite', 'darkgreen', 'lightcyan', 'darksalmon', 'pink', 'royalblue', 'sienna', 'green', 'orangered', 'bisque', 'antiquewhite', 'rosybrown', 'whitesmoke', 'darkred', 'burlywood', 'skyblue', 'mediumvioletred', 'mintcream', 'limegreen', 'lightsteelblue', 'grey', 'coral', 'indigo', 'gold', 'cadetblue']
    for level, staging in stages.items():
        for i, stage in enumerate(staging):
            if all([isinstance(i, int) for i in stage.list_repr]):
                stage.color = "black"
            else:
                stage.color = colors[level]
    
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
