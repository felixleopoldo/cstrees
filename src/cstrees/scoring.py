import itertools
import cstrees.cstree as ct
from scipy.special import loggamma
import numpy as np


def counts_at_level(t, l, dataperm):
    """ Collect all the observed counts at a specific level by stages.
        So the counts for level l depends on the stage of level l-1.
    """
    stage_counts = {}  # maybe context counts..
    ord = list(t.co.order)
    dataperm = dataperm[:, ord] # reorder the columns according to the order

    #print("get counts at level {}".format(l))
    for i in range(len(dataperm)):  # iterate over the samples
        pred_vals = dataperm[i, :l]
        stage = t.get_stage(pred_vals)  # or context                
        #print("stages at level {}: {}".format(l-1, t.stages[l-1]))
        if stage == None:  # singleton stage. Shold note be any of these in our setting.
            print("singleton stage")
        if stage not in stage_counts:
            # only save the observed ones #[0] * t.cards[l]  # initiate with zeros.x
            stage_counts[stage] = {}
        if dataperm[i, l] in stage_counts[stage]:            
            stage_counts[stage][dataperm[i, l]] += 1
        else:
            stage_counts[stage][dataperm[i, l]] = 1
    return stage_counts

def score_stage():
    pass

def score_var():
    pass

def score_level(t, l, level_counts, alpha_tot=1.0, method="BDeu"):
    """BDe score at a level. We could also consider splitting the alpha into
    colored and non-colored stages.    


    Args:
        t (_type_): CStree
        l (int): Level.
        data (_type_): Data.
        alpha_tot (float, optional): _description_. Defaults to 1.0.
        method (str, optional): _description_. Defaults to "BDeu".

    Returns:
        _type_: _description_
    """

    # n_stages_at_level = t.stages[l] # including singleton
    # dont know if I shold divide by the total number of stages at the level or the observed ones
    # In BN I think it is the total number of parent settings, but there there are quite many
    # stages, counting the singleton ones, so the used alphas would be very small.
    # To make it consistent with BNs we might have to do so though.

    if method == "K2":
        assert (alpha_tot == 1)
        alpha_obs = alpha_tot
        alpha_stage = alpha_tot * t.cards[l]
    if method == "BD":  # This should be the Cooper-Herzkovits
        alpha_obs = alpha_tot
        alpha_stage = alpha_tot * t.cards[l]
    # elif method == "BDe": # not used in practice
    #    alpha_obs = alpha_tot / (t.n_stages_at_level(l) * t.cards[l]) # TODO: is this ok?
    #    alpha_stage = alpha_tot / t.n_stages_at_level(l) # TODO: is this ok?

    elif method == "BDeu":
        # TODO: assert that all stages are colored.
        n_stages = len(t.stages[l-1])#max(len(t.stages[l-1]), 1) # level 0 has no stages
        alpha_obs = alpha_tot / (n_stages * t.cards[l])
        alpha_stage = alpha_tot / n_stages

    score = 0  # log score
    for stage, counts in level_counts.items():
        stage_counts_stage = sum(counts.values())
        #print("stage counts: {}".format(stage_counts_stage))

        score += loggamma(alpha_stage) - loggamma(alpha_stage + stage_counts_stage)
        for val in range(t.cards[l]):
            if val not in counts:  # as we only store the observed values
                continue
            score += loggamma(alpha_obs + counts[val]) - loggamma(alpha_obs) 

    return score

def estimate_parameters(t, stage, stage_counts, method, alpha_tot):
    l = stage.level + 1 # estimating fot the level above the stage
    if method == "K2":
        assert (alpha_tot == 1)
        alpha_obs = alpha_tot
        alpha_stage = alpha_tot * t.cards[l]
    if method == "BD":  # This should be the Cooper-Herzkovits
        alpha_obs = alpha_tot
        alpha_stage = alpha_tot * t.cards[l]
    elif method == "BDeu":
        # TODO: assert that all stages are colored.
        n_stages = max(len(t.stages[l-1]), 1) # level 0 has no stages. it has [] actually...
        alpha_obs = alpha_tot / (n_stages * t.cards[l])
        alpha_stage = alpha_tot / n_stages

    probs = [None] * t.cards[l]
    
    stage_counts_total = sum(stage_counts[stage].values())
    for i in range(t.cards[l]):
        if i not in stage_counts[stage]: # no observations here so use only prior
            probs[i] = alpha_obs / alpha_stage
        else: # posterior mean or posterior predictive probabilites.
            probs[i] = (alpha_obs + stage_counts[stage][i]) / (alpha_stage + stage_counts_total)
    return probs

def score(t: ct.CStree, data: list, alpha_tot=1.0, method="BDeu"):
    """Score a CStree.

    Args:
        t (ct.CStree): CStree.
        data (list): _description_
        alpha_tot (float, optional): _description_. Defaults to 1.0.
        method (str, optional): _description_. Defaults to "BDeu".

    Returns:
        _type_: _description_
    """

    # reorder the columns in data according to t.order.
    #ord = list(t.co.order)
    #data_ordered = data[:, ord] # reorder the columns according to the order

    score = 0  # log score
    for l in range(t.p):
        print("level {} in the tree scoring procedure".format(l))
        level_counts = counts_at_level(t, l, data)

        for s, cnt in level_counts.items():
            print("{}: {}".format(s, cnt))

        score += score_level(t, l, level_counts, alpha_tot, method)
    return score

def score_order(order, data, strategy="max", max_cvars=1, alpha_tot=1.0, method="BDeu"):
    score = 0  # log score
    #print("")
    #print("scoring order: {}".format(order))
    for l in range(len(order)):
        s = score_order_at_level(order, l, data, strategy=strategy, max_cvars=max_cvars, alpha_tot=alpha_tot, method=method)
    #    print("level {} score: {}".format(l, s))
        score += s
    return score

def score_order_at_level(order, l, data, strategy="max", max_cvars=1, alpha_tot=1.0, method="BDeu"):
    """ Without singletons, there are 2*level stagings at level level.
        val1 side and val2 side colored in different ways
        val1 side and val2 side colored in the same way

        For max_cvars = 2 we have the combinations:
        1 == one stage, with 0 cvars
        0.5, 0.5 == two stages, with 1 cvar each
        0.5, 0.25, 0.25 == three stages, one with 1 cvar, two with 2 cvars 
        0.25, 0.25, 0.25, 0.25 == four stages, all with 2 cvars

    Args:
        order (_type_): _description_
        cards (_type_): _description_
        data (_type_): _description_
        max_cvars (int, optional): _description_. Defaults to 1.
        alpha_tot (float, optional): _description_. Defaults to 1.0.
        method (str, optional): _description_. Defaults to "BDeu".

    Returns:
        _type_: _description_
    """
    if max_cvars != 1:
        print("Only max_cvars = 1 implemented")
        return None

    p = len(order)
    cards = [2] * p
    co = ct.CausalOrder(order)
    tree = ct.CStree(co)
    tree.set_cardinalities(cards)
    stagings = ct.all_stagings(p, cards, l-1, max_cvars=max_cvars)
    
    if strategy == "max":
        score = -np.inf  # log score
    elif strategy == "sum":
        score = 0
        
    # This is a bit clumsy but shoud work.    
    for stlist in stagings:
        tree.set_stages({l-1: stlist}) # Need to set the stagings in order to count.
        level_counts = counts_at_level(tree, l, data)
        tmp = score_level(tree, l, level_counts, alpha_tot, method)
        print("level {} score: {}".format(l, tmp))   

        if strategy == "max":
            if tmp > score:
                score = tmp
        if strategy == "sum":
            score += tmp
    
    return score

    # With max_cvars = 1, for each each level can have 4 stagings, including the singletons:
    # val1 side colored rest uncolored
    # val2 side colored rest uncolored
    # both colored
    # none colored
    # so there are 4*level stagings at level level.
        # TODO: The paret below allows for singletons, but we dont want that.
        # so it is commented out.

        #     if l == 0:
        # # There are only 2 stagings at level 0
        # Whithot singletons, this is ok.
        # # Only singletons, i.e. two stages.
        # tree.set_stages({})
        # score += score_level(tree, l, level_counts, alpha_tot, method)
        # print(score)

        # # Just one stage
        # st = ct.Stage([set(vals[l])])
        # print(st)
        # tree.set_stages({l: [st]})
        # score += score_level(tree, l, level_counts, alpha_tot, method)
        # print(score)

        # loop through all possible combinations.
        # use iter tools to go trough posssible values and the mask in cards.
        # for k  in range(l+1): # all variables up to l can be context variables

        #     prodset = [[True, False]] * cards[k]
        #     print("cvar level: {}".format(k))
        #     for binarray in itertools.product(*prodset): # each represents a staging
        #         stlist = []
        #         cvars = [vals[k][ind] for ind, val in enumerate(binarray) if val]
        #         print("value to use as contexts: {}".format(cvars))
        #         if cvars == []: #only singletons case treated separately. We dont allow for singletons.
        #             #print("only singletons -> skip")
        #             #continue
        #             tree.set_stages({})
        #             score += score_level(tree, l, level_counts, alpha_tot, method)
        #         else:
        #             for v in cvars:

        #                 left = [set(vals[i]) for i in range(k)]
        #                 right = [set(vals[j]) for j in range(k+1,l+1)]
        #                 stagelistrep = left + [v] +  right # For example: [[0,1], [0,1], 0, [0, 1]]
        #                 st = ct.Stage(stagelistrep)

        #                 stlist += [st]
        #             print("Staging")
        #             for st in stlist:
        #                 print("stage: {}".format(st))

        #             tree.set_stages({l: stlist}) # This is a bit clumsy but shoud work.

        #             score += score_level(tree, l, level_counts, alpha_tot, method)
        #         print(score)