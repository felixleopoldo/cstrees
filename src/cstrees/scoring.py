import itertools
import cstrees.cstree as ct
from scipy.special import loggamma
import numpy as np


def counts_at_level(t, l, data):
    """ Collect all the observed counts at a specific level by stages.
        So the counts for level l depends on the stage of level l-1.
    """
    stage_counts = {}  # maybe context counts..
    for i in range(len(data)):  # iterate over the samples
        pred_vals = data[i, :l]
        #print("possible context values {}".format(parents_vals))
        stage = t.get_stage(pred_vals)  # or context
        if stage == None:  # singleton stage
            stage = ct.Stage(list(data[i, :l]))  # Singleton stage
        if stage not in stage_counts:
            # only save the observed ones #[0] * t.cards[l]  # initiate with zeros.x
            stage_counts[stage] = {}
        #print("value : {}".format(data[i,l]))
        if data[i, l] in stage_counts[stage]:
            stage_counts[stage][data[i, l]] += 1
        else:
            stage_counts[stage][data[i, l]] = 1
    return stage_counts


def score_stage():
    pass


def score_var():
    pass


def score_order(order, data):
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
        # assert(alpha_tot==1)
        # alpha_obs = alpha_tot / (len(level_counts) * t.cards[l]) # TODO: is this ok?
        # alpha_stage = alpha_tot / len(level_counts) # TODO: is this ok?
        alpha_obs = alpha_tot / (t.n_stages_at_level(l)
                                 * t.cards[l])  # TODO: is this ok?
        alpha_stage = alpha_tot / t.n_stages_at_level(l)  # TODO: is this ok?

    score = 0  # log score
    for stage, counts in level_counts.items():
        stage_counts_stage = sum(counts.values())
        #print("stage counts: {}".format(stage_counts_stage))

        score += loggamma(alpha_stage) - \
            loggamma(alpha_stage + stage_counts_stage)
        for val in range(t.cards[l]):
            if val not in counts:  # as we only store the observed values
                continue
            score += loggamma(alpha_obs) - loggamma(alpha_obs + counts[val])

    return score


def score(t: ct.CStree, data: list, alpha_tot=1.0, method="BDeu"):
    # reorder the columns in data according to t.order.
 
 
    ord = list(t.co.order)
    print(ord)
    data_ordered = data[:, ord]
    print(data_ordered)
    
    score = 0  # log score
    for l in range(t.p):
        print("level {}".format(l))
        level_counts = counts_at_level(t, l, data)

        for s, cnt in level_counts.items():
            print("{}: {}".format(s, cnt))

        score += score_level(t, l, level_counts, alpha_tot, method)
    return score


def score_order(order, cards, data, max_cvars=1, alpha_tot=1.0, method="BDeu"):
    score = 0  # log score

    # With max_cvars = 1, for each each level can have 4 stagings:
    # val1 side colored rest uncolored
    # val2 side colored rest uncolored
    # both colored
    # none colored
    # so there are 4*level stagings at level level.

    if max_cvars != 1:
        return None
        
    p = len(order)
    co = ct.CausalOrder(range(p))
    tree = ct.CStree(co)
    tree.set_cardinalities(cards)

    vals = [list(range(cards[l])) for l in range(p)]

    for l in range(len(order)):
        if l == 0:
            print("level 0 is special I think so skip.")
            continue
        print("level: {}".format(l))
        # Generate the possible stagings and att the score.
        
        # loop through all possible combinations.
        # use iter tools to go trough posssible values and the mask in cards.
        level_counts = counts_at_level(tree, l, data)
        for k  in range(l+1): # all variables up to l can be context variables
            
            prodset = [[True, False]] * cards[k]
            print("cvar level: {}".format(k))
            for binarray in itertools.product(*prodset): # each represents a staging
                stlist = []
                cvars = [vals[k][ind] for ind, val in enumerate(binarray) if val]
                print("value to use as contexts: {}".format(cvars)) 
                #if cvars == []:
                    
                for v in cvars:
                    
                    left = [set(vals[i]) for i in range(k)]
                    right = [set(vals[j]) for j in range(k+1,l+1)]
                    stagelistrep = left + [v] +  right
                    st = ct.Stage(stagelistrep)
                    
                    stlist += [st]
                print("Staging")
                for st in stlist:
                    print("stage: {}".format(st))

                
                tree.set_stages({l: stlist}) # This is a bit clumsy but shoud work. 
                
                
                score += score_level(tree, l, level_counts, alpha_tot, method)
                print(score)
        
    return score
