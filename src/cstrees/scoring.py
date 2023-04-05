import cstrees.cstree as ct
from scipy.special import loggamma

def score(t: ct.CStree, data):
    sc = 0
    for l in t.levels:
        sc += score_level(0, data)
    return score


def counts_at_level(t, l, data):
    """ Collect all the observed counts at a specific level by stages.
    """
    stage_counts = {}  # maybe context counts..
    for i in range(len(data)): # iterate over the samples
        parents_vals = data[i, :l]
        #print("possible context values {}".format(parents_vals))
        stage = t.get_stage(parents_vals)  # or context
        if stage == None: # singleton stage
            stage = ct.Stage(list(data[i, :l])) # Singleton stage
        if stage not in stage_counts:
            stage_counts[stage] = {} # only save the observed ones #[0] * t.cards[l]  # initiate with zeros.x
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
    

    #n_stages_at_level = t.stages[l] # including singleton
    # dont know if I shold divide by the total number of stages at the level or the observed ones
    # In BN I think it is the total number of parent settings, but there there are quite many
    # stages, counting the singleton ones, so the used alphas would be very small.
    # To make it consistent with BNs we might have to do so though.

    

    if method == "K2":
        assert(alpha_tot == 1)
        alpha_obs = alpha_tot
        alpha_stage = alpha_tot * t.cards[l] 
    if method == "BD": # This should be the Cooper-Herzkovits
        alpha_obs = alpha_tot
        alpha_stage = alpha_tot * t.cards[l]
    #elif method == "BDe": # not used in practice
    #    alpha_obs = alpha_tot / (t.n_stages_at_level(l) * t.cards[l]) # TODO: is this ok?
    #    alpha_stage = alpha_tot / t.n_stages_at_level(l) # TODO: is this ok?
        
    elif method == "BDeu":
        #assert(alpha_tot==1)
        #alpha_obs = alpha_tot / (len(level_counts) * t.cards[l]) # TODO: is this ok?
        #alpha_stage = alpha_tot / len(level_counts) # TODO: is this ok?
        alpha_obs = alpha_tot / (t.n_stages_at_level(l) * t.cards[l]) # TODO: is this ok?
        alpha_stage = alpha_tot / t.n_stages_at_level(l) # TODO: is this ok?
    
    score = 0 # log score
    for stage, counts in level_counts.items():
        stage_counts_stage = sum(counts.values())
        print("stage counts: {}".format(stage_counts_stage))

        score += loggamma(alpha_stage) - loggamma(alpha_stage + stage_counts_stage)
        for val in range(t.cards[l]):
            if val not in counts: # as we only store the observed values
                continue
            score += loggamma(alpha_obs) - loggamma(alpha_obs + counts[val])

    print(score)
    return score

def score(t, data, alpha_tot=1.0, method="BDeu"):
    score = 0 # log score
    #for l in range(1):
    for l in range(t.p):
        level_counts = counts_at_level(t, l, data)
        score += score_level(t, l, level_counts, alpha_tot, method)
        
    return score


def score_order(order, data, max_cvars=1, alpha_tot=1.0, method="BDeu"):
    score = 0 # log score
    
    # With max_cvars = 1, for each each level can have 4 stagings:
    # val1 side colored rest uncolores
    # val2 side colored rest uncolores
    # both colored
    # none colored
    # so there are 4*level stagings at level level.
    
     
    for l in range(len(order)):
        # sum the scores for all differents stagings at level l
        pass
        
        
    return score