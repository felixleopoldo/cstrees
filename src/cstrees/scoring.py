import cstrees.cstree as ct
from scipy.special import loggamma

def score(t: ct.CStree, data):
    sc = 0
    for l in t.levels:
        sc += score_level(0, data)
    return score


def counts_at_level(t, l, data):
    """ Collect all the observed counts at a specific level by stages/contexts.
    """

    stage_counts = {}  # maybe context counts..
    for i in len(data):
        parents_vals = data[i, :l]
        stage = t.get_stage(parents_vals)  # or context

        if stage in t.stages:  # colored stage
            context = stage.context
        else:  # singleton stage
            context = data[i, :l]

        if context not in stage_counts:
            stage_counts[stage] = {} # only save the observed ones #[0] * t.cards[l]  # initiate with zeros.x
            stage_counts[stage][data[i, l]] = 1
        else:
            stage_counts[stage][data[i, l]] += 1
    return stage_counts

def score_stage():
    pass

def score_var():
    pass

def score_order(order, data):
    pass

def BDe_level(t, l, data, alpha_tot=1.0, method="BDeu"):
    
    level_counts = counts_at_level(t, l, data)
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
    
    
    for stage in level_counts:
        stage_counts = [c for c in level_counts[stage]]
        stage_counts_stage = sum(stage_counts)    
        
        score += loggamma(alpha_stage) - loggamma(alpha_stage + stage_counts_stage)
        for vals in t.cards[l]:
            score += loggamma(alpha_obs) - loggamma(alpha_obs + stage_counts[vals])
            
    return score

def BGe(t, data):
    score = 0 # log score
    for l in levels:
        score += level_score(l, data)
        
                