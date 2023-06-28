from scipy.special import loggamma
from itertools import combinations, product
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import pp

import cstrees.learning as learn
import cstrees.cstree as ct
import cstrees.stage as st
import cstrees.csi_relation as csi_rel


def counts_at_level(cstree: ct.CStree, level: int, data):
    """ Collect all the observed counts at a specific level by stages.
    So the counts for level l depends on the stage of level l-1.
    (we probably have to ase these on context instead of stage)
    Args:
        cstree (ct.CStree): A CStree
        level (int): The level to get counts for.
        data (pd.DataFrame): The data.

    Example:
        >>> import random
        >>> import numpy as np
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.to_df()
                0	1	2	3
        0	2	2	2	2
        1	*	-	-	-
        2	*	1	-	-
        3	*	0	-	-
        4	0	*	*	-
        5	1	*	*	-
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> counts = sc.counts_at_level(tree, 2, df)
        >>> for key, val in counts.items():
        >>>    print("Stage: {}".format(key))
        >>>    print("Counts: {}".format(val))
        Stage: [{0, 1}, 1]; probs: [0.24134031 0.75865969]; color: blueviolet
        Counts: {1: 756, 0: 241}
        Stage: [{0, 1}, 0]; probs: [0.58026003 0.41973997]; color: orange
        Counts: {0: 3}
    """
    stage_counts = {} # TODO: Maybe it should be context counts instead!

    # reorder the columns according to the order.
    # cardinalities are at first row.

    dataperm = data[cstree.labels].values[1:, :]

    #print("get counts at level {}".format(l))
    for i in range(len(dataperm)):  # iterate over the samples
        pred_vals = dataperm[i, :level]
        stage = cstree.get_stage(pred_vals)  # or context
        #print('pred_vals: ', pred_vals)
        #print("stages at level {}: {}".format(l-1, t.stages[l-1]))
        if stage is None:  # singleton stage. Shold note be any of these in our setting.
            print("singleton stage")
        if stage not in stage_counts:
            # only save the observed ones #[0] * t.cards[l]  # initiate with zeros.x
            stage_counts[stage] = {}
        if dataperm[i, level] in stage_counts[stage]:
            stage_counts[stage][dataperm[i, level]] += 1
        else:
            stage_counts[stage][dataperm[i, level]] = 1
    return stage_counts

def context_counts(data):
    """ Collect all the observed counts at a specific level by stages.
    So the counts for level l depends on the stage of level l-1.
    (we probably have to ase these on context instead of stage)
    Args:
        cstree (ct.CStree): A CStree
        level (int): The level to get counts for.
        data (pd.DataFrame): The data.

    Example:
        >>> import random
        >>> import numpy as np
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.to_df()
                0	1	2	3
        0	2	2	2	2
        1	*	-	-	-
        2	*	1	-	-
        3	*	0	-	-
        4	0	*	*	-
        5	1	*	*	-
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> counts = sc.counts_at_level(tree, 2, df)
        >>> for key, val in counts.items():
        >>>    print("Stage: {}".format(key))
        >>>    print("Counts: {}".format(val))
        Stage: [{0, 1}, 1]; probs: [0.24134031 0.75865969]; color: blueviolet
        Counts: {1: 756, 0: 241}
        Stage: [{0, 1}, 0]; probs: [0.58026003 0.41973997]; color: orange
        Counts: {0: 3}
    """
    context_counts = {lab: {} for lab in data.labels} # TODO: Maybe it should be context counts instead!

    # reorder the columns according to the order.
    # cardinalities are at first row.

    #print("get counts at level {}".format(l))
    for i in data.labels:  # iterate over the samples



        pred_vals = dataperm[i, :level]
        stage = cstree.get_stage(pred_vals)  # or context
        #print('pred_vals: ', pred_vals)
        #print("stages at level {}: {}".format(l-1, t.stages[l-1]))
        if stage is None:  # singleton stage. Shold note be any of these in our setting.
            print("singleton stage")
        if stage not in stage_counts:
            # only save the observed ones #[0] * t.cards[l]  # initiate with zeros.x
            stage_counts[stage] = {}
        if dataperm[i, level] in stage_counts[stage]:
            stage_counts[stage][dataperm[i, level]] += 1
        else:
            stage_counts[stage][dataperm[i, level]] = 1
    return stage_counts


def score_level(cstree, level, level_counts, alpha_tot=1.0, method="BDeu"):
    """ CS-BDeu score at a level folowing
    Conor Hughes, Peter Strong, Aditi Shenvi. Score Equivalence for Staged Trees.

    Args:
        cstree (CStree): CStree
        level (int): Level.
        data (pandas DataFrame): Data.
        alpha_tot (float, optional): Hyper parameter for the stage parameters Dirichlet prior distribution. Defaults to 1.0.
        method (str, optional): Parameter estimator. Defaults to "BDeu".

    Reference:
        https://arxiv.org/abs/2206.15322

    Example:
        >>> import random
        >>> import numpy as np
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> import cstrees.scoring as sc
        >>> sc.score_level(tree, 2, counts, alpha_tot=1.0, method="BDeu")
        -556.4949720501456

    Returns:
        float: The level score.
    """

    staging_score = 0  # log score
    for stage, counts in level_counts.items():
        #print("\nstage: {}".format(stage))
        if method == "K2":
            assert (alpha_tot == 1)
            alpha_obs = alpha_tot
            alpha_stage = alpha_tot * cstree.cards[level]
        if method == "BD":  # This should be the Cooper-Herzkovits
            alpha_obs = alpha_tot
            alpha_stage = alpha_tot * cstree.cards[level]
        elif method == "BDeu":
            alpha_stage = alpha_tot * cstree.stage_proportion(stage)
            alpha_obs = alpha_stage / cstree.cards[level]

        stage_counts = sum(counts.values())
        #print("stage counts: {}".format(stage_counts))
        #print("stage alpha: {}".format(alpha_stage))
        
        # Note that the score is depending on the context in the stage. So
        # note really th satge as such.
        score = loggamma(alpha_stage) - loggamma(alpha_stage + stage_counts)
        #print("stage all values score: {}".format(score))
        for val in range(cstree.cards[level]):
            if val not in counts:  # as we only store the observed values
                continue
            
            #print("value score {}:".format(loggamma(alpha_obs + counts[val]) - loggamma(alpha_obs)))
            score += loggamma(alpha_obs + counts[val]) - loggamma(alpha_obs)
        staging_score += score
    return staging_score

def score_context(var, context, context_vars, cards, counts, alpha_tot=1.0, method="BDeu"):
    """ CS-BDeu score at a level folowing
    Conor Hughes, Peter Strong, Aditi Shenvi. Score Equivalence for Staged Trees.

    Args:
        cstree (CStree): CStree
        level (int): Level.
        data (pandas DataFrame): Data.
        alpha_tot (float, optional): Hyper parameter for the stage parameters Dirichlet prior distribution. Defaults to 1.0.
        method (str, optional): Parameter estimator. Defaults to "BDeu".

    Reference:
        https://arxiv.org/abs/2206.15322

    Example:
        >>> import random
        >>> import numpy as np
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> import cstrees.scoring as sc
        >>> sc.score_level(tree, 2, counts, alpha_tot=1.0, method="BDeu")
        -556.4949720501456

    Returns:
        float: The level score.
    """

    if method == "K2":
        assert (alpha_tot == 1)
        alpha_obs = alpha_tot
        alpha_context = alpha_tot * cards[var]
    if method == "BD":  # This should be the Cooper-Herzkovits
        alpha_obs = alpha_tot
        alpha_context = alpha_tot * cards[var]
    elif method == "BDeu":
        context_prop = 1 / np.prod([cards[c] for c in context_vars])
        alpha_context = alpha_tot * context_prop
        alpha_obs = alpha_context / cards[var]
    #print("var: {}".format(var))
    #print("context: {}".format(context))
    #print(counts[var][context]["counts"])
    context_counts = sum(counts[var][context]["counts"].values())

    #print("stage counts: {}".format(context_counts))
    #print("stage alpha: {}".format(alpha_context))
    #print("context vars: {}".format(context_vars))

    # Note that the score is depending on the context in the stage. So
    # note really th satge as such.
    score = loggamma(alpha_context) - loggamma(alpha_context + context_counts)
    #print("all values score: {}".format(score))
    for val, count in counts[var][context]["counts"].items():
        #print(val, count) # value does not matter
        #print("value score: {}".format(loggamma(alpha_obs + count) - loggamma(alpha_obs)))
        score += loggamma(alpha_obs + count) - loggamma(alpha_obs)
    #counts[var][context]["score"] = score # just temporary

    #print("total context score: {}".format(score))
    # for val in range(cards[var]):
    #     if val not in counts:  # as we only store the observed values
    #         continue
    #     score += loggamma(alpha_obs + counts[val]) - loggamma(alpha_obs)

    return score


def estimate_parameters(cstree: ct.CStree, stage, stage_counts, method="BDeu", alpha_tot=1.0):
    """Estimate the parameters for a stage.

    Args:
        cstree (ct.CStree): A CStree.
        stage (Stage): A stage.
        stage_counts (dict): Counts for all the stage (sufficient statistics).
        method (string): Estimation method. Defaults to "BDeu".
        alpha_tot (float): Hyper parameter for the stage parameters Dirichlet prior distribution. Defaults to 1.0.

    Returns:
        list: List of probabilities associated with the stage. I.e conditional probabilities for the variable one level up.

    Example:
        >>> import random
        >>> import numpy as np
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> stage = tree.stages[1][0]
        >>> print(stage)
        >>> param_est = sc.estimate_parameters(tree, stage, counts, alpha_tot=1.0, method="BDeu")
        >>> print(param_est)
        [{0, 1}, 1]; probs: [0.24134031 0.75865969]; color: blueviolet
        [0.24185463659147868, 0.7581453634085213]
    """
    level = stage.level + 1  # estimating fot the level above the stage
    if method == "K2":
        assert alpha_tot == 1
        alpha_obs = alpha_tot
        alpha_stage = alpha_tot * cstree.cards[level]
    if method == "BD":  # This should be the Cooper-Herzkovits
        alpha_obs = alpha_tot
        alpha_stage = alpha_tot * cstree.cards[level]
    elif method == "BDeu":
        # TODO: assert that all stages are colored.
        # level 0 has no stages. it has [] actually...
        n_stages = max(len(cstree.stages[level-1]), 1)
        alpha_obs = alpha_tot / (n_stages * cstree.cards[level])
        alpha_stage = alpha_tot / n_stages

    probs = [None] * cstree.cards[level]

    stage_counts_total = sum(stage_counts[stage].values())
    for i in range(cstree.cards[level]):
        # no observations here so use only prior
        if i not in stage_counts[stage]:
            probs[i] = alpha_obs / alpha_stage
        else:  # posterior mean or posterior predictive probabilites.
            probs[i] = (alpha_obs + stage_counts[stage][i]) / \
                (alpha_stage + stage_counts_total)
    return probs


def cstree_posterior(cstee: ct.CStree, data: pd.DataFrame, alpha_tot=1.0, method="BDeu"):
    pass

def score_tables(data: pd.DataFrame,
                 strategy="posterior", max_cvars=2,
                 alpha_tot=1.0, method="BDeu"):


    scores = {lab : {} for lab in data.columns}
    counts = {lab : {} for lab in data.columns}
    #counts = context_counts(data)
    # TODO: checkk its the correct cards
    #print(" cards: {}".format(data.loc[0, :].values))

    cards_dict = {var: data.loc[0, var] for var in data.columns }
    #print("cards: {}".format(cards))
    # go through all variables
    for var in data.columns:
        #print("\nvariable: {}".format(var))
        # Iterate through all context sizes
        for csize in range(max_cvars+1):
            # Iterate through all possible contexts
            # remove the current variable from the active labels
            labels = [l for l in data.columns if l != var]
            #print("context size: {}".format(csize))
            for context_variables in combinations(labels, csize):
                #print("context variables: {}".format(context_variables))
                # get the active labels like A,B,C
                active_labels = [l for l in labels if l in context_variables]
                tmp = {c: None for c in active_labels}

                #print("active labels: {}".format(active_labels))
                if len(active_labels) == 0:
                    test = data[1:][var].value_counts()
                else:
                    test = data[1:].groupby(active_labels)[var].value_counts()
                #print(test)

                testdf = test.to_frame().rename(columns={var: var+" counts"})
                for index, r in testdf.iterrows():
                    value = None
                    context = ""
                    if len(active_labels) > 0:
                        for cvarind, val in enumerate(index[:-1]):
                            context += "{}={},".format(active_labels[cvarind], val)
                        context = context[:-1]
                        value = index[-1]
                    else:
                        context = "None"
                        value = index

                    #print(context)
                    if context not in counts[var]:
                        counts[var][context] = {"counts": {}}
                    counts[var][context]["counts"][value] = r.values[0]
                    counts[var][context]["context_vars"] = active_labels
                #print("counts: {}".format(counts[var][context]["counts"]))
                # loop over all observed contexts.
                # or maybe it wiill be hard to keep track of the contexts now..

                
                #print("scoring var: {}, cont: {}".format(var, context))
                #score = score_context(var, context, active_labels, cards_dict, counts, alpha_tot=alpha_tot, method=method)
                #scores[var][context] = score

            # print("going through contexts in counts[{}]: {}".format(var, counts[var]))

        for count_context in counts[var]: # this may double count some contexts
            #print("\nscoring var: {}, cont: {}".format(var, count_context))
            active_labels = counts[var][count_context]["context_vars"]
            score = score_context(var, count_context, active_labels, cards_dict, counts, alpha_tot=alpha_tot, method=method)
            scores[var][count_context] = score
            #print("score: {}".format(score))
    #print("counts:")
    #pp(counts)
    return scores

def order_score_tables(data: pd.DataFrame,
                 strategy="posterior", max_cvars=2,
                 alpha_tot=1.0, method="BDeu"):

    context_scores = score_tables(data, strategy=strategy, max_cvars=max_cvars,
                                alpha_tot=alpha_tot, method=method)
    labels = list(data.columns)
    #print("labels: {}".format(labels))
    cards_dict = {var: data.loc[0, var] for var in data.columns }

    #log_n_stagings = [np.log(learn.n_stagings(cards, lev, max_cvars=max_cvars)) for lev in range(len(labels))]

    p = data.shape[1]
    #print(p)
    order_scores = {var: {} for var in labels}
    #for var in tqdm(labels):
    for var in labels:
        #print("\nVARIABLE: {}".format(var))
        for subset in csi_rel._powerset(set(labels) - {var}):
            # choosing one reperesentateve for each subset
            staging_level = len(subset)-1
            print("staging level: {}".format(staging_level))
            subset = list(subset)
            subset.sort()
            #print(subset)
            subset_str = ','.join([str(v) for v in subset])
            if subset_str == "":
                subset_str = "None"
            order_scores[var][subset_str] = 0
            cards = [cards_dict[l] for l in subset]

            # put the variable t the right and use all_stagings
            for i, staging in enumerate(learn.all_stagings(cards, staging_level, max_cvars)):

                #print("\ngoing trough stages in staging at level {}".format(staging_level))
                staging_marg_lik = 0

                if staging == []: # special case at level -1
                    staging_marg_lik = context_scores[var]["None"]

                for stage in staging: # this might be empty?
                    #print("stage {}".format(stage))
                    # get the context variables
                    stage_context = ""
                    if all([isinstance(stl, set) for stl in stage.list_repr]) or (len(stage.list_repr)==0):
                        stage_context = "None"
                    else:
                        for cvarind, val in enumerate(stage.list_repr):
                            if isinstance(val, int): # a context variable
                                stage_context += "{}={},".format(subset[cvarind], val)
                        stage_context = stage_context[:-1]


                    #print("context {} score: {}".format(stage_context, context_scores[var][stage_context]))
                    staging_marg_lik += context_scores[var][stage_context]


                #print("staging score: {}".format(staging_marg_lik))

                if i == 0: # this is for the log sum trick. It needs a starting scores.
                    order_scores[var][subset_str] = staging_marg_lik
                else:
                    #print([order_scores[var][subset_str], staging_marg_lik])
                    order_scores[var][subset_str] = logsumexp([order_scores[var][subset_str], staging_marg_lik])

            #log_staging_prior = log_n_stagings[staging_level]
            log_level_prior = -np.log(p- staging_level-1) # BUG: Is this correct?
            log_staging_prior = -np.log(learn.n_stagings(cards, staging_level, max_cvars=max_cvars))
            log_unnorm_post = order_scores[var][subset_str] + log_staging_prior + log_level_prior
            print("log_level_prior: {}".format(log_level_prior))
            print("log_staging_prior: {}".format(log_staging_prior))
            print("log_likelihood: {}".format(order_scores[var][subset_str]))
            order_scores[var][subset_str] = log_unnorm_post

    return order_scores

def score(cstree: ct.CStree, data: pd.DataFrame, alpha_tot=1.0, method="BDeu"):
    """Score a CStree.

    Args:
        cstree (ct.CStree): CStree.
        data (pandas DataFrame): The data.
        alpha_tot (float): Hyper parameter for the stage parameters Dirichlet prior distribution. Defaults to 1.0.
        method (str, optional): Prior for the stage parameters. Defaults to "BDeu".

    Returns:
        float: The score of the CStree.

    Example:
        >>> import random
        >>> import numpy as np
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> sc.score(tree, df, alpha_tot=1.0, method="BDeu")
        -1829.2311869978726
    """

    score = 0  # log score
    for level in range(cstree.p):
        level_counts = counts_at_level(cstree, level, data)
        score += score_level(cstree, level, level_counts, alpha_tot, method)
    return score

def score_order_tables(order, order_scores):
    log_score = 0  # log score
    for level, var in enumerate(order):
        poss_parents = order[:level]
        # possible parents as string
        poss_parents_str = ','.join([str(v) for v in poss_parents])
        if poss_parents_str == "":
            poss_parents_str = "None"
        #print("var: {}, poss_parents: {}, poss_parents_str: {}".format(var, poss_parents, poss_parents_str))
        score = order_scores[var][poss_parents_str]
        print("score: {}".format(score))
        log_score += score

    return log_score


def score_order(order, data, strategy="max", max_cvars=1, alpha_tot=1.0, method="BDeu"):
    """Score an order.

    Args:
        order (list): The order of the variables.
        data (pandas DataFrame): The data.
        strategy (str, optional): Strategy for scoring. Defaults to "max".
        max_cvars (int, optional): Maximum number of children per variable. Defaults to 1.
        alpha_tot (float, optional): Hyper parameter for the stage parameters Dirichlet prior distribution. Defaults to 1.0.
        method (str, optional): Prior for the stage parameters. Defaults to "BDeu".

    Returns:
        float: The score of the order.

    Example:
        >>> import random
        >>> import numpy as np
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> sc.score_order([0, 1, 2, 3], df, max_cvars=1, alpha_tot=1.0, method="BDeu")
        -1829.2311869978726
    """

    log_score = 0  # log score
    for level in range(len(order)):
        s = _score_order_at_level(order, level, data, strategy=strategy,
                                  max_cvars=max_cvars, alpha_tot=alpha_tot,
                                  method=method)
        if strategy == "max":
            log_score += s
        elif strategy == "posterior":
            print("score at level", level, ":", s)
            log_score += s
    #print(log_score)
    return log_score


def logsumexp(x):
    """Log sum exp trick function.

    Args:
        x (numpy array): Array of numbers.
    Returns:
        float: The log of the sum of the exponentials of the numbers in x.
    """

    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _score_order_at_level(order, level, data, strategy="max", max_cvars=1,
                          alpha_tot=1.0, method="BDeu"):
    """ Without singletons, there are 2*level stagings at level level.
        val1 side and val2 side colored in different ways
        val1 side and val2 side colored in the same way

    Args:
        order (list): The order of the variables.
        level (int): The level at which to score.
        data (pandas DataFrame): The data.
        max_cvars (int, optional): _description_. Defaults to 1.
        alpha_tot (float, optional): _description_. Defaults to 1.0.
        method (str, optional): _description_. Defaults to "BDeu".

    Returns:
        float: The score of the order at level level.
    """

    if max_cvars > 2:
        print("Only max_cvars < 3 implemented")
        return None

    p = len(order)

    # BUG?: Maybe these have to be adapted to the order.
    cards = data.iloc[0].values
    tree = ct.CStree(cards)
    # here we set the labels/order for the CStree, to be used in the counting.
    tree.labels = order
    #print("var: {}".format(tree.labels[level]))
    stagings = learn.all_stagings(cards, level-1, max_cvars=max_cvars)

    if strategy == "max":
        log_unnorm_post = -np.inf  # log score
    elif strategy == "posterior":
        log_unnorm_post = 0
        log_unnorm_posts = []

    # This is a bit clumsy but shoud work.
    # I think we need to go through all these to get the max value and then
    # do the logsumexp trick.
    for stlist in stagings:
        # print(stlist)
        # print all stges in a stlist
        # for st in stlist:
        #    print(st)

        # Need to set the stagings in order to count.
        tree.update_stages({level-1: stlist})
        level_counts = counts_at_level(tree, level, data)
        #for st, cnts in level_counts.items():
        #    print(st, cnts)

        log_marg_lik = score_level(
            tree, level, level_counts, alpha_tot, method)
        # print("level {} score: {}".format(l, tmp))

        if strategy == "max":
            if log_marg_lik > log_unnorm_post:
                log_unnorm_post = log_marg_lik
        if strategy == "posterior":
            log_staging_prior = -np.log(learn.n_stagings(
                cards, level-1, max_cvars=max_cvars))
            
            print("stagings level: {}".format(level-1))
            log_level_prior = -np.log(p-level) # BUG: Is this correct?
            print("log_level_prior: {}".format(log_level_prior))
            print("log_staging_prior: {}".format(log_staging_prior))
            print("log_marg_lik: {}".format(log_marg_lik))
            log_unnorm_post = log_marg_lik + log_staging_prior + log_level_prior
            log_unnorm_posts.append(log_unnorm_post)
            #log_unnorm_posts.append(log_marg_lik)

            #print("level {} log lik score: {}".format(level, log_unnorm_post))

    if strategy == "posterior":
        # TODO: Maybe one should use a generator isntead to save some space.
        # Then one can first fint the max and then to the logsumexp trick.
        #print("number of stagings: {}".format(len(log_unnorm_posts)))
        #print("lc: {}".format(log_unnorm_posts))
        outp = logsumexp(log_unnorm_posts)
        #print("logsumexp: {}".format(outp))
        return logsumexp(log_unnorm_posts)
    elif strategy == "max":
        return log_unnorm_post
