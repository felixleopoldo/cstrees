from scipy.special import loggamma
import numpy as np
import pandas as pd

import cstrees.learning as learn
import cstrees.cstree as ct
import cstrees.stage as st


def counts_at_level(cstree: ct.CStree, level: int, data):
    """ Collect all the observed counts at a specific level by stages.
    So the counts for level l depends on the stage of level l-1.
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
    stage_counts = {}

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

    score = 0  # log score
    for stage, counts in level_counts.items():
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
        
        stage_counts_stage = sum(counts.values())
        #print("stage counts: {}".format(stage_counts_stage))

        score += loggamma(alpha_stage) - \
            loggamma(alpha_stage + stage_counts_stage)
        for val in range(cstree.cards[level]):
            if val not in counts:  # as we only store the observed values
                continue
            score += loggamma(alpha_obs + counts[val]) - loggamma(alpha_obs)

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
        log_marg_lik = score_level(
            tree, level, level_counts, alpha_tot, method)
        #print("level {} score: {}".format(l, tmp))

        if strategy == "max":
            if log_marg_lik > log_unnorm_post:
                log_unnorm_post = log_marg_lik
        if strategy == "posterior":
            log_staging_prior = np.log(learn.n_stagings(
                cards, level-1, max_cvars=max_cvars))
            log_level_prior = np.log((level+1) / p) # BUG: Is this correct?
            log_unnorm_post = log_marg_lik - log_staging_prior - log_level_prior
            log_unnorm_posts.append(log_unnorm_post)

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
