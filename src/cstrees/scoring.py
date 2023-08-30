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
    stage_counts = {}  # TODO: Maybe it should be context counts instead!

    # reorder the columns according to the order.
    # cardinalities are at first row.

    dataperm = data[cstree.labels].values[1:, :]

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

    context_counts = sum(counts[var][context]["counts"].values())

    # Note that the score is depending on the context in the stage. So
    # note really th satge as such.
    score = loggamma(alpha_context) - loggamma(alpha_context + context_counts)
    for val, count in counts[var][context]["counts"].items():
        score += loggamma(alpha_obs + count) - loggamma(alpha_obs)

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
        alpha_stage = alpha_tot * cstree.stage_proportion(stage)
        alpha_obs = alpha_stage / cstree.cards[level]

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


def score_tables(data: pd.DataFrame,
                 strategy="posterior", max_cvars=2, poss_cvars=[],
                 alpha_tot=1.0, method="BDeu"):

    cards_dict = {var: data.loc[0, var] for var in data.columns}
    scores = {}
    scores["cards"] = cards_dict
    scores["scores"] = {lab: {} for lab in data.columns}


    counts = {}
    counts["cards"] = cards_dict
    counts["var_counts"] = {lab: {} for lab in data.columns}

    # go through all variables
    for var in tqdm(data.columns, desc="Context score tables"):
        # Iterate through all context sizes
        for csize in range(max_cvars+1):
            # Iterate through all possible contexts
            # remove the current variable from the active labels
            labels = [l for l in data.columns if l != var]

            # Restricting to some possible context variables.
            for context_variables in combinations([l for l in labels if l in poss_cvars[var]], csize):
                # get the active labels like A,B,C
                active_labels = sorted([l for l in labels if l in context_variables])                
                tmp = {c: None for c in active_labels}

                if len(active_labels) == 0:
                    test = data[1:][var].value_counts()
                else:
                    test = data[1:].groupby(active_labels)[var].value_counts()

                # get the counts
                testdf = test.to_frame().rename(
                    columns={var: str(var)+" counts"})
                for index, r in testdf.iterrows():
                    value = None

                    # Sort variables
                    context = ""
                    if len(active_labels) > 0:
                        for cvarind, val in enumerate(index[:-1]):
                            context += "{}={},".format(
                                active_labels[cvarind], val)
                        context = context[:-1]
                        value = index[-1]
                    else:
                        context = "None"
                        value = index

                    if context not in counts["var_counts"][var]:
                        counts["var_counts"][var][context] = {"counts": {}}
                    counts["var_counts"][var][context]["counts"][value] = r.values[0]
                    counts["var_counts"][var][context]["context_vars"] = active_labels

        # Using the counts to compute the scores
        for count_context in counts["var_counts"][var]:
            active_labels = counts["var_counts"][var][count_context]["context_vars"]
            score = score_context(var, count_context, active_labels, cards_dict,
                                  counts["var_counts"], alpha_tot=alpha_tot, method=method)
            scores["scores"][var][count_context] = score
    
    return scores, counts


def list_to_score_key(labels: list):
    subset = labels
    subset.sort()
    subset_str = ','.join([str(v) for v in subset])
    if subset_str == "":
        subset_str = "None"
    return subset_str


def stage_to_context_key(stage: st.Stage, labels: list):
    stage_context = ""
    if stage.to_csi().context.context == {}:
        stage_context = "None"
    else:
        # need to relabeled first
        cvars = {}
        for cvarind, val in enumerate(stage.list_repr):

            if isinstance(val, int):  # a context variable
                cvars[labels[cvarind]] = val

        for cvar, val in sorted(cvars.items()):
            stage_context += "{}={},".format(cvar, val)
        stage_context = stage_context[:-1]

    return stage_context


def log_n_stagings_tables(labels, cards_dict, poss_cvars, max_cvars=2):
    n_stagings = {}

    # the number of staging for a set of cardinalities [2,3,2] should be
    # independent of the order, so same for [2,2,3]

    for var in tqdm(labels, desc="Creating #stagings tables"):
        # all cards except the current one
        cur_cards = [cards_dict[l] for l in labels if (l != var) and (l in poss_cvars[var])]
        for subset in csi_rel._powerset(cur_cards):
            staging_lev = len(subset) - 1
            subset_str = list_to_score_key(list(subset))

            if subset_str not in n_stagings:
                n_stagings[subset_str] = np.log(learn.n_stagings(list(subset),
                                                                 staging_lev,
                                                                 max_cvars=max_cvars))
    return n_stagings


def order_score_tables(data: pd.DataFrame,
                       strategy="posterior", max_cvars=2,
                       poss_cvars=None,
                       alpha_tot=1.0, method="BDeu"):

    labels = list(data.columns)

    # If poss_cvars is None, then all variables are possible context variables
    if poss_cvars is None:
        poss_cvars = {l:list(set(labels) - {l}) for l in labels}

    context_scores, context_counts = score_tables(data, strategy=strategy, max_cvars=max_cvars,
                                                  poss_cvars=poss_cvars,
                                                  alpha_tot=alpha_tot, method=method)

    context_scores["max_cvars"] = max_cvars
    context_scores["poss_cvars"] = poss_cvars


    cards_dict = {var: data.loc[0, var] for var in data.columns}

    log_n_stagings = log_n_stagings_tables(
        labels, cards_dict, poss_cvars, max_cvars=max_cvars)

    p = data.shape[1]

    order_scores = {}
    order_scores["max_cvars"] = max_cvars
    order_scores["poss_cvars"] = poss_cvars
    order_scores["scores"] = {var: {} for var in labels}
    for var in tqdm(labels, desc="Order score tables"):
        
        # Ths subset are the variables before var in the order
        for subset in csi_rel._powerset(poss_cvars[var]):
            # TODO: It should sum over all the subsets for each subset. 
            # This could be done faster using Hasse diagrams? 
            staging_level = len(subset)-1

            subset_str = list_to_score_key(list(subset))

            
            cards = [cards_dict[l] for l in subset]

            # Te prior is uniform voer all stagings so we have it outside (ok?)
            cards_str = list_to_score_key(cards[:staging_level+1])
            log_staging_prior = -log_n_stagings[cards_str]
            log_level_prior = -np.log(p - staging_level-1) 

            for i, staging in enumerate(learn.all_stagings(cards, staging_level, max_cvars=max_cvars)):

                staging_unnorm_post = log_level_prior + log_staging_prior
                # this is for the level -1 
                if staging == []:  # special case at level -1                    
                    staging_unnorm_post += context_scores["scores"][var]["None"]

                # Sum log-marginal likelihood of all stages in the staging
                for stage in staging:
                    stage_context = stage_to_context_key(stage, subset) # OK! even when restricting to some possible cvars
                    staging_unnorm_post += context_scores["scores"][var][stage_context]

                if i == 0:
                    order_scores["scores"][var][subset_str] = staging_unnorm_post
                else:                    
                    order_scores["scores"][var][subset_str] = logsumexp(
                        [order_scores["scores"][var][subset_str], staging_unnorm_post])

    return order_scores, context_scores, context_counts


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


def score_order(order, order_scores):
    log_score = 0  # log score
    for level, var in enumerate(order):
        poss_parents = list(set(order[:level]) & set(order_scores["poss_cvars"][var]))
        # possible parents as string
        poss_parents_str = list_to_score_key(poss_parents)
        score = order_scores["scores"][var][poss_parents_str]
        log_score += score

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

