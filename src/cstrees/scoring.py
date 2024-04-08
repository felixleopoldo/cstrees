from itertools import combinations

from scipy.special import loggamma
import numpy as np
import pandas as pd
from tqdm import tqdm

import cstrees.learning as learn
import cstrees.cstree as ct
import cstrees.stage as st
import cstrees.dependence as csi_rel


def _counts_at_level(cstree: ct.CStree, level: int, data):
    """Collect all the observed counts at a specific level by stages.
    So the counts for level l depends on the stage of level l-1.
    (we probably have to ase these on context instead of stage)
    This is a bit legacy, it is only used when estimating parameters for a stage in th CStee class.
    It should probably be replaced with some precalculations.. But that would induce more calculations..

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
           0  1  2  3
        0  2  2  2  2
        1  *  -  -  -
        2  *  1  -  -
        3  *  0  -  -
        4  0  *  *  -
        5  1  *  *  -
        6  -  -  -  -
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> counts = sc._counts_at_level(tree, 2, df)
        >>> for key, val in counts.items():
        >>>    print("Stage: {}".format(key))
        >>>    print("Counts: {}".format(val))
        Stage: [{0, 1}, 0]; probs: [0.58753532 0.41246468]; color: blueviolet
        Counts: {0: 184, 1: 146}
        Stage: [{0, 1}, 1]; probs: [0.45616876 0.54383124]; color: peru
        Counts: {0: 289, 1: 381}
    """
    stage_counts = {}  # TODO: Maybe it should be context counts instead!

    # reorder the columns according to the order.
    # cardinalities are at first row.

    dataperm = data[cstree.labels].values[1:, :]

    for i in range(len(dataperm)):  # iterate over the samples
        pred_vals = dataperm[i, :level]
        stage = cstree.get_stage(pred_vals)  # or context
        # print('pred_vals: ', pred_vals)
        # print("stages at level {}: {}".format(l-1, t.stages[l-1]))
        if stage is None:  # singleton stage. Shold note be any of these in our setting.
            print("singleton stage")
        if stage not in stage_counts:
            # only save the observed ones #[0] * t.cards[l]  # initiate with
            # zeros.x
            stage_counts[stage] = {}
        if dataperm[i, level] in stage_counts[stage]:
            stage_counts[stage][dataperm[i, level]] += 1
        else:
            stage_counts[stage][dataperm[i, level]] = 1
    return stage_counts


def _score_context(
    var, context, context_vars, cards, counts, alpha_tot=1.0, method="BDeu"
):
    """Building block for the CS-BDeu score as defined in C. Hughes et al., but here we calculate it for a specific variable and a context.
    These are then combined to get the CS-BDeu score.

    Args:
        var (str): Variable label e.g. X4.
        context (str): Context e.g. the string "X1=1,X3=0".
        context_vars (list): List of context variables, e.g. ["X1", "X3"].
        cards (dict): Dictionary with cardinalities for all variables.
        counts (dict): Counts for all the stage (sufficient statistics).
        alpha_tot (float, optional): Hyper parameter for the stage parameters Dirichlet prior distribution. Defaults to 1.0.
        method (str, optional): Parameter estimator. Defaults to "BDeu".

    Reference:
        C. Hughes, P. Strong, and A. Shenvi. Score equivalence for staged trees, 2023, https://arxiv.org/abs/2206.15322


    Returns:
        float: The context score for var.
    """
    if method == "K2":
        assert alpha_tot == 1
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
    # note really the stage as such.
    score = loggamma(alpha_context) - loggamma(alpha_context + context_counts)
    for val, count in counts[var][context]["counts"].items():
        score += loggamma(alpha_obs + count) - loggamma(alpha_obs)

    return score


def _estimate_parameters(
    cstree: ct.CStree, stage, stage_counts, method="BDeu", alpha_tot=1.0
):
    """Estimate the parameters for a stage.
        TODO: This should probably depend on the context counts instead of the stage counts.
        It is legacy code and only called from the CStree class atm so its works anyway.

    Args:
        cstree (ct.CStree): A CStree.
        stage (Stage): A stage.
        stage_counts (dict): Counts for all the stage (sufficient statistics).
        method (string): Estimation method. Defaults to "BDeu".
        alpha_tot (float): Hyper parameter for the stage parameters Dirichlet prior distribution. Defaults to 1.0.

    Returns:
        list: List of probabilities associated with the stage. I.e conditional probabilities for the variable one level up.


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
            if alpha_obs == 0:
                probs[i] = 0
            else:
                probs[i] = alpha_obs / alpha_stage
        else:  # posterior mean or posterior predictive probabilites.
            probs[i] = (alpha_obs + stage_counts[stage][i]) / (
                alpha_stage + stage_counts_total
            )
    return probs


def _context_score_tables(
    data: pd.DataFrame,
    strategy="posterior",
    max_cvars=2,
    poss_cvars: dict | None = None,
    alpha_tot=1.0,
    method="BDeu",
):
    """Generates the context score tables for a dataset.

    Args:
        data (pd.DataFrame): A dataset.
        strategy (str, optional): Defaults to "posterior".
        max_cvars (int, optional): Maximum number of variables in a context. Defaults to 2.
        poss_cvars (dict | None, optional): Possible context variabels for each variable. Defaults to None which means all.
        alpha_tot (float, optional): Hyper parameter for the Dirichlet distribution, total pseudo counts per variable. Defaults to 1.0.
        method (str, optional): Scoring method. Defaults to "BDeu".

    Returns:
        dict: Context scores for each variable.

    Example:
        >>> import random
        >>> import numpy as np
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> import pp
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.sample_stage_parameters(alpha=1.0)
        >>> df = tree.sample(1000)
        >>> context_scores, context_counts = sc.context_score_tables(df, strategy="posterior",
        >>>                                                          max_cvars=1,
        >>>                                                          poss_cvars=None,
        >>>                                                          alpha_tot=1.0,
        >>>                                                          method="BDeu")
        >>> pp.pprint(context_scores)
        Context score tables: 100%|██████████| 3/3 [00:00<00:00, 320.68it/s]
        {'cards': {0: 2, 1: 2, 2: 2},
        'scores': {0: {'1=0': -16.792280918610103,
                        '1=1': -660.7109764001227,
                        '2=0': -238.70011824545145,
                        '2=1': -438.93117780248804,
                        'None': -675.4564323010165},
                    1: {'0=0': -54.16769454491233,
                        '0=1': -53.46673897600048,
                        '2=0': -33.2126022029158,
                        '2=1': -74.90053928069584,
                        'None': -105.58760850319653},
                    2: {'0=0': -381.93689163806187,
                        '0=1': -266.9232447908356,
                        '1=0': -14.774052078307868,
                        '1=1': -634.4367535840818,
                        'None': -646.6852726819745}}}
    """

    labels = list(data.columns)

    if poss_cvars is None:
        poss_cvars = {l: list(set(labels) - {l}) for l in labels}

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
        for csize in range(max_cvars + 1):
            # Iterate through all possible contexts
            # remove the current variable from the active labels
            labels = [l for l in data.columns if l != var]

            # Restricting to some possible context variables.
            for context_variables in combinations(
                [l for l in labels if l in poss_cvars[var]], csize
            ):
                # get the active labels like A,B,C
                active_labels = sorted([l for l in labels if l in context_variables])
                tmp = {c: None for c in active_labels}

                if len(active_labels) == 0:
                    test = data[1:][var].value_counts()
                else:
                    test = data[1:].groupby(active_labels)[var].value_counts()

                # get the counts
                testdf = test.to_frame().rename(columns={var: str(var) + " counts"})
                for index, r in testdf.iterrows():
                    value = None

                    # Sort variables
                    context = ""
                    if len(active_labels) > 0:
                        for cvarind, val in enumerate(index[:-1]):
                            context += "{}={},".format(active_labels[cvarind], val)
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
            score = _score_context(
                var,
                count_context,
                active_labels,
                cards_dict,
                counts["var_counts"],
                alpha_tot=alpha_tot,
                method=method,
            )
            scores["scores"][var][count_context] = score

    return scores, counts


def _list_to_score_key(labels: list):
    subset = sorted(labels)
    subset_str = ",".join([str(v) for v in subset])
    if subset_str == "":
        subset_str = "None"
    return subset_str


def _stage_to_context_key(stage: st.Stage, labels: list):
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


def _log_n_stagings_tables(labels, cards_dict, poss_cvars, max_cvars=2):
    n_stagings = {}

    # the number of staging for a set of cardinalities [2,3,2] should be
    # independent of the order, so same for [2,2,3]

    for var in tqdm(labels, desc="Creating #stagings tables"):
        # all cards except the current one
        cur_cards = [
            cards_dict[l] for l in labels if (l != var) and (l in poss_cvars[var])
        ]
        for subset in csi_rel._powerset(cur_cards):
            staging_lev = len(subset) - 1
            subset_str = _list_to_score_key(list(subset))

            if subset_str not in n_stagings:
                n_stagings[subset_str] = np.log(
                    learn.n_stagings(list(subset), staging_lev, max_cvars=max_cvars)
                )
    return n_stagings


def order_score_tables(
    data: pd.DataFrame,
    strategy="posterior",
    max_cvars=2,
    poss_cvars: dict | None = None,
    alpha_tot=1.0,
    method="BDeu",
):
    """Calculatee the order score tables for a dataset.

    Args:
        data (pd.DataFrame): A dataset.
        strategy (str, optional):  Defaults to "posterior".
        max_cvars (int, optional): Max number of variables in a context. Defaults to 2.
        poss_cvars (dict | None, optional): Dict with possible context variables for a varible. Defaults to None meaning all.
        alpha_tot (float, optional): BDeu score parameter (pseudo counts). Defaults to 1.0.
        method (str, optional): Scoring method. Defaults to "BDeu".

    Returns:
        tuple: The order score tables, context score tables, and context counts.

    Example:

        >>> import cstrees.learning as ctl
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> import pp
        >>> import numpy as np
        >>> import random
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>>
        >>> tree = ct.sample_cstree([2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1,
        >>>                         labels=["X"+str(i) for i in range(1, 4)])
        >>> tree.sample_stage_parameters(1.0)
        >>> df = tree.sample(500)
        >>> score_table, context_scores, context_counts = sc.order_score_tables(df,
        >>>                                                                     max_cvars=1,
        >>>                                                                     alpha_tot=1.0,
        >>>                                                                     method="BDeu",
        >>>                                                                     poss_cvars=None)
        >>> print("Order score table:")
        >>> pp.pprint(score_table)
        >>> print("Context scores:")
        >>> pp.pprint(context_scores)
        >>> print("Context counts:")
        >>> pp.pprint(context_counts)
        Order score table:
        {'max_cvars': 1,
        'poss_cvars': {'X1': ['X2', 'X3'], 'X2': ['X1', 'X3'], 'X3': ['X2', 'X1']},
        'scores': {'X1': {'None': -337.8948102114355,
                        'X2': -338.07301225936493,
                        'X2,X3': -337.66375682421136,
                        'X3': -338.04776148301414},
                    'X2': {'None': -68.29479077800046,
                        'X1': -68.47299282592986,
                        'X1,X3': -68.06642821338981,
                        'X3': -68.4507053561074},
                    'X3': {'None': -321.52156911602725,
                        'X1': -321.67452038760587,
                        'X1,X2': -321.27075455994964,
                        'X2': -321.6774836941342}}}
        Context scores:
        {'cards': {'X1': 2, 'X2': 2, 'X3': 2},
        'max_cvars': 1,
        'poss_cvars': {'X1': ['X2', 'X3'], 'X2': ['X1', 'X3'], 'X3': ['X2', 'X1']},
        'scores': {'X1': {'None': -336.7961979227674,
                        'X2=0': -11.762075046683464,
                        'X2=1': -327.19089667302535,
                        'X3=0': -116.56072426147412,
                        'X3=1': -222.1718285846568},
                    'X2': {'None': -67.19617848933235,
                        'X1=0': -36.38485547387102,
                        'X1=1': -32.96809681240273,
                        'X3=0': -17.62998621847001,
                        'X3=1': -51.52630149961991},
                    'X3': {'None': -320.42295682735914,
                        'X1=0': -191.00840864197295,
                        'X1=1': -131.3509031087497,
                        'X2=0': -9.235342639991355,
                        'X2=1': -313.14772341612536}}}
        Context counts:
        {'cards': {'X1': 2, 'X2': 2, 'X3': 2},
        'var_counts': {'X1': {'None': {'context_vars': [], 'counts': {0: 307, 1: 193}},
                            'X2=0': {'context_vars': ['X2'], 'counts': {0: 7, 1: 7}},
                            'X2=1': {'context_vars': ['X2'],
                                        'counts': {0: 300, 1: 186}},
                            'X3=0': {'context_vars': ['X3'],
                                        'counts': {0: 92, 1: 73}},
                            'X3=1': {'context_vars': ['X3'],
                                        'counts': {0: 215, 1: 120}}},
                        'X2': {'None': {'context_vars': [], 'counts': {0: 14, 1: 486}},
                            'X1=0': {'context_vars': ['X1'],
                                        'counts': {0: 7, 1: 300}},
                            'X1=1': {'context_vars': ['X1'],
                                        'counts': {0: 7, 1: 186}},
                            'X3=0': {'context_vars': ['X3'],
                                        'counts': {0: 3, 1: 162}},
                            'X3=1': {'context_vars': ['X3'],
                                        'counts': {0: 11, 1: 324}}},
                        'X3': {'None': {'context_vars': [], 'counts': {0: 165, 1: 335}},
                            'X1=0': {'context_vars': ['X1'],
                                        'counts': {0: 92, 1: 215}},
                            'X1=1': {'context_vars': ['X1'],
                                        'counts': {0: 73, 1: 120}},
                            'X2=0': {'context_vars': ['X2'],
                                        'counts': {0: 3, 1: 11}},
                            'X2=1': {'context_vars': ['X2'],
                                        'counts': {0: 162, 1: 324}}}}}
    """

    labels = list(data.columns)

    # If poss_cvars is None, then all variables are possible context variables
    if poss_cvars is None:
        poss_cvars = {l: list(set(labels) - {l}) for l in labels}

    context_scores, context_counts = _context_score_tables(
        data,
        strategy=strategy,
        max_cvars=max_cvars,
        poss_cvars=poss_cvars,
        alpha_tot=alpha_tot,
        method=method,
    )

    context_scores["max_cvars"] = max_cvars
    context_scores["poss_cvars"] = poss_cvars

    cards_dict = {var: data.loc[0, var] for var in data.columns}

    log_n_stagings = _log_n_stagings_tables(
        labels, cards_dict, poss_cvars, max_cvars=max_cvars
    )

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
            staging_level = len(subset) - 1

            subset_str = _list_to_score_key(list(subset))

            cards = [cards_dict[l] for l in subset]

            # Te prior is uniform voer all stagings so we have it outside (ok?)
            cards_str = _list_to_score_key(cards[: staging_level + 1])
            log_staging_prior = -log_n_stagings[cards_str]
            log_level_prior = -np.log(p - staging_level - 1)

            for i, staging in enumerate(
                learn.all_stagings(cards, staging_level, max_cvars=max_cvars)
            ):
                staging_unnorm_post = log_level_prior + log_staging_prior
                # this is for the level -1
                if staging == []:  # special case at level -1
                    staging_unnorm_post += context_scores["scores"][var]["None"]

                # Sum log-marginal likelihood of all stages in the staging
                for stage in staging:
                    # OK! even when restricting to some possible cvars
                    stage_context = _stage_to_context_key(stage, subset)
                    if stage_context in context_scores["scores"][var]:
                        staging_unnorm_post += context_scores["scores"][var][stage_context]

                if i == 0:
                    order_scores["scores"][var][subset_str] = staging_unnorm_post
                else:
                    order_scores["scores"][var][subset_str] = _logsumexp(
                        [order_scores["scores"][var][subset_str], staging_unnorm_post]
                    )

    return order_scores, context_scores, context_counts


def score_order(order, order_scores):
    """Scores an order using the order score tables. The score is the sum of the individual variable scores for each level.

    Args:
        order (list): List of variables in the order.
        order_scores (dict): Order scores.

    Returns:
        double: Score of an order.

    Example:

        >>> import cstrees.learning as ctl
        >>> import cstrees.cstree as ct
        >>> import cstrees.scoring as sc
        >>> import pp
        >>> import numpy as np
        >>> import random
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>>
        >>> tree = ct.sample_cstree([2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1,
        >>>                          labels=["X"+str(i) for i in range(1, 4)])
        >>> tree.sample_stage_parameters(1.0)
        >>> df = tree.sample(500)
        >>> score_table, context_scores, context_counts = sc.order_score_tables(df,
        >>>                                                                     max_cvars=1,
        >>>                                                                     alpha_tot=1.0,
        >>>                                                                     method="BDeu",
        >>>                                                                     poss_cvars=None)
        >>> sc.score_order(["X3","X2","X1"], score_table)
        -727.636031296346

    """
    log_score = 0  # log score
    for level, var in enumerate(order):
        poss_parents = list(set(order[:level]) & set(order_scores["poss_cvars"][var]))
        # possible parents as string
        poss_parents_str = _list_to_score_key(poss_parents)
        score = order_scores["scores"][var][poss_parents_str]
        log_score += score

    return log_score


def _logsumexp(x):
    """Log sum exp trick function.

    Args:
        x (numpy array): Array of numbers.
    Returns:
        float: The log of the sum of the exponentials of the numbers in x.
    """

    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))
