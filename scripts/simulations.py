#!python3
import contextlib
from functools import partial, reduce
from importlib.metadata import version
from itertools import pairwise, product
import operator
import os
from pathlib import Path
import random
import time
import warnings

old_environ = dict(os.environ)
os.environ.update({"TQDM_DISABLE": "1", "TQDM_ENABLE": "0", "NUMEXPR_MAX_THREADS": "8"})


from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.PermutationBased.GRaSP import grasp
import numpy as np
import matplotlib as mpl

import pandas as pd

from scipy.special import rel_entr


from cstrees import cstree as ct
from cstrees.evaluate import kl_divergence
from cstrees.evaluate import KL_divergence
import cstrees.learning as ctl
import cstrees.scoring as sc


def generate_data_and_true_distr(path, seeds, samp_size_range, num_levels_range):
    """ Generate data and true distributions for all combinations of parameters.
    """

    data_path = f"{path}/data"
    distr_path = f"{path}/distr/true"
    Path(data_path).mkdir(parents=True, exist_ok=True)
    Path(distr_path).mkdir(parents=True, exist_ok=True)

    for num_levels in num_levels_range:
        cards = [2] * num_levels
        for samp_size in samp_size_range:
            for seed in seeds:
                print(f"Starting run {seed} for p={num_levels}...")

                np.random.seed(seed)
                random.seed(seed)

                # generate random cstree and compute its kl divergence
                tree = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
                tree.sample_stage_parameters(alpha=2)
                data = tree.sample(samp_size)
                tree_df = tree.to_joint_distribution(label_order=list(data.columns))
                tree_df.to_csv(f"{distr_path}/p={num_levels}_n={samp_size}_seed={seed}.csv", index=False)
                data.to_csv(f"{data_path}/p={num_levels}_n={samp_size}_seed={seed}.csv", index=False)

def estimate_cstree_distr(data_path, est_path, seeds, samp_size_range, num_levels_range):

    """ Estimate the CStree for all datasets in data_path.
    """
    Path(est_path).mkdir(parents=True, exist_ok=True)
    
    # get all data files
    for num_levels in num_levels_range:
        cards = [2] * num_levels
        for samp_size in samp_size_range:
            for seed in seeds:
                name = f"p={num_levels}_n={samp_size}_seed={seed}.csv"
    
                print(f"Estimating CStree for {data_path}/{name}...")
                if Path(f"{est_path}/{name}").is_file():
                    continue
                data = pd.read_csv(f"{data_path}/{name}")
                
                score_table, context_scores, context_counts = sc.order_score_tables(
                data, max_cvars=2, alpha_tot=1.0, method="BDeu", poss_cvars=None)
                orders, scores = ctl.gibbs_order_sampler(5000, score_table)
                maporder = orders[scores.index(max(scores))]
                tree = ctl._optimal_cstree_given_order(maporder, context_scores)
                tree.estimate_stage_parameters(data)

                tree_df = tree.to_joint_distribution(label_order=list(data.columns))
                tree_df.to_csv(f"{est_path}/{name}", index=False)
                        

def kl_div_from_files(true_path, est_path, seeds, samp_size_range, num_levels_range):
    """ Read in true and estimated distributions from files and compute KL divergence.
    """
    # get all distribution files
    kl_divs = []
    for num_levels in num_levels_range:
        cards = [2] * num_levels
        for samp_size in samp_size_range:
            for seed in seeds:
                name = f"p={num_levels}_n={samp_size}_seed={seed}.csv"
                true_dist_df = pd.read_csv(f"{true_path}/{name}")
                est_dist_df = pd.read_csv(f"{est_path}/{name}")
                
                kl_divs.append(KL_divergence(true_dist_df, est_dist_df))
    return np.array(kl_divs)

if __name__ == "__main__":
    # check versions to ensure accurate reproduction
    if version("cstrees") != "1.2.0":
        warnings.warn(f"Current `cstrees` version unsupported.")

    path = "sim_results"
    warnings.simplefilter(action="ignore", category=FutureWarning)

    samp_size_range = [1000]
    seeds = list(range(10))
    num_levels_range = [5, 7]

    generate_data_and_true_distr(path, seeds, samp_size_range, num_levels_range)
    estimate_cstree_distr(f"{path}/data", f"{path}/distr/cstree_est", seeds, samp_size_range, num_levels_range)
    #kldivs = kl_div_from_files(f"{path}/distr/true", f"{path}/distr/cstree_est", seeds, samp_size_range, num_levels_range)
    #kldivs2 = kl_div_from_files(f"{path}/distr/true", f"{path}/distr/stagedtrees", seeds, samp_size_range, num_levels_range)
    kldivs = kl_div_from_files(f"{path}/distr/cstree_est", f"{path}/distr/true", seeds, samp_size_range, num_levels_range)
    kldivs2 = kl_div_from_files(f"{path}/distr/stagedtrees", f"{path}/distr/true", seeds, samp_size_range, num_levels_range)
    
    
    print(kldivs.mean(), kldivs.std())
    print(kldivs2.mean(), kldivs2.std())
    os.environ.clear()
    os.environ.update(old_environ)
