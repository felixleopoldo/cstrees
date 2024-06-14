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
import seaborn as sns
import matplotlib.pyplot as plt
import causaldag as cd
from pgmpy.models import BayesianNetwork
import networkx as nx

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


def pgmpy_to_joint_distribution(model, label_order, cards=None):
    
    # Iterate over all possible outcomes and calculate the probability mass function.
    # Store the outcomes together with the probabilities in a Pandas Dataframe.
    outcomes = product(*[range(card) for card in cards])

    # Create an empty dataframe with the correct column names
    #df_outcomes = pd.DataFrame(columns=self.labels)
    df_outcomes = pd.DataFrame(columns=label_order)
    # store all the outcomes and probabilities
    pmfs = [None]*np.prod(cards)
    pmfs_log = [None]*np.prod(cards)
    for i, outcome in enumerate(outcomes):
        df_outcomes.loc[i] = outcome
        
        pgmpy_outcome = {lab: marg_outcome for lab, marg_outcome in zip(label_order, outcome)}
        prob = model.get_state_probability(pgmpy_outcome)
        
        pmfs[i] = prob
        pmfs_log[i] = np.log(prob)

    df_pmf = pd.DataFrame(pmfs, columns=["prob"])
    df_pmf_log = pd.DataFrame(pmfs_log, columns=["log_prob"])
    # join the two dataframes
    df = pd.concat([df_outcomes, df_pmf, df_pmf_log], axis=1)
    
    return df

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

                if Path(f"{data_path}/p={num_levels}_n={samp_size}_seed={seed}.csv").is_file():
                    data = pd.read_csv(f"{data_path}/p={num_levels}_n={samp_size}_seed={seed}.csv")
                    if all([data[col][1:].nunique() >= 2 for col in data.columns]):
                        continue
                print(f"Starting run {seed} for p={num_levels}...")
                np.random.seed(seed)
                random.seed(seed)

                # generate random cstree and compute its kl divergence
                tree = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
                tree.sample_stage_parameters(alpha=1)

                data = tree.sample(samp_size)

                # make sure all columns are binary otherwise resample the data
                while not all([data[col][1:].nunique() >= 2 for col in data.columns]):
                    data = tree.sample(samp_size)

                tree_df = tree.to_joint_distribution(label_order=list(data.columns))
                tree_df.to_csv(f"{distr_path}/p={num_levels}_n={samp_size}_seed={seed}.csv", index=False)
                data.to_csv(f"{data_path}/p={num_levels}_n={samp_size}_seed={seed}.csv", index=False)


def estimate_pc_distr(data_path, est_path, seeds, samp_size_range, num_levels_range):

    """ Estimate the CStree for all datasets in data_path.
    """

    Path(est_path +"/est").mkdir(parents=True, exist_ok=True)
    Path(est_path +"/time").mkdir(parents=True, exist_ok=True)
    alpha = 0
    # get all data files
    for num_levels in num_levels_range:
        cards = [2] * num_levels
        for samp_size in samp_size_range:
            for seed in seeds:
                name = f"p={num_levels}_n={samp_size}_seed={seed}.csv"
                np.random.seed(seed)
                random.seed(seed)

                if Path(f"{est_path}/est/{name}/").is_file():
                    continue
                print(f"Estimating PC for {data_path}/est/{name}...")
                data = pd.read_csv(f"{data_path}/{name}")

                start = time.time()

                pcgraph = pc(data[1:].values, 0.05, "chisq", node_names=data.columns)

                edge_list = pcgraph.find_adj()
                adj_mat = np.zeros((num_levels, num_levels), bool)
                adj_mat[tuple(zip(*edge_list))] = True
                cpdag = cd.PDAG.from_amat(adj_mat)  # circumvent bug in pgmpy
                nxdag = cpdag.to_dag().to_nx()
                nxdag.add_nodes_from(range(num_levels))  # circumvent bug in cd
                
                #nxdag.add_nodes_from(data.columns)  # circumvent bug in cd
                # relabel nodes 
                p = len(data.columns)
                newlabs = {ind:lab for ind, lab in zip(range(p), data.columns)}
                                
                nxdag = nx.relabel_nodes(nxdag, newlabs)                
                pgm = BayesianNetwork(nxdag)
                pgm.fit(data[1:])
                
                # Write/tabulate joint distribution of pgm to file
                
                distr_df = pgmpy_to_joint_distribution(pgm, label_order=list(data.columns), cards=cards)
                #print(distr_df)

                #tree_df = tree.to_joint_distribution(label_order=list(data.columns))
                totaltime = time.time() - start
                #print(f"Time taken: {totaltime}")
                # save total time taken to dataframe with columns method, p, n_samples, seed, time
                time_df = pd.DataFrame(columns=["method", "p", "n_samples", "seed", "time"])
                time_df["method"] = ["pc"]
                time_df["p"] = [num_levels]
                time_df["n_samples"] = [samp_size]
                time_df["seed"] = [seed]
                time_df["time"] = [totaltime]
                time_df.to_csv(f"{est_path}/time/{name}", index=False)

                distr_df.to_csv(f"{est_path}/est/{name}", index=False)

def estimate_cstree_distr(data_path, est_path, seeds, samp_size_range, num_levels_range):

    """ Estimate the CStree for all datasets in data_path.
    """

    Path(est_path +"/est").mkdir(parents=True, exist_ok=True)
    Path(est_path +"/time").mkdir(parents=True, exist_ok=True)
    alpha = 0
    # get all data files
    for num_levels in num_levels_range:
        cards = [2] * num_levels
        for samp_size in samp_size_range:
            for seed in seeds:
                name = f"p={num_levels}_n={samp_size}_seed={seed}.csv"
                np.random.seed(seed)
                random.seed(seed)

                if Path(f"{est_path}/est/{name}/").is_file():
                    continue
                print(f"Estimating CStree for {data_path}/est/{name}...")
                data = pd.read_csv(f"{data_path}/{name}")

                start = time.time()

                pcgraph = pc(data[1:].values, 0.05, "chisq", node_names=data.columns)
                poss_cvars = ctl.causallearn_graph_to_posscvars(pcgraph, labels=data.columns)

                score_table, context_scores, context_counts = sc.order_score_tables(
                    data, max_cvars=2, alpha_tot=1, method="BDeu", poss_cvars=poss_cvars)

                orders, scores = ctl.gibbs_order_sampler(5000, score_table)
                maporder = orders[scores.index(max(scores))]

                #maporder, score = ctl._find_optimal_order(score_table)

                tree = ctl._optimal_cstree_given_order(maporder, context_scores)


                tree.estimate_stage_parameters(data, alpha_tot=alpha)

                tree_df = tree.to_joint_distribution(label_order=list(data.columns))
                totaltime = time.time() - start
                print(f"Time taken: {totaltime}")
                # save total time taken to dataframe with columns method, p, n_samples, seed, time
                time_df = pd.DataFrame(columns=["method", "p", "n_samples", "seed", "time"])
                time_df["method"] = ["cstree"]
                time_df["p"] = [num_levels]
                time_df["n_samples"] = [samp_size]
                time_df["seed"] = [seed]
                time_df["time"] = [totaltime]
                time_df.to_csv(f"{est_path}/time/{name}", index=False)

                tree_df.to_csv(f"{est_path}/est/{name}", index=False)


def kl_div_from_files(true_path, est_path, alg, seeds, samp_size_range, num_levels_range):
    """ Read in true and estimated distributions from files and compute KL divergence.
    """
    # get all distribution files
    # create empty pandas dataframe to store results
    # with info cstree, aplha, kl_divergence, p, n, seed

    df = pd.DataFrame(columns=["method", "kl_divergence", "p", "n_samples", "seed"])
    df_time = pd.DataFrame(columns=["method", "p", "n_samples", "seed", "time"])
    for num_levels in num_levels_range:
        cards = [2] * num_levels
        for samp_size in samp_size_range:
            kl_divs = []
            for seed in seeds:

                name = f"p={num_levels}_n={samp_size}_seed={seed}.csv"
                #print(f"Computing KL divergence for {name}...")
                true_dist_df = pd.read_csv(f"{true_path}/est/{name}")
                est_dist_df = pd.read_csv(f"{est_path}/{name}")

                kl_divs.append(KL_divergence(true_dist_df, est_dist_df))
                dftmp = pd.DataFrame(columns=["method", "kl_divergence", "p", "n_samples", "seed"])
                # add row to dataframe

                if alg=="stagedtree":
                    alg = "best order search"
                dftmp["method"] = [alg]
                dftmp["p"] = [num_levels]
                dftmp["n_samples"] = [samp_size]
                dftmp["seed"] = [seed]
                dftmp["kl_divergence"] = [KL_divergence(true_dist_df, est_dist_df)]
                df = pd.concat([df, dftmp])
                # concat time dataframes
                df_time_tmp = pd.read_csv(f"{true_path}/time/{name}")
                df_time = pd.concat([df_time, df_time_tmp])


            name = f"p={num_levels}_n={samp_size}"
            print(f"KL divergence for {name}:")
            #print(kl_divs)
            print(f"mean:{np.array(kl_divs).mean():.2f} median:{np.median(kl_divs):.2f} std:{np.array(kl_divs).std():.2f}")


    return df, df_time

if __name__ == "__main__":
    # check versions to ensure accurate reproduction
    if version("cstrees") != "1.2.0":
        warnings.warn(f"Current `cstrees` version unsupported.")
    sns.set_style("whitegrid")

    path = "sim_results"
    warnings.simplefilter(action="ignore", category=FutureWarning)

    samp_size_range = [100, 250, 500, 1000, 10000]
    seeds = list(range(10))
    num_levels_range = [5, 7, 10] #10, 15, 20 For all except orderseach

    print("Generating data and true distributions")
    generate_data_and_true_distr(path, seeds, samp_size_range, num_levels_range)


    print("Estimating CStree distributions")
    estimate_cstree_distr(f"{path}/data", f"{path}/distr/cstrees", seeds, samp_size_range, num_levels_range)
    
    print("Estimating PC distributions")
    estimate_pc_distr(f"{path}/data", f"{path}/distr/pc", seeds, samp_size_range, num_levels_range)

    # Estimate the KL divergence between the true and estimated distributions

    print("Estimated cstree KL")
    df_kl_cstree, df_time_cstree = kl_div_from_files(f"{path}/distr/cstrees/", f"{path}/distr/true", "cstree", seeds, samp_size_range, num_levels_range)

    print("Estimated PC KL")
    df_kl_pc, df_time_pc = kl_div_from_files(f"{path}/distr/pc/", f"{path}/distr/true", "pc", seeds, samp_size_range, num_levels_range)    
    # time pc
    print("Estimated PC time")
    print(df_time_pc)
    
    # print("Staged trees KL")
    # df_kl_st, df_time_st = kl_div_from_files(f"{path}/distr/stagedtrees/", f"{path}/distr/true","stagedtree", seeds, samp_size_range, num_levels_range)
    
    # print("PC + staged trees KL")
    # df_kl_pc, df_time_pc_st = kl_div_from_files(f"{path}/distr/pc_st/", f"{path}/distr/true","pc_st", seeds, samp_size_range, num_levels_range)

    df_kl = pd.concat([df_kl_pc, df_kl_cstree])
    df_kl[["p", "n_samples", "seed"]] = df_kl[["p", "n_samples", "seed"]].apply(pd.to_numeric)
    print("KL divergence results:")
    print(df_kl)
   
    for p in num_levels_range:
        
        sns_plot = sns.boxplot(data=df_kl[df_kl["p"]==p], x="n_samples", y="kl_divergence", hue="method")
        # add seeds and p to title
        sns_plot.set_title(f"KL divergence from true CStree for p={p} based on {len(seeds)} seeds")
        #sns_plot.facet("p")
        fig = sns_plot.get_figure()
        fig.savefig(f"kl_p={p}.png")
        plt.clf()

    df_time = pd.concat([df_time_pc, df_time_cstree])
    df_time[["p", "n_samples", "seed"]] = df_time[["p", "n_samples", "seed"]].apply(pd.to_numeric)
    print("Time taken results:")
    # print summary of datadframe
    
    # plot time taken for each method
    for p in num_levels_range:
        #print(df_time[df_time["p"]==p])
        sns_plot = sns.boxplot(data=df_time[df_time["p"]==p], x="n_samples", y="time", hue="method")
        sns_plot.set_title(f"Time taken for p={str(p)} based on {str(len(seeds))} seeds")
        fig = sns_plot.get_figure()
        fig.savefig(f"time_p={p}.png")
        plt.clf()
        
    # Plot 1: 
    # KL-divergence on y-axis
    # x-axis: number of nodes (p = 5,7,10,15,20)
    # algorithms:
    #  - PC-estimated DAG MLE for n = 10000
    #  - PC + CStrees with n = 250
    #  - PC + CStrees with n = 500
    #  - PC + CStrees with n = 1000
    #  - PC + CStrees with n = 10000
    #  - bos (stagedtrees) with n = 1000
    #  - PC + bhc (stagedtrees) with n = 1000
    #  - PC + bhc (stagedtrees) with n = 10000

    # * notes:
    #  - best order search is only done for p = 5,7,10
    #  - PC + bhc is done for p = 5,7,10,15,20
    #  - bhc = backwards hill climbing
    #  - bos = best order search
    #  - come up with a name to replace "CStrees" with
    print("Plotting KL divergence")
    
    df_kl_plot1 = df_kl
    df_kl_plot1["method_n"] = df_kl_plot1["method"] + " for n=" + df_kl_plot1["n_samples"].astype(str)
    
    print(df_kl_plot1)
    
    sns_plot = sns.boxplot(data=df_kl_plot1, x="p", y="kl_divergence", hue="method_n")
    sns_plot.set_title(f"KL divergence from true CStree based on {len(seeds)} seeds")
    fig = sns_plot.get_figure()
    fig.savefig(f"plot_1_kl.png")
    plt.clf()

