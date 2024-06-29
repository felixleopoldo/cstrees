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
    # df_outcomes = pd.DataFrame(columns=self.labels)
    df_outcomes = pd.DataFrame(columns=label_order)
    # store all the outcomes and probabilities
    pmfs = [None] * np.prod(cards)
    pmfs_log = [None] * np.prod(cards)
    for i, outcome in enumerate(outcomes):
        df_outcomes.loc[i] = outcome

        pgmpy_outcome = {
            lab: marg_outcome for lab, marg_outcome in zip(label_order, outcome)
        }
        prob = model.get_state_probability(pgmpy_outcome)

        pmfs[i] = prob
        pmfs_log[i] = np.log(prob)

    df_pmf = pd.DataFrame(pmfs, columns=["prob"])
    df_pmf_log = pd.DataFrame(pmfs_log, columns=["log_prob"])
    # join the two dataframes
    df = pd.concat([df_outcomes, df_pmf, df_pmf_log], axis=1)

    return df


def generate_data_and_true_distr(path, seeds, samp_size_range, num_levels_range):
    """Generate data and true distributions for all combinations of parameters."""

    data_path = f"{path}/data"
    distr_path = f"{path}/distr/true"
    Path(data_path).mkdir(parents=True, exist_ok=True)
    Path(distr_path).mkdir(parents=True, exist_ok=True)

    for num_levels in num_levels_range:
        cards = [2] * num_levels
        for samp_size in samp_size_range:
            for seed in seeds:

                if Path(
                    f"{data_path}/p={num_levels}_n={samp_size}_seed={seed}.csv"
                ).is_file():
                    data = pd.read_csv(
                        f"{data_path}/p={num_levels}_n={samp_size}_seed={seed}.csv"
                    )
                    if all([data[col][1:].nunique() >= 2 for col in data.columns]):
                        continue
                print(f"Starting run {seed} for p={num_levels}...")
                np.random.seed(seed)
                random.seed(seed)

                # generate random cstree and compute its kl divergence
                tree = ct.sample_cstree(
                    cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1
                )
                tree.sample_stage_parameters(alpha=1)

                data = tree.sample(samp_size)

                # make sure all columns are binary otherwise resample the data
                while not all([data[col][1:].nunique() >= 2 for col in data.columns]):
                    data = tree.sample(samp_size)

                tree_df = tree.to_joint_distribution(label_order=list(data.columns))
                tree_df.to_csv(
                    f"{distr_path}/p={num_levels}_n={samp_size}_seed={seed}.csv",
                    index=False,
                )
                data.to_csv(
                    f"{data_path}/p={num_levels}_n={samp_size}_seed={seed}.csv",
                    index=False,
                )


def estimate_pc_distr(data_path, est_path, seeds, samp_size_range, num_levels_range):
    """Estimate the CStree for all datasets in data_path."""

    Path(est_path + "/est").mkdir(parents=True, exist_ok=True)
    Path(est_path + "/time").mkdir(parents=True, exist_ok=True)
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

                # nxdag.add_nodes_from(data.columns)  # circumvent bug in cd
                # relabel nodes
                p = len(data.columns)
                newlabs = {ind: lab for ind, lab in zip(range(p), data.columns)}

                nxdag = nx.relabel_nodes(nxdag, newlabs)
                pgm = BayesianNetwork(nxdag)
                pgm.fit(data[1:])

                # Write/tabulate joint distribution of pgm to file

                distr_df = pgmpy_to_joint_distribution(
                    pgm, label_order=list(data.columns), cards=cards
                )
                # print(distr_df)

                # tree_df = tree.to_joint_distribution(label_order=list(data.columns))
                totaltime = time.time() - start
                # print(f"Time taken: {totaltime}")
                # save total time taken to dataframe with columns method, p, n_samples, seed, time
                time_df = pd.DataFrame(
                    columns=["method", "p", "n_samples", "seed", "time"]
                )
                time_df["method"] = ["pc"]
                time_df["p"] = [num_levels]
                time_df["n_samples"] = [samp_size]
                time_df["seed"] = [seed]
                time_df["time"] = [totaltime]
                time_df.to_csv(f"{est_path}/time/{name}", index=False)

                distr_df.to_csv(f"{est_path}/est/{name}", index=False)


def estimate_cstree_distr(
    method, data_path, est_path, seeds, samp_size_range, num_levels_range
):
    """Estimate the CStree for all datasets in data_path."""

    Path(est_path + "/est").mkdir(parents=True, exist_ok=True)
    Path(est_path + "/time").mkdir(parents=True, exist_ok=True)
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

                # try grasp first
                poss_cvars = None
                print(f"Method: {method}")
                if method == "grasp_cslearn":
                    print("Using GRaSP")
                    graspgraph = grasp(data[1:].values)
                    print("GRaSP done")
                    print(graspgraph)
                    poss_cvars = ctl.causallearn_graph_to_posscvars(
                        graspgraph, labels=data.columns, alg="grasp"
                    )
                elif method == "pc_cslearn":
                    pcgraph = pc(
                        data[1:].values, 0.05, "chisq", node_names=data.columns
                    )
                    poss_cvars = ctl.causallearn_graph_to_posscvars(
                        pcgraph, labels=data.columns, alg="pc"
                    )
                else:
                    raise ValueError("Method not recognized.")

                score_table, context_scores, context_counts = sc.order_score_tables(
                    data, max_cvars=2, alpha_tot=1, method="BDeu", poss_cvars=poss_cvars
                )

                orders, scores = ctl.gibbs_order_sampler(5000, score_table)
                maporder = orders[scores.index(max(scores))]

                # maporder, score = ctl._find_optimal_order(score_table)

                tree = ctl._optimal_cstree_given_order(maporder, context_scores)

                tree.estimate_stage_parameters(data, alpha_tot=alpha)

                tree_df = tree.to_joint_distribution(label_order=list(data.columns))
                totaltime = time.time() - start
                print(f"Time taken: {totaltime}")
                # save total time taken to dataframe with columns method, p, n_samples, seed, time
                time_df = pd.DataFrame(
                    columns=["method", "p", "n_samples", "seed", "time"]
                )
                time_df["method"] = [method]
                time_df["p"] = [num_levels]
                time_df["n_samples"] = [samp_size]
                time_df["seed"] = [seed]
                time_df["time"] = [totaltime]
                time_df.to_csv(f"{est_path}/time/{name}", index=False)

                tree_df.to_csv(f"{est_path}/est/{name}", index=False)


def kl_div_from_files(
    true_path, est_path, alg, seeds, samp_size_range, num_levels_range
):
    """Read in true and estimated distributions from files and compute KL divergence."""
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
                # print(f"Computing KL divergence for {name}...")
                true_dist_df = pd.read_csv(f"{true_path}/est/{name}")
                est_dist_df = pd.read_csv(f"{est_path}/{name}")

                kl_divs.append(KL_divergence(true_dist_df, est_dist_df))
                dftmp = pd.DataFrame(
                    columns=["method", "kl_divergence", "p", "n_samples", "seed"]
                )
                # add row to dataframe

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
            # print(kl_divs)
            print(
                f"mean:{np.array(kl_divs).mean():.2f} median:{np.median(kl_divs):.2f} std:{np.array(kl_divs).std():.2f}"
            )

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
    num_levels_range = [5, 7, 10]  # 15, 20

    ######## Generate data and true distributions ##########

    print("Generating data and true distributions")
    generate_data_and_true_distr(path, seeds, samp_size_range, num_levels_range)

    ######## Estimate the distributions ##########

    print("Estimating PC + CStree distributions")
    estimate_cstree_distr(
        "pc_cslearn",
        f"{path}/data",
        f"{path}/distr/pc_cslearn",
        seeds,
        samp_size_range,
        num_levels_range,
    )

    print("Estimating GRaSP + CStree distributions")
    estimate_cstree_distr(
        "grasp_cslearn",
        f"{path}/data",
        f"{path}/distr/grasp_cslearn",
        seeds,
        samp_size_range,
        num_levels_range,
    )

    print("Estimating PC distributions")
    estimate_pc_distr(
        f"{path}/data", f"{path}/distr/pc", seeds, samp_size_range, num_levels_range
    )

    ######### KL divergence ##########

    print("Estimated PC + CSlearn KL")
    df_kl_cstree, df_time_cstree = kl_div_from_files(
        f"{path}/distr/pc_cslearn/",
        f"{path}/distr/true",
        "pc_cslearn",
        seeds,
        samp_size_range,
        num_levels_range,
    )

    print("Estimated GRaSP + CSlearn KL")
    df_kl_grasp_cslearn, df_time_grasp_cslearn = kl_div_from_files(
        f"{path}/distr/grasp_cslearn/",
        f"{path}/distr/true",
        "grasp_cslearn",
        seeds,
        samp_size_range,
        num_levels_range,
    )

    print("Estimated PC KL")
    df_kl_pc, df_time_pc = kl_div_from_files(
        f"{path}/distr/pc/",
        f"{path}/distr/true",
        "pc",
        seeds,
        [10000],
        num_levels_range,
    )

    print("Staged trees BOS KL")
    df_kl_bos, df_time_bos = kl_div_from_files(
        f"{path}/distr/bos/",
        f"{path}/distr/true",
        "bos",
        seeds,
        [1000],
        num_levels_range,
    )

    print("PC + staged trees KL")
    df_kl_pc_bhc, df_time_pc_bhc = kl_div_from_files(
        f"{path}/distr/pc_bhc",
        f"{path}/distr/true",
        "pc_bhc",
        seeds,
        samp_size_range,
        num_levels_range,
    )

    ######### Join KL results ##########

    df_kl = pd.concat(
        [df_kl_pc, df_kl_cstree, df_kl_pc_bhc, df_kl_bos, df_kl_grasp_cslearn]
    )
    df_kl[["p", "n_samples", "seed"]] = df_kl[["p", "n_samples", "seed"]].apply(
        pd.to_numeric
    )

    # relabel methods
    df_kl["method"] = df_kl["method"].replace(
        {
            "grasp_cslearn": "GRaSP + CSlearn",
            "pc_cslearn": "PC+CSlearn",
            "pc": "PC",
            "pc_bhc": "PC + BHC",
            "bos": "BOS",
        }
    )

    print("KL divergence results:")
    print(df_kl)

    ####### Join time results #######

    df_time = pd.concat(
        [df_time_pc, df_time_cstree, df_time_pc_bhc, df_time_bos, df_time_grasp_cslearn]
    )
    df_time[["p", "n_samples", "seed"]] = df_time[["p", "n_samples", "seed"]].apply(
        pd.to_numeric
    )
    # relabel methods
    df_time["method"] = df_time["method"].replace(
        {"cslearn": "PC+CSlearn", "pc": "PC", "pc_bhc": "PC + BHC", "bos": "BOS"}
    )

    print("Plotting KL divergence")

    ############# Figure 2 in paper (Three plots (2a), (2b) and (2c)): ###############

    ####### Plot (2a): comparison of performance of parameter MAP estimate versus parameter MLE est with PC used in Phase 1:

    # KL-divergence on y-axis
    # x-axis: number of nodes (p = 5,7,10,15,20)
    # algorithms:

    #  - PC-estimated DAG (MLE) for n=10000

    #  - CSlearn with PC (MLE) for n=250
    #  - CSlearn with PC (MLE) for n=500
    #  - CSlearn with PC (MLE) for n=1000
    #  - CSlearn with PC (MLE) for n=10000

    #  - CSlearn with PC (MAP) for n=250
    #  - CSlearn with PC (MAP) for n=500
    #  - CSlearn with PC (MAP) for n=1000
    #  - CSlearn with PC (MAP) for n=10000

    # Create a column of the form "method for n=10000" for each method, to be used as hue in the plot
    df_kl_plot_2a = df_kl
    df_kl_plot_2a["method_n"] = (
        df_kl_plot_2a["method"]
        + "("
        + df_kl_plot_2a["estimator"]
        + ") for n="
        + df_kl_plot_2a["n_samples"].astype(str)
    )
    # now, select the rows according to the spec above
    df_kl_plot_2a = df_kl_plot_2a[
        (
            df_kl_plot_2a["method"] == "pc"
            and df_kl_plot_2a["estimator"] == "mle"
            and df_kl_plot_2a["n_samples"] == 10000
        )
        or (
            df_kl_plot_2a["method"] == "pc_cslearn"
            and df_kl_plot_2a["estimator"] in ["map", "mle"]
            and (df_kl_plot_2a["n_samples"] in [250, 500, 1000, 10000])
        )
    ]

    print(df_kl_plot_2a)

    sns_plot = sns.boxplot(data=df_kl_plot_2a, x="p", y="kl_divergence", hue="method_n")
    sns_plot.set_title(f"KL divergence from true CStree based on {len(seeds)} seeds")
    fig = sns_plot.get_figure()
    fig.savefig(f"fig_2a_kl.png")
    plt.clf()

    ########### Plot (2b): comparison of performance of parameter MAP estimate versus parameter MLE est with GrASP used in Phase 1:

    # KL-divergence on y-axis
    # x-axis: number of nodes (p = 5,7,10,15,20)
    # algorithms:
    #  - GrASP-estimated DAG (MLE) for n=10000
    #  - CSlearn with GrASP (MAP) for n=250
    #  - CSlearn with GrASP (MAP) for n=500
    #  - CSlearn with GrASP (MAP) for n=1000
    #  - CSlearn with GrASP (MAP) for n=10000
    #  - CSlearn with GrASP (MLE) for n=250
    #  - CSlearn with GrASP (MLE) for n=500
    #  - CSlearn with GrASP (MLE) for n=1000
    #  - CSlearn with GrASP (MLE) for n=10000

    # ***The likely outcome is that GrASP with MLE is the best performer out of the above results.  In any case the best performer should be used in the following plot***

    # Create a column of the form "method for n=10000" for each method, to be used as hue in the plot
    df_kl_plot_2b = df_kl
    df_kl_plot_2b["method_n"] = (
        df_kl_plot_2b["method"]
        + "("
        + df_kl_plot_2b["estimator"]
        + ") for n="
        + df_kl_plot_2b["n_samples"].astype(str)
    )
    # now, select the rows according to the spec above
    df_kl_plot_2b = df_kl_plot_2b[
        (
            df_kl_plot_2b["method"] == "grasp"
            and df_kl_plot_2b["estimator"] == "mle"
            and df_kl_plot_2b["n_samples"] == 10000
        )
        or (
            df_kl_plot_2b["method"] == "graph_cslearn"
            and df_kl_plot_2b["estimator"] in ["map", "mle"]
            and (df_kl_plot_2b["n_samples"] in [250, 500, 1000, 10000])
        )
    ]

    print(df_kl_plot_2b)

    sns_plot = sns.boxplot(data=df_kl_plot_2b, x="p", y="kl_divergence", hue="method_n")
    sns_plot.set_title(f"KL divergence from true CStree based on {len(seeds)} seeds")
    fig = sns_plot.get_figure()
    fig.savefig(f"fig_2b_kl.png")
    plt.clf()

    # Plot (2c): Comparison of CSlearn with other methods:

    # KL-divergence on y-axis
    # x-axis: number of nodes (p = 5,7,10,15,20)
    # algorithms:
    #  - GrASP-estimated DAG (MLE) for n = 10000
    #  - CSlearn with GrASP for n = 1000
    #  - CSlearn with GrASP for n = 10000
    #  - bos (stagedtrees) with n = 1000
    #  - GrASP + bhc (stagedtrees) with n = 1000
    #  - GrASP + bhc (stagedtrees) with n = 10000
    # * notes:
    #  - best order search is only done for p = 5,7,10
    #  - GrASP + bhc is done for p = 5,7,10,15,20
    #  - bhc = backwards hill climbing
    #  - bos = best order search

    # Create a column of the form "method for n=10000" for each method, to be used as hue in the plot
    df_kl_plot_2c = df_kl
    df_kl_plot_2c["method_n"] = (
        df_kl_plot_2c["method"]
        + "("
        + df_kl_plot_2c["estimator"]
        + ") for n="
        + df_kl_plot_2c["n_samples"].astype(str)
    )
    # now, select the rows according to the spec above
    df_kl_plot_2c = df_kl_plot_2c[
        (
            df_kl_plot_2c["method"] == "grasp"
            and df_kl_plot_2c["estimator"] == "mle"
            and df_kl_plot_2c["n_samples"] == 10000
        )
        or (
            df_kl_plot_2c["method"] == "grasp_cslearn"
            and df_kl_plot_2c["estimator"] in ["mle"]
            and (df_kl_plot_2c["n_samples"] in [1000, 10000])
        )
        or (
            df_kl_plot_2c["method"] == "bos"
            and df_kl_plot_2c["estimator"] in ["mle"]
            and (df_kl_plot_2c["n_samples"] in [1000, 10000])
            and (df_kl_plot_2c["p"] in [5, 7, 10])
        )
        or (
            df_kl_plot_2c["method"] == "grasp_bhc"
            and df_kl_plot_2c["estimator"] in ["mle"]
            and (df_kl_plot_2c["n_samples"] in [1000, 10000])
        )
    ]

    print(df_kl_plot_2c)

    sns_plot = sns.boxplot(data=df_kl_plot_2c, x="p", y="kl_divergence", hue="method_n")
    sns_plot.set_title(f"KL divergence from true CStree based on {len(seeds)} seeds")
    fig = sns_plot.get_figure()
    fig.savefig(f"fig_2c_kl.png")
    plt.clf()


    ############# Time taken results ###############
    print("Time taken results:")
    
    # Figure 3 in paper (Three plots (3a) and (3b)):

    # Plot (3a): comparison of runtimes (total times only) for all algorithms on small graphs with multiple sample sizes:

    # Total Runtime on y-axis
    # x-axis: number of nodes (p = 5,7,10,15,20)
    # algorithms:
    #  - CSlearn with PC for n = 250
    #  - CSlearn with PC for n = 1000
    #  - CSlearn with GrASP for n = 250
    #  - CSlearn with GrASP for n = 1000
    #  - bos (stagedtrees) with n = 250
    #  - bos (stagedtrees) with n = 1000
    #  - GrASP + bhc (stagedtrees) with n = 250
    #  - GrASP + bhc (stagedtrees) with n = 1000    
    
   # Create a column of the form "method for n=10000" for each method, to be used as hue in the plot
    df_time_plot_3a = df_time
    df_time_plot_3a["method_n"] = (
        df_time_plot_3a["method"]
        + "("
        + df_time_plot_3a["estimator"]
        + ") for n="
        + df_time_plot_3a["n_samples"].astype(str)
    )
    # now, select the rows according to the spec above
    df_time_plot_3a = df_time_plot_3a[
        (
            df_time_plot_3a["method"] == "pc_cslearn"
            and df_time_plot_3a["estimator"] == "mle"
            and df_time_plot_3a["n_samples"] in [250, 1000]
        )
        or (
            df_time_plot_3a["method"] == "grasp_cslearn"
            and df_time_plot_3a["estimator"] in ["mle"]
            and (df_time_plot_3a["n_samples"] in [250, 1000])
        )
        or (
            df_time_plot_3a["method"] == "bos"
            and df_time_plot_3a["estimator"] in ["mle"]
            and (df_time_plot_3a["n_samples"] in [250, 1000])
        )
        or (
            df_time_plot_3a["method"] == "grasp_bhc"
            and df_time_plot_3a["estimator"] in ["mle"]
            and (df_time_plot_3a["n_samples"] in [250, 1000])
        )
    ]

    print(df_time_plot_3a)

    sns_plot = sns.boxplot(data=df_time_plot_3a, x="p", y="Time (s)", hue="method_n")
    sns_plot.set_title(f"Timings based on {len(seeds)} seeds")
    fig = sns_plot.get_figure()
    fig.savefig(f"fig_3a_kl.png")
    plt.clf()    




    # Plot (3b): comparison of performance of scalable methods (total times only... I don't think there is much benefit to including the phase breakdown... text me if you want to discuss it):

    # Total Runtime on y-axis
    # x-axis: number of nodes (p = 10, 25, 50, 100, 250, 500)
    # algorithms:
    #  - CSlearn with PC
    #  - CSlearn with GrASP
    #  - GrASP + bhc (stagedtrees)
    # *notes:
    #  - number of samples n = 1000
    # cslearn_plots_2024_06_14.txt
    # Displaying cslearn_plots_2024_06_14.txt.

    df_time_plot_3b = df_time
    df_time_plot_3b["method_n"] = (
        df_time_plot_3b["method"]
        + "("
        + df_time_plot_3b["estimator"]
        + ") for n="
        + df_time_plot_3b["n_samples"].astype(str)
    )
    # now, select the rows according to the spec above
    df_time_plot_3b = df_time_plot_3b[
        (
            df_time_plot_3b["method"] == "pc_cslearn"
            and df_time_plot_3b["estimator"] == "mle"
            and df_time_plot_3b["n_samples"] in [250, 1000]
        )
        or (
            df_time_plot_3b["method"] == "grasp_cslearn"
            and df_time_plot_3b["estimator"] in ["mle"]
            and (df_time_plot_3b["n_samples"] in [250, 1000])
        )
        or (
            df_time_plot_3b["method"] == "bos"
            and df_time_plot_3b["estimator"] in ["mle"]
            and (df_time_plot_3b["n_samples"] in [250, 1000])
        )
        or (
            df_time_plot_3b["method"] == "grasp_bhc"
            and df_time_plot_3b["estimator"] in ["mle"]
            and (df_time_plot_3b["n_samples"] in [250, 1000])
        )
    ]

    print(df_time_plot_3b)

    sns_plot = sns.boxplot(data=df_time_plot_3b, x="p", y="Time (s)", hue="method_n")
    sns_plot.set_title(f"Timings based on {len(seeds)} seeds")
    fig = sns_plot.get_figure()
    fig.savefig(f"fig_3b_kl.png")
    plt.clf()    

