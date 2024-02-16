# %load_ext autoreload
# %autoreload 2
import os
import cstrees.cstree as st
import numpy as np
import networkx as nx
import random
import pandas as pd
import time
import logging
import sys
import pathlib
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


seeds = range(5)
ps = range(5, 15, 1)
# ps = [5, 6, 7, 8, ]
# ps = [7]
maxs = [3, 4]
# maxs = range(1, 4)
plot = False

prob_cvars = [1]
prop_nonsingletons = [0.5]

df = pd.DataFrame(columns=["p", "max_cvars", "time", "seed"])
folder = pathlib.Path("csvs")
for seed in seeds:
    for prob_cvar in prob_cvars:
        for prop_nonsingleton in prop_nonsingletons:
            for m in maxs:
                print(m)
                for p in ps:
                    filename = pathlib.Path(
                        "seed~{}_p~{}_max_cvars~{}_cvar_prob~{}_prop_colored~{}.csv".format(
                            seed, p, m, prob_cvar, prop_nonsingleton
                        )
                    )
                    if (folder / filename).is_file():
                        tmp = pd.read_csv(folder / filename)

                    else:
                        np.random.seed(seed)
                        random.seed(seed)

                        sample_times = []
                        to_mindag_times = []
                        print(p)

                        start = time.perf_counter()
                        print("Sample tree")
                        cards = [2] * p
                        t = st.sample_cstree(cards, m, prob_cvar, prop_nonsingleton)
                        t.sample_stage_parameters()
                        print("df {}".format(t.to_df()))
                        stop = time.perf_counter()
                        # save the tree as well
                        tdf = t.to_df().to_csv(Path("cstrees") / filename, index=False)
                        sample_times.append(stop - start)

                        start = time.perf_counter()
                        print("Minl dags")
                        cdags = t.to_minimal_context_graphs()
                        stop = time.perf_counter()
                        print(stop - start)
                        if plot:
                            print("Plotting CSDags")
                            for key, graph in cdags.items():
                                agraph = nx.nx_agraph.to_agraph(graph)
                                agraph.layout("dot")
                                d = "src/figures/{}/".format(filename.stem)
                                pathlib.Path(d).mkdir(exist_ok=True)
                                agraph.draw(
                                    "{}/{}_csi.png".format(d, key),
                                    args='-Glabel="' + str(key) + '"   ',
                                )

                        tmp = pd.DataFrame(
                            {
                                "seed": [seed],
                                "p": [p],
                                "max_cvars": [m],
                                "cvar_prob": [prob_cvar],
                                "prop_colored": [prop_nonsingleton],
                                "time": [stop - start],
                            }
                        )

                        tmp.to_csv(folder / filename, index=False)

                    df = pd.concat([df, tmp]).reset_index(drop=True)
                    df.to_csv("src/timings.csv", index=False)


print(df)
