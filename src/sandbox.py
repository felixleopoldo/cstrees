
#%load_ext autoreload
#%autoreload 2
import os
import cstrees.cstree as ct
import numpy as np
import networkx as nx
import random
import pandas as pd
import time
import logging, sys
import pathlib
from mpl_toolkits.mplot3d import Axes3D

#seeds = [1,2,3]
seeds = [1, 2, 3]
ps = range(5, 12, 1)
#ps = [5, 6, 7, 8, ]
#ps = [7]
maxs = [3, 4] 
#maxs = range(1, 4)

prob_cvars=[0.5]
prop_nonsingletons=[0.5]

df = pd.DataFrame(columns=["p", "max_cvars", "time"])

for seed in seeds:
    for prob_cvar in prob_cvars:
        for prop_nonsingleton in prop_nonsingletons:
            for m in maxs:
                print(m)
                for p in ps:
                    np.random.seed(seed)#9
                    random.seed(seed)

                    sample_times = []
                    to_mindag_times = []
                    print(p)

                    start = time.perf_counter()
                    print("Sample tree")
                    cards = [2]*p
                    t = ct.sample_cstree(cards, m, prob_cvar, prop_nonsingleton)
                    t.set_random_stage_parameters()
                    print("df {}".format(t.to_df()))
                    stop = time.perf_counter()
                    sample_times.append(stop - start)
                    
                    start = time.perf_counter()
                    print("Minl dags")
                    cdags = t.to_minimal_context_graphs()
                    stop = time.perf_counter()
                    
                    
                    for key, graph in cdags.items():
                        agraph = nx.nx_agraph.to_agraph(graph)
                        agraph.layout("dot")
                        d = "src/figures/seed_{}_m_{}_p_{}/".format(seed,m,p)
                        pathlib.Path(d).mkdir(exist_ok=True)
                        agraph.draw("{}/{}_csi.png".format(d,key), args='-Glabel="'+str(key)+'"   ')

                    tmp = pd.DataFrame({"seed": [seed], 
                                        "p": [p], 
                                        "max_cvars": [m], 
                                        "cvar_prob": [prob_cvar], 
                                        "prop_colored": [prop_nonsingleton], 
                                        "time": [stop-start]})
                    
                    df = pd.concat([df, tmp]).reset_index(drop=True)      
                    to_mindag_times.append(stop - start)
                    
                    print(stop-start)
                
        
print(df)
df.to_csv("src/timings.csv", index=False)