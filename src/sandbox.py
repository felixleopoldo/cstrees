
#%load_ext autoreload
#%autoreload 2

import cstrees.cstree as ct
import numpy as np
import networkx as nx
import random
import pandas as pd
import time
import logging, sys

seed = 1
np.random.seed(seed)#9
random.seed(seed)
ps = range(10, 20, 1)
#ps = [15]
maxs = [4] #range(1, 4)
#maxs = range(1, 5)

prob_cvar=0.9
prop_nonsingleton=0.5

df = pd.DataFrame(columns=["p", "contvars", "time"])

for m in maxs:
    print(m)
    for p in ps:
        sample_times = []
        to_mindag_times = []
        print(p)
        start = time.perf_counter()
        print("Sample tree")
        cards = [2]*p
        t = ct.sample_cstree(cards, m, prob_cvar, prop_nonsingleton)
        t.set_random_stage_parameters()
        stop = time.perf_counter()
        sample_times.append(stop - start)
        start = time.perf_counter()
        print("Minl dags")
        cdags = t.to_minimal_context_graphs()
        stop = time.perf_counter()
        
        tmp = pd.DataFrame({"p": [p], "contvars": [m], "cvar_prob": [prob_cvar], "prop_colored": [prop_nonsingleton], "time": [stop-start]})
        
        df = pd.concat([df, tmp]).reset_index(drop=True)      
        to_mindag_times.append(stop - start)
        
        print(stop-start)
        
        
print(df)     