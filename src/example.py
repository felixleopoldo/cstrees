# %load_ext autoreload
# %autoreload 2

import cstrees.cstree as ct
import numpy as np
import networkx as nx
import itertools
# CStree from Figure 1 in (Duarte & Solus, 2022)
np.random.seed(1)

p = 4
cards = [2] * p



t = ct.sample_cstree(cards, 2, 0.5, 1)
#t.plot()
t.set_random_stage_parameters()

x = t.sample(5)
t.plot()
rels = t.csi_relations()

print("level and csis")
for key, val in rels.items():
    print(val)

x = t.sample(5)
print(x)
print()
