# %load_ext autoreload
# %autoreload 2

import cstrees.cstree as ct
import numpy as np

# CStree from Figure 1 in (Duarte & Solus, 2022)
np.random.seed(1)
p = 4
co = range(1, p+1)
tree = ct.CStree(co)
tree.set_cardinalities([None] + [2] * p)
tree.add_stages({
    0: [],
    1: [],
    2: [{(0, 0), (1, 0)}],  # green
    3: [{(0, 0, 0), (0, 1, 0)},  # blue
        {(0, 0, 1), (0, 1, 1)},  # orange
        {(1, 0, 0), (1, 1, 0)}]  # red
})

tree.create_tree()
tree.set_random_parameters()
tree.plot()
x = tree.sample(5)
print
csi_rels = tree.csi_relations()

for key, val in csi_rels.items():
    print("{}".format(val))
dags = tree.to_minimal_context_graphs()

#nodes = [("X"+str(i), "X"+str(i+1)) for i in range(1, p)]
# tree.tree.add_edges_from(nodes)
# tree.tree.add_edge("Ø","X1")
