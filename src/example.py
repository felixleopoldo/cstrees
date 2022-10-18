# %load_ext autoreload
# %autoreload 2

import cstrees.cstree as ct
import numpy as np
import networkx as nx
# CStree from Figure 1 in (Duarte & Solus, 2022)
np.random.seed(1)
p = 10
co = range(1, p+1)
tree = ct.CStree(co)
tree.set_cardinalities([None] + [2] * p)
# These do not have to be in a dict like this as the levels are
# determined from the length of the tuples.
Tree.Add_Stages({
    0: [],
    1: [],
    2: [{(0, 0), (1, 0)}],       # Green
    3: [{(0, 0, 0), (0, 1, 0)},  # Blue
        {(0, 0, 1), (0, 1, 1)},  # Orange
        {(1, 0, 0), (1, 1, 0)}]  # Red
})
Tree.Add_Stages({
    0: [],
    1: [],
    2: [ct.Stage([None, 0])],    # Green
    3: [ct.Stage([0, None, 0]),  # Blue
        ct.Stage([0, None, 1]),  # Orange
        ct.Stage([1, None, 0])]  # Red
})


tree.create_tree()
tree.set_random_parameters()
# tree.plot()
x = tree.sample(5)
# print(x)
rels = tree.csi_relations()
# print(rels)
for key, val in rels.items():
    print(val)

adjmats = ct.csi_relations_to_dags(rels, co)

for key, graph in adjmats.items():
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout("dot")
    agraph.draw(str(key) + ".png")

    # print(graph.nodes)
    # print(graph.edges())


#dags = tree.to_minimal_context_graphs()

#nodes = [("X"+str(i), "X"+str(i+1)) for i in range(1, p)]
# tree.tree.add_edges_from(nodes)
# tree.tree.add_edge("Ø","X1")
