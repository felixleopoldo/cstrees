# %load_ext autoreload
# %autoreload 2

import cstrees.cstree as ct
import numpy as np
import networkx as nx
# CStree from Figure 1 in (Duarte & Solus, 2022)
#np.random.seed(1)
p = 5

co = ct.CausalOrder(range(1, p+1))
tree = ct.CStree(co)
cards = [2] * p
stage = ct.sample_random_stage(cards[:4])

tree.set_cardinalities([None] + cards)


# These do not have to be in a dict like this as the levels are
# determined from the length of the tuples.

ct.Stage([None, 0])
tree.add_stages({
    0: [],
    1: [],
    2: [ct.Stage([[0, 1], 0])],    # Green
    3: [ct.Stage([0, [0, 1], 0]),  # Blue
        ct.Stage([0, [0, 1], 1]),  # Orange
        ct.Stage([1, [0, 1], 0])],  # Red
    4: [stage]
})

# Context + level + order == stage

tree.create_tree()
tree.set_random_parameters()
tree.plot()
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


# The strategy is to first generate some random stages.
# Then merge them randomly into new stages in some way.
# Maybe pick two and merge.

# Note that, one CSI relation can have several stages, as in the
# paper exapmle. Maybe its more complex than that. We might have to
# absorb CSI relations and maybe other things as well. Lets see.
