import cstrees.cstree as st
import numpy as np
import networkx as nx
import logging, sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# CStree from Figure 1 in (Duarte & Solus, 2022)



np.random.seed(2)
levelplus1=4
cards = [2] * levelplus1
co = st.CausalOrder(range(levelplus1))
tree = st.CStree(co)


#stage = ct.sample_random_stage(cards,2)
#stage.set_random_params(cards)

tree.set_cardinalities(cards)

# These do not have to be in a dict like this as the levels are
# determined from the length of the tuples.

tree.update_stages({
    0: [],
    1: [st.Stage([{0, 1}, 0])],    # Green
    2: [st.Stage([0, {0, 1}, 0]),  # Blue
        st.Stage([0, {0, 1}, 1]),  # Orange
        st.Stage([1, {0, 1}, 0])]  # Red
})


tree.sample_stage_parameters()
#x = tree.sample(5)

a = tree.plot()
a.draw("testplot.png")

x = tree.sample(5)
#print(x)
#rels = tree.csi_relations()

#print("Initial rels")
#print(rels)
adjmats = tree.to_minimal_context_graphs()

for key, graph in adjmats.items():
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout("dot")
    agraph.draw(str(key) + "_csi.png", args='-Glabel="'+str(key)+'"   ')