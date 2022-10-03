# %load_ext autoreload
# %autoreload 2

import cstrees.cstree as ct
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout


# CStree from Figure 1 in (Duarte & Solus, 2022)

p = 4
co = range(1, p+1)
tree = ct.CStree(co)

tree.set_cardinalities([None] + [2] * p)
tree.add_stages({
    0: None,
    1: None,
    2: [{(0, 0), (1, 0)}],  # green
    3: [{(0, 0, 0), (0, 1, 0)},  # blue
        {(0, 0, 1), (0, 1, 1)},  # orange
        {(1, 0, 0), (1, 1, 0)}]  # red
})

tree.create_tree()


tree.set_random_parameters()

# print(tree.tree.edges)
#print(tree.tree.nodes)

#pos = graphviz_layout(tree.tree, prog="twopi")
#nx.draw(tree, pos)
# plt.show()

# plt.savefig("test.png")


# options = {
#     "font_size": 5,
#     "node_size": 500,
#     "node_color": "white",
#     "edgecolors": "black",
#     "linewidths": 1,
#     "width": 1,
# }
# plt.figure(figsize=(8, 8))
# pos = graphviz_layout(tree.tree, prog="dot",)
# nx.draw_networkx(tree.tree, pos=pos, with_labels=True, **options)

# # Set margins for the axes so that nodes aren't clipped
# ax = plt.gca()
# ax.margins(0.20)
# plt.axis("off")
# plt.show()

tree.tree.nodes[(1,)]["color"] = "red"

#print(tree.tree.nodes)

agraph = nx.nx_agraph.to_agraph(tree.tree)
agraph.layout("dot")


agraph.draw("test.png")


#dags = tree.to_minimal_context_graphs()
