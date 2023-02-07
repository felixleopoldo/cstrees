# %load_ext autoreload
# %autoreload 2

import cstrees.cstree as ct
import numpy as np
import networkx as nx
import itertools
# CStree from Figure 1 in (Duarte & Solus, 2022)
np.random.seed(1)

p = 4

t = ct.sample_cstree(p)
t.set_random_stage_parameters()
t.plot()
rels = t.csi_relations()

print("level and csis")
for key, val in rels.items():
    print(val)

x = t.sample(5)
print(x)
print()


co = ct.CausalOrder(range(1, p+1))
tree = ct.CStree(co)
cards = [2] * p

#stage = ct.sample_random_stage(cards,2)
#stage.set_random_params(cards)

tree.set_cardinalities([None] + cards)

# These do not have to be in a dict like this as the levels are
# determined from the length of the tuples.

tree.add_stages({
    0: [],
    1: [],
    2: [ct.Stage([[0, 1], 0])],    # Green
    3: [ct.Stage([0, [0, 1], 0]),  # Blue
        ct.Stage([0, [0, 1], 1]),  # Orange
        ct.Stage([1, [0, 1], 0])]  # Red
})

p=4
tree.set_random_stage_parameters()

a = tree.plot()
a.draw("testplot.png")

x = tree.sample(5)
# print(x)
rels = tree.csi_relations()

print("Initial rels")
print(rels)
adjmats = ct.csi_relations_to_dags(rels, co)

i = 1
for key, graph in adjmats.items():
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout("dot")

    #agraph.draw(str(key) + "_csi.png", args="-Glabel=Context:"+str(key) +"   ")
    agraph.draw(str(key) + "_csi.png", args='-Glabel="'+str(key)+'"   ')
    # print(graph.nodes)
    # print(graph.edges())
    i += 1

rels = tree.csi_relations_per_level()


for l, csilist in enumerate(paired_csis):
    print("level {}:".format(l))
    print(csilist)

print("ret")
print(ret)
for l, csilist in enumerate(ret):
    print("level {} {}:".format(l, csilist))

print(p)
co = ct.CausalOrder(range(1, p+1))
tree = ct.CStree(co)
cards = [2] * p


tree.set_cardinalities([None] + cards)

# These do not have to be in a dict like this as the levels are
# determined from the length of the tuples.

rels_at_level = stages
#print(stages)

    
# tree.add_stages(stages)

# tree.set_random_stage_parameters()

# a = tree.plot()
# a.draw("minlcsi.png")

#rels = tree.csi_relations_per_level()
print(rels_at_level)

# TODO: The keys should be contexts here, not levels.


print(rels)
            
cdags = ct.csi_relations_to_dags(rels, co)

for key, graph in cdags.items():
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout("dot")
    agraph.draw("minl_cont_dag_"+str(key) + ".png", args="-Glabel="+str(key) +"    ")

