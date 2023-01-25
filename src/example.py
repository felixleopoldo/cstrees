# %load_ext autoreload
# %autoreload 2

import cstrees.cstree as ct
import numpy as np
import networkx as nx
# CStree from Figure 1 in (Duarte & Solus, 2022)
np.random.seed(1)

p = 6

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

stage = ct.sample_random_stage(cards,4)
stage.set_random_params(cards)

tree.set_cardinalities([None] + cards)

# These do not have to be in a dict like this as the levels are
# determined from the length of the tuples.

tree.add_stages({
    0: [],
    1: [],
    2: [ct.Stage([[0, 1], 0])],    # Green
    3: [ct.Stage([0, [0, 1], 0]),  # Blue
        ct.Stage([0, [0, 1], 1]),  # Orange
        ct.Stage([[0,1], [0, 1], 0])]  # Red
})

tree.set_random_stage_parameters()

a = tree.plot()
a.draw("testplot.png")

x = tree.sample(5)
# print(x)
rels = tree.csi_relations()
adjmats = ct.csi_relations_to_dags(rels, co)

for key, graph in adjmats.items():
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout("dot")
    agraph.draw(str(key) + ".png")
    # print(graph.nodes)
    # print(graph.edges())

rels = tree.csi_relations_per_level()
paired_csis = [None] * len(rels)

for l, val in rels.items():
    #print("level: {}".format(l))
    csi_pairs = [] # X_i _|_ X_j | something
    for v in val:
        print(v)
        csis = ct.pairwise_csis(v)
        csi_pairs = csi_pairs + csis
        
        # Loop though all levels for each of these and try to mix.
        #print("pairwise")
        #print("")
    #print("All pairs")
    cis_by_pairs = {}
    for c in csi_pairs:
        #print(c)
        clist = c.as_list()
        #print(clist)
        
        pair = tuple([i for i, j in enumerate(clist) if (j is None) and i > 0])
        #print(pair)
        if pair in cis_by_pairs:
            cis_by_pairs[pair].append(clist)
        else:
            cis_by_pairs[pair] = [clist]
        
        
    #print(csi_pairs)
    #print(cis_by_pairs)
    paired_csis[l] = cis_by_pairs
    
    #print("")
for l, csilist in enumerate(paired_csis):
    print("level {}:".format(l))
    print(csilist)
#l5rels = tree.csi_relations(level=1)
#l5indpairs = makeindpairs(l5rels) # These should be grouped already I guess
#merged = mergeindpairs(l5indpairs) # This shoudl remove the superfalus pairs. 
#These can also be subjects for merging but a a different level?.
# - No, must be at the same level. And the same ind pair.
# How to check minimail context/csi? 
# - Its minimal if it cant be merged, without residuals, with anything else in the set. 
#   Since then a context var would vanish.
#   minimal does not mean that it cant be absorbed.! Most are minimal I think.
#   its just if ic can be created from absorbing 2 others in the set, WITHOUT RESIDUALS, i guess. 
#   I mean exact matching absorbtion. 
#   Actually, it is minimal if ther doesnt exist any CSI which contians it, 
#   ie one with a conditioning variable instead of just a context vaariable..
#   So which I write about is that we do the merging/absorbing first, and then check this.
#   More generally, we remove one of a pair of CSIs if it is contained in the other. (this i knew before).
#
#   Our absorbtion is not only to chek for minimal CSI but also tro create new CSIs, that mey be minimal or not?

# One type of pair (X_j _|_ X_j | something) will only come from
# level j. But may come from different stages at that level.


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

# Loop through all levels l
# 1. For each stage in the level do weak union to get the pairs Xi _|_ Xj | something, and group.
# 2. For each such pair go through all levels and try to find mixable CSI by partition on value.
# 3. If mixable, mix and put the result in a set woth newly created.
# 
# When we loop through al levels again by where the old CSI are not mixed with each other
# that is, each tuple needs at least one CSI from the new CSIs.
#
# Does this stop automatically? Will the set with new CSI eventually be empty?
