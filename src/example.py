# %load_ext autoreload
# %autoreload 2

import cstrees.cstree as ct
import numpy as np
import networkx as nx
import itertools
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

def compare_csilists():
    pass

def do_mix(csilist_tuple,l, cards):
    p = len(csilist_tuple[0]) - 1
    mix = [None] * (p+1)

    # Going through all the levels and mix at all levels.
    # The result should be stored somewhere.
    for i, a in enumerate(zip(*csilist_tuple)):
        #print(i, a)
        if a[0] is None: # None means that a[0] is either at level 0 or some of the CI tuple.
                            # So just skip then.
            continue
        if i == l: # if at the current level, the values are joined.
            mix[i] = set(range(cards[l]))
        else:
            mix[i] = set.intersection(*a)
            if len(mix[i]) == 0:
                return None

    return mix

def partition_csis(pair, csilist_list, l, cards):

    print("level {}".format(l))
    # Put the csis in different sets that can
    # possibly be mixed to create new csis.
    csis_to_mix = [[] for _ in range(cards[l])]
    for csilist in csilist_list:
        if len(csilist[l]) > 1: # Only consider those with single value
            continue
        var_val = list(csilist[l])[0] # just to get the value from the set
        csis_to_mix[var_val].append(csilist)
    return csis_to_mix

print("Pairing")

ret = [{} for _ in range(p+1)]
for level in range(p):
    # initiate nebwies in the first run to be
    print("\n#### Level {}".format(level))
    for pair, csilist_list in paired_csis[level].items():
        print("\n#### CI pair {}".format(pair))
        oldies = []
        # This must be here, since we dont know on which variable new mixed will be mixed
        newbies = csilist_list

        iteration = 1
        while len(newbies) > 0:
            print("\n#### Iteration {}".format(iteration))
        #for pair, csilist_list in newbies[level].items():
            #current[pair] = paired_csis[level]
            print(pair)
            print("going through the levels for partitions")
            # should join with csi_list the oldies?

            tmp = [] # list of created csis
            csis_to_absorb = [] # remove from the old ones due to mixing
            # Go through all levels, potentially many times.
            for l in range(1, level+1):
                if l in pair:
                    continue
                #csis_to_mix = partition_csis(pair, csilist_list, l, cards)
                csis_to_mix = partition_csis(pair, newbies + oldies, l, cards)
                print(csis_to_mix)

                #old_csis[pair][l] = csis_to_mix

                # Need to separate the newly created csis from the old.
                # The old should not be re-mixed, i.e., new mixes must contain
                # at least one new csi. How do we implement that? Just create new to mix
                # and skip if not containing a new?
                # E.g. combinations of {a, b, c} X {d, e}
                for csilist_tuple in itertools.product(*csis_to_mix):
                    # Shoyld check that at least one is in the list of newbies?
                    print("mixing")
                    print(csilist_tuple)

                    # Check that at least one is a newbie
                    no_newbies = True
                    for csi in csilist_tuple:
                        if csi in newbies:
                            no_newbies = False
                    if no_newbies:
                        print("no newbies, so skip")
                        continue

                    # Mix
                    mix = do_mix(csilist_tuple, l, cards)
                    print("mix result: ")
                    print(mix)

                    if mix is None:
                        print("Not mixeable")
                        continue
                    else:
                        # Check if some of the oldies should be removed.
                        # Check if som in csilist_tuple is a subset of mix.
                        # If so, remove it from oldies.
                        
                        for csilist in csilist_tuple:
                           a = [x[0] <= x[1] if x[0] is not None else True for x in zip(csilist, mix)] # O(p*K)
                           if all(a):                                
                               print("{} <= {} Its a subset, so will be absorbed/removed later".format(csilist, mix))
                               csis_to_absorb.append(csilist)
                               #oldies.remove(csilist)  # O(#csis)
                                                    
                        # Dont know where in which level this will be grabbed.
                        # mix shul be added somewhere. Or just be part of a set to be considered.
                        # should also check if some of the csis in csilist_tuple should be removed.
                        # In guess from oldies. Otherwise we can also double check in oldies that no csi is
                        # contained in any other. I not minimal. (the can be othe non minimals that we dont see
                        # directly as well, that why we do this process, at least partly).
                        tmp.append(mix)
            # Update the lists
            oldies += newbies
            
            print("CSIS to absorb/remove (can be duplicates)")
            print(csis_to_absorb)
            for csi in csis_to_absorb:
                print("REMOVING {}".format(csi))
                if csi in oldies:
                    oldies.remove(csi)

            newbies = tmp 
            # check the diplicates here somewhere.
            iteration += 1
        ret[level][pair] = oldies
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

stages = {l:[] for l in range(p)}

def csi_set_to_list_format(csilist):
    tmp = []
    print(csilist)
    for i, e in enumerate(csilist[1:p]):
        
        if e is None:
            tmp.append(list(range(cards[i])))
        elif len(e) == 1:
            tmp.append(list(e)[0])
        elif len(e) > 1:
            tmp.append(list(e)) # this should be == range(cards[i]))
        
    return tmp

def csilist_to_csi(csilist):
    
    print(csilist)
    context = {}
    indpair = []
    sep = set()
    for i, e in enumerate(csilist):
        if i == 0:
            continue
        elif e is None:
            indpair.append(i)
        elif len(e) == 1:
            context[i] = list(e)[0]
        elif len(e) > 1:
            sep.add(i) # this should be == range(cards[i]))
    
    context = ct.Context(context)
    #print(indpair)
    #print(context)
    ci = ct.CI_relation({indpair[0]}, {indpair[1]}, sep)
    csi = ct.CSI_relation(ci, context)
    print(csi)
    return csi

for l, csilist in enumerate(ret):
    print("level {}:".format(l))
    tmp = []
    # Convert formats
    for pair, csil in csilist.items():
        print(pair)
        #print(csil)
        for csi in csil:
            #tmplist = csi_set_to_list_format(csi)
            csiobj = csilist_to_csi(csi)
            #print(tmplist)
            #stage = ct.Stage(tmplist)
            #stage.set_random_params(cards)            
            #tmp.append(stage)
            tmp.append(csiobj)
    stages[l] = tmp
    print(csilist)

rels_at_level = stages
#print(stages)

    
# tree.add_stages(stages)

# tree.set_random_stage_parameters()

# a = tree.plot()
# a.draw("minlcsi.png")

#rels = tree.csi_relations_per_level()
print(rels_at_level)

# TODO: The keys should be contexts here, not levels.

rels = {}
for l, r in rels_at_level.items():
    for rel in r:
        if rel.context in rels:
            rels[rel.context].append(rel)
        else:
            rels[rel.context] = [rel]

print(rels)
            
adjmats = ct.csi_relations_to_dags(rels, co)

for key, graph in adjmats.items():
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout("dot")
    agraph.draw(str(key) + "_minlcsi.png")

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
