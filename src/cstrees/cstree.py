import math
from math import comb
from random import uniform
from re import S
import networkx as nx
import numpy as np
import matplotlib
from itertools import chain, combinations
import itertools
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import pandas as pd
import logging
import sys
import cstrees.scoring as sc
from itertools import permutations

from cstrees.stage import * 
from cstrees.csi_relation import *
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


def plot(graph, layout="dot"):
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout(layout)
    return agraph


class CStree(nx.Graph):
    """
        However, there can be like O(2^p) differnet minimal contexts,
        so maybe its impossible. Then we would need some limit on the number
        of nodes in the minimal contexts. But even if we limit the number of
        nodes in the context to 1, the are about 2^p such sequences/sets.

    Args:
        causal_order (list): A causal order of the variables.

    Example:
        >>> import cstrees.cstree as ct
        >>> import numpy as np
        >>> # CStree from Figure 1 in (Duarte & Solus, 2022)
        >>> np.random.seed(1)
        >>> p = 4
        >>> co = range(1, p+1)
        >>> tree = ct.CStree(co)
        >>> tree.set_cardinalities([None] + [2] * p)
        >>> tree.set_stages({
        >>>     0: [],
        >>>     1: [],
        >>>     2: [{(0, 0), (1, 0)}],  # green
        >>>     3: [{(0, 0, 0), (0, 1, 0)},  # blue
        >>>         {(0, 0, 1), (0, 1, 1)},  # orange
        >>>         {(1, 0, 0), (1, 1, 0)}]  # red
        >>> })

    """

    def __init__(self, cards, labels=None, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)
        self.tree = None
        self.stages = None
        self.cards = cards
        self.p = len(cards)  # causal_order.p
        if labels is None:
            self.labels = list(range(self.p))
        else:
            self.labels = labels

        self.colors = list(mcolors.cnames.keys())
        random.shuffle(self.colors)
        self.color_no = 0
        self.stages = {i: [] for i in range(self.p)}

    def n_colored_stages_at_level(self, l):
        # n_colored may include singleton stages, on low levels.
        n_colored = len(self.stages[l])

        # Count the colored size and subtract it from the total number of nodes.
        # This gives the number of singleton stages.
        n_singleton = 0
        size_colored = np.sum([s.size() for s in self.stages[l]])

        return 1

    def update_stages(self, stages: dict):
        """Adds a stage.

        Example:

        """
        #self.stages = stages
        self.stages.update(stages)
        self.stages[-1] = [Stage([])]

        # Add support for the set format too
        # self.stage_probs = {key: [None]*len(val)
        #                    for key, val in stages.items()}

    def get_stages(self, level: int):
        """ Get all the stages in one level.

        Args:
            level (int): A level corresponds to variable in the causal ordering.
        """
        pass

    def get_stage(self, node: tuple):
        """ Get the stages of node.
            TODO: It should probably also give the singteton stage if so.

        Args:
            node (int): node.
        """
        assert(self.stages is not None)
        #print("get_stage of ", node)

        # if len(node) == 0:
        #    print("empty stage")

        # if self.stages is None:
        #    return None

        lev = len(node)-1

        #print(self.stages[lev])
        stage = None
        if lev in self.stages:
            for s in self.stages[lev]:
                # print(s)
                if node in s:
                    stage = s
                    break

        if stage is None:
            print("No stage found for {}".format(node))

        #print("No stage found for {}".format(node))
        # return None

        assert(stage is not None)
        return stage

    def to_df(self):

        # cardinalities header
        d = {i: [c] for i, c in enumerate(self.cards)}
        #d = {self.co.order[i]: [c] for i, c in enumerate(self.cards)}
        #df = pd.DataFrame(d, columns=self.co.order)
        df = pd.DataFrame(d, columns=self.labels)

        for l, stages in self.stages.items():
            for s in stages:
                #dftmp = s.to_df(self.co.order)
                dftmp = s.to_df(self.labels)
                df = pd.concat([df, dftmp])

        return df

    def from_df(self, df):

        for row in df.iterrows():
            pass

    def set_random_stage_parameters(self, alpha=1):
        # Set stage probabilities
        for lev, stages in self.stages.items():

            for i, stage in enumerate(stages):
                probs = np.random.dirichlet([alpha] * self.cards[lev+1])
                stage.probs = probs
                stage.color = self.colors[i]

        self.set_tree_probs()

    def estimate_stage_parameters(self, data, method="BDeu", alpha_tot=1):
        # Set stage probabilities

        for lev, stages in self.stages.items():
            #print("Level", lev)
            if lev == self.p-1:
                continue

            stage_counts = sc.counts_at_level(self, lev+1, data)  # lev = node?
            # printing stage counts
            # for key, value in stage_counts.items():
            #    print(str(key), value)

            for i, stage in enumerate(stages):
                probs = sc.estimate_parameters(
                    self, stage, stage_counts, method, alpha_tot)
                stage.probs = probs
                stage.color = self.colors[i]

        self.set_tree_probs()

    def set_tree_probs(self, alpha=1):
        """ This is dependent on if one has sampled from the tree already.
            I.e., if a probablity is already set for an edge, it
            should not be overwritten.
        """
        # First create the tree.
        # This has to do with that, if we allow for singleton stages, we need to
        # make sure it is not too big sort of.

        self.create_tree()

        # Check if the node is part part of a context (stage?)
        # if so we may overwrite probs. Otherwise, generate new ones.
        for node in self.tree.nodes():
            if len(node) == self.p:
                continue

            children = sorted(list(self.tree.successors(node)))
            probs = np.random.dirichlet([alpha] * self.cards[len(node)])
            # Seems like a stage at level l holds the probtable for the variabel at level l+1.
            for i, ch in enumerate(children):
                #print("i ", i, "ch ", ch, "node ", node)
                stage = self.get_stage(node)

                if stage != None:  # NO singleton stages allowed!
                    prob = stage.probs[i]

                    #print("edge prob {}".format(prob))
                    self.tree[node][ch]["cond_prob"] = prob
                    self.tree[node][ch]["label"] = round(prob, 2)
                    self.tree[node][ch]["color"] = stage.color
                    self.tree.nodes[node]["color"] = stage.color
                else:
                    #print("Singleton stage")
                    self.tree[node][ch]["cond_prob"] = probs[i]
                    self.tree[node][ch]["label"] = round(probs[i], 2)

    def estimate_parameters(self, data, method="BDeu", alpha_tot=1):

        # node is equal to level I think.. we treat the labelling outside.
        for node in self.tree.nodes():
            if len(node) == self.p:
                continue

            stage_counts = sc.counts_at_level(self.tree, node, data)

            children = self.tree.successors(node)
            #probs = np.random.dirichlet([1] * self.cards[len(node)])

            # estimatest the probabilites for the nodes having this stage.
            # actually the nodes should have the distribution of the stage but
            # since can be quite many we assign it to ste stage instead for convenience.
            probs = sc.estimate_parameters(
                self, stage, stage_counts, method, alpha_tot)

            # print(list(children))
            # print(probs)

            for i, ch in enumerate(children):
                stage = self.get_stage(node)

                if stage != None:
                    prob = stage.probs[i]  # the probs ar already set here..
                    self.tree[node][ch]["cond_prob"] = prob
                    self.tree[node][ch]["label"] = round(prob, 2)
                    self.tree[node][ch]["color"] = stage.color
                    self.tree.nodes[node]["color"] = stage.color
                else:
                    self.tree[node][ch]["cond_prob"] = probs[i]
                    self.tree[node][ch]["label"] = round(probs[i], 2)

    def get_stage_no(self, node):
        lev = len(node)
        for stages in self.stages[lev]:
            for i, stage_dict in enumerate(stages):
                if node in stage_dict:
                    return i
        return None

    def create_tree(self):
        if self.tree is None:
            self.tree = nx.DiGraph()

        # All the levels of the firs variable
        tovisit = [(i,) for i in range(self.cards[0])]

        while len(tovisit) != 0:
            # Visit/create node in the tree
            node = tovisit.pop()
            lev = len(node)-1  # added -1
            fr = node[:lev]
            to = node
            if not self.tree.has_edge(fr, to):
                self.tree.add_edge(fr, to)  # check if exists first
            else:
                pass
            self.tree.nodes[to]["label"] = to[-1]
            # Add more nodes to visit
            if lev < self.p-1:
                #np.random.dirichlet([1] * lev)
                for i in range(self.cards[lev + 1]):
                    tovisit.append(to + (i,))

    def minimal_context_csis(self):
        """ Returns the minimal contexts.
        """
        pass

    # Here is the only place we need labels/orders.
    def to_minimal_context_graphs(self):
        """ This returns a sequence of minimal context graphs (minimal I-maps).
        """
        logging.debug("getting csirels per level")
        logging.debug(self.stages)
        rels = self.csi_relations_per_level()
        logging.debug("rels")
        logging.debug(rels)
        for k, rs in rels.items():
            for r in rs:
                logging.debug(r)
        paired_csis = csis_by_levels_2_by_pairs(rels)
        logging.debug("###### paired_csis")
        logging.debug(paired_csis)

        logging.debug("\n ######### minl cslisist")
        minl_csislists = minimal_csis(paired_csis, self.cards[1:])
        logging.debug(minl_csislists)

        logging.debug("\n ############### get minl csis in list format")
        minl_csis = csi_lists_to_csis_by_level(minl_csislists, self.p)
        logging.debug(minl_csislists)
        for key in minl_csislists:
            # logging.debug(key)
            for pair, val in key.items():
                logging.debug("{}: {}".format(pair, val))
                #logging.debug("{} {}".format(val))

        logging.debug("#### minimal csis")
        minl_csis_by_context = rels_by_level_2_by_context(minl_csis)
        logging.debug(minl_csis_by_context)
        for pair, val in minl_csis_by_context.items():
            for csi in val:
                logging.debug(csi)

        cdags = csi_relations_to_dags(
            minl_csis_by_context, self.p, labels=self.labels)

        return cdags

    def csi_relations_per_level(self, level="all"):

        return {l: [s.to_csi() for s in stages] for l, stages in self.stages.items()}

    def csi_relations(self, level="all"):
        """ Returns all the context specific indepencende (CSI) relations.
            These should normally be thinned out using absorption, and then we would extract
            the minmal contexts based on that.

            TODO: This should be returned by level.
        """
        csi_rels = {}
        # print(self.stages)
        for key, stage_list in self.stages.items():
            #print("{} {}".format(key, stage_list))
            for stage in stage_list:
                #       print(stage)
                csi_rel = stage.to_csi()

                if csi_rel.context not in csi_rels.keys():  # should be a list
                    csi_rels[csi_rel.context] = [csi_rel]
                else:
                    csi_rels[csi_rel.context].append(csi_rel)

        #print(set([ str(k)  for k in csi_rels.keys()]))

        return csi_rels

    def sample(self, n):
        """Draws n random samples from the CStree.
            Dynamically generates nodes in the underlying tree
            and associated parameters on the fly in order to avoid
            creating the whole tree, which is O(2^p), just to sample data.

        Args:
            n (int): number of random samples.
        """

        if self.tree is None:
            print("Creating tree on the fly on sampling.")
            self.tree = nx.DiGraph()
        xs = []

        xs.append(self.cards)  # cardinalities at the first row

        for _ in range(n):
            node = ()
            x = []
            while len(x) < self.p:
                #print(node, x)
                # Create tree dynamically.
                if (node not in self.tree) or len(self.tree.out_edges(node)) == 0:
                    lev = len(node)-1  # changed to -1
                    edges = [(node, node + (ind,))
                             for ind in range(self.cards[lev+1])]
                    #print("adding edges {}".format(edges))
                    self.tree.add_edges_from(edges)

                    # Sample parameters

                    # We set the parametres at random. But if the node belongs to a
                    # stage we overwrite.
                    probs = np.random.dirichlet([1] * self.cards[lev+1])

                    # Check if node is in some stage
                    s = self.get_stage(node)
                    color = ""
                    if s is not None:
                        probs = s.probs
                        if s.color is None:
                            s.color = self.colors[self.color_no]
                            self.color_no += 1

                        color = s.color

                    # Hope this gets in the right order
                    edges = list(self.tree.out_edges(node))
                    # print(edges)
                    # Set parameters
                    for i, e in enumerate(edges):
                        # print(i)
                        self.tree[e[0]][e[1]]["cond_prob"] = probs[i]
                        self.tree[e[0]][e[1]]["label"] = round(probs[i], 2)
                        self.tree.nodes[e[1]]["label"] = e[1][-1]
                        self.tree[e[0]][e[1]]["color"] = color
                        self.tree.nodes[e[0]]["color"] = color

                edges = list(self.tree.out_edges(node))
                # print(self.tree[()][(0,)])
                probabilities = [self.tree[e[0]][e[1]]["cond_prob"]
                                 for e in edges]
                # print(probabilities)
                #elements = [str(e[1]) for e in edges]
                # ind is the index or the outcome of the set_d variable
                vals = len(edges)
                # print(vals)
                ind = np.random.choice(len(edges), 1, p=probabilities)[0]
                node = edges[ind][1]  # This is a typle like (0,1,1,0)
                x.append(node[-1])  # Take the last element, 0, above
                #print("One sample! {}".format(x))

            xs.append(x)

        df = pd.DataFrame(xs)
        df.columns = self.labels
        return df

    def pdf(self, x):
        """Density function evaluated at x.

        Args:
            x (array type): a vector.
        """

    def plot(self, fill=False):

        if fill or (self.tree is None):
            self.create_tree()
            # self.set_random_parameters()

        return plot(self.tree)
        #agraph = nx.nx_agraph.to_agraph(self.tree)
        # agraph.layout("dot")
        # return agraph
        # agraph.draw(filename)


def comp_bit_strings(a):
    """Takes a set of bit strings,representaing outcome paths in a CStree. and returns
        a tuple representing the CSI relation.

    Args:
        a (set of paths): Set of outcome strings.

    Returns:
        list: List telling at which positions the bitstrings are the same and which
        value they have (i.e. the context).

    Example:
        >>> {(1,0,2), (1,1,2)} -> (1, False, 2)

    """
    lev = len(list(a)[0])
    levels = [False]*lev
    for i in range(lev):
        tmp = set()
        for el in a:
            tmp.add(el[i])
        if len(tmp) == 1:
            levels[i] = tmp.pop()

    Stage()
    return levels


def sample_cstree(cards: list, max_cvars: int, prob_cvar: int, prop_nonsingleton: float) -> CStree:
    """
       Sample a random CStree with given cardinalities.

       K = cardinality at level.
       Maximal number of stages at a level is K^level.
       Assume prob_contextvars = 1.
       max_contextvars = level => #stages = K^level.
       max_contextvars = 1 => #stages = K^1 * (level choose 1). The size (number of leafs in the tree) if each stage will be K^(l-1).
       max_contextvars = 2 => #stages = K^2 * (level choose 2). The size if each stage will be K^(l-2).
       ...       
       These can overlap! Without overlaps it is a bit different i guess. Think about the leafs instead. there is room for max K^(l+1) stages (including singletons).
       Maybe count the proportion of singletons left! Keeping trac of the rest of the space!! This is the way to do it.

    Args:
        cardinalities (list): cardinalities of the variables.

    Returns:
        CStree: a CStree.
    """
    p = len(cards)
    ct = CStree(cards)

    stages = {}
    for level, val in enumerate(cards[:-1]):  # not the last level

        # fix max_context_vars if higher than level
        #print("level {}".format(level))
        stage_space = [Stage([set(range(cards[l])) for l in cards[:level+1]])]

        full_stage_space_size = stage_space[0].size()

        # proportion_left = 1.0 # stage_space.size() / full_state_space_size
        singleton_space_size = full_stage_space_size
        # print(proportion_left)

        mc = max_cvars
        if level < mc:
            mc = level + 1  # should allow for "singleton" stages at level 0 as this
            # this correponds to the half of the space.
        # in general, when the level is the liming factor, the number of stages
        # should be as many as possible. Hence, mc = level +1
        # It should not make abig difference. But in this way, we actually show
        # all the stages expicitly, even when saving to file, without risking
        # to blow up the space.

        #print("level: {}, mc: {}".format(level, mc))

        # BUG: for binary.. take max of mc elements in cards.
        minimal_stage_size = 2**(level+1-mc)

        #print("full_stage_space_size: {}".format(full_stage_space_size))
        #print("level: {}, mc: {}, minimal_stage_size: {}".format(level, mc, minimal_stage_size))

        #minimal_stage_prop = minimal_stage_size / full_stage_space_size
        # The special case when the granularity is to coarse.
        # Randomly add anyway?
        # Setting the upper boudn likje this may generalize. However, the stage should not
        # always be accepted..
        #prop_nonsingleton = max(prop_nonsingleton, minimal_stage_size / full_state_space_size)

        # Allowing only non-singleton stages, this should never happen.
        if prop_nonsingleton < minimal_stage_size / full_stage_space_size:
            logging.info("The size (proportion {}) of a minimal is larger than {}.".format(
                minimal_stage_size / full_stage_space_size, prop_nonsingleton))
            b = np.random.multinomial(
                1, [prob_cvar, 1-prob_cvar], size=1)[0][0]
            stages[level] = []

            if b == 0:
                print(b)
                # it should be the whole space to begin with
                stage_restr = stage_space.pop(0)
                new_stage = sample_stage_restr_by_stage(
                    stage_restr, mc, 1.0, cards)
                stages[level].append(new_stage)
            continue  # cant add anything anyway so just go to the next level.

        # print(mc)
        #max_n_stages = comb(level+1, mc) * (cards[level]**mc)
        # logging.debug("Max # stages with {} context variables: {}".format(mc, max_n_stages))
        stages[level] = []
        #m = math.ceil(max_n_stages * frac_stages_per_level)
        #logging.debug("Trying to add max of {} stages".format(m))

        while (1 - (singleton_space_size / full_stage_space_size)) < prop_nonsingleton:
            colored_size_old = full_stage_space_size - singleton_space_size
            # Choose randomly a stage space
            space_int = np.random.randint(len(stage_space))
            stage_restr = stage_space.pop(space_int)
            #print("stage restr: {}".format(stage_restr))
            #print(mc, prob_cvar)
            new_stage = sample_stage_restr_by_stage(
                stage_restr, mc, prob_cvar, cards)
            #print("proposed new stage: {}".format(new_stage))

            new_space = stage_restr - new_stage  # this is a list of substages after removal
            stage_space += new_space  # adds a list of stages
            # subtract the size of the sampled stage
            singleton_space_size -= new_stage.size()
            # Problem with level 0 since then the whole level is always filled.
            new_stage_size = new_stage.size()
            colored_prop = 1 - singleton_space_size / full_stage_space_size
            # Check it its over full after adding the new space
            if colored_prop > prop_nonsingleton:
                # Check if it would be possible to add the smallest stage.
                # Restore the old space and maybe try again

                if (minimal_stage_size + colored_size_old)/full_stage_space_size <= prop_nonsingleton:
                    stage_space = stage_space[:-len(new_space)]
                    stage_space += [stage_restr]
                    singleton_space_size += new_stage.size()
                else:
                    # This is when it is impossible to push in a new stages since all are be too big.
                    break
            else:
                stages[level].append(new_stage)
            #print("proportion left")
            #print(space_left / full_state_space_size)

    stages[-1] = [Stage([])]
    ct.update_stages(stages)
    # ct.set_random_stage_parameters()

    return ct



def df_to_cstree(df):
    cards = cards = df.iloc[0].values  # [int(x[1]) for x in df.columns]

    stages = {i: [] for i in range(len(cards)+1)}
    stages[-1] = [Stage([])]
    cstree = CStree(cards)
    #cstree.set_cardinalities([None] + cards)
    cstree.labels = df.columns

    for row in df.iterrows():
        stage_list = []

        for level, val in enumerate(row[1]):
            if val == "*":
                stage_list.append(list(range(cards[level])))
            elif val == "-":
                s = Stage(stage_list)
                stages[level].append(s)
                break
            else:
                stage_list.append(int(val))

    cstree.update_stages(stages)

    return cstree


def multivariate_multinomial(probs):
    x = []
    for dist in probs:
        s = np.random.multinomial(1,  dist, size=1)
        x.append(s)
    return x

    # make sure the last entry is the same


def all_stagings(p, cards, l, max_cvars=1):
    """ Generates an iterator over all stagings of a given level.
        A staging with 2 stages for a binary CStree at level 2 
        (numbering levels from 0) could e.g. be: 
        [Stage([{0, 1}, 0, {0, 1}]), Stage([{0, 1}, 1, {0, 1}])]

    Args:
        p (int): Number of variables.
        cards (list): List of cardinalities of the variables.
        l (int): The level of the stage.
        max_cvars (int, optional): The maximum number of context variables . Defaults to 1.

    Raises:
        NotImplementedError: Exception if max_cvars > 1.

    Yields:
        _type_: Iterator over all stagings of a given level.
    """

    if max_cvars == 1:
        if l == -1:  # This is an imaginary level -1, it has no stages.
            yield [Stage([])]
            return

        # All possible values for each variable
        vals = [list(range(cards[l])) for l in range(p)]
        for k in range(l+1):  # all variables up to l can be context variables
            # When we restrict to max_cvars = 1, we have two cases:
            # Either all are in 1 color or all are in 2 different colors.
            stlist = []  # The staging: list of Stages.
            # Loop through the values of the context variables.
            for v in vals[k]:
                left = [set(vals[i]) for i in range(k)]
                right = [set(vals[j]) for j in range(k+1, l+1)]
                # For example: [{0,1}, {0,1}, 0, {0, 1}]
                stagelistrep = left + [v] + right
                st = Stage(stagelistrep)
                stlist += [st]
            yield stlist

        # The staging with no context variables
        stagelistrep = [set(v) for v in vals][:l+1]

        st = Stage(stagelistrep)
        yield [st]
    elif max_cvars == 2:
        from cstrees.double_cvar_stagings import enumerate_stagings
        

        for staging_list in enumerate_stagings(l+1):
            
            staging = []
            for stage_list in staging_list:
                # Fix repr bug
                if isinstance(stage_list, set):
                    stage_list = [stage_list]
                
                stage = Stage(stage_list)
                staging.append(stage)
            yield staging
                
    else:
        raise NotImplementedError("max_cvars > 1 not implemented yet")


