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
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


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

    def __init__(self, causal_order, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)
        self.tree = None
        self.stages = None
        self.co = causal_order
        self.p = causal_order.p
        # random.seed(5) # For the colors
        self.colors = list(mcolors.cnames.keys())
        # random.shuffle(self.colors)
        self.color_no = 0
        self.stages = {i: [] for i in range(self.p)}
        random.shuffle(self.colors)

    def set_cardinalities(self, cards):
        self.cards = cards

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
        
        #if len(node) == 0:
        #    print("empty stage")
        
        #if self.stages is None:
        #    return None

        lev = len(node)-1
        #print("lev", lev)
        #print(self.stages)
        #print(self.stages[lev])
        stage = None
        if lev in self.stages:
            for s in self.stages[lev]:
                #print(s)
                if node in s:
                    stage = s
                    break

        if stage is None:
            print("No stage found for {}".format(node))
            
        #print("No stage found for {}".format(node))
        #return None
        
        assert(stage is not None)
        return stage                    
            

    def to_df(self):

        # cardinalities header
        d = {self.co.order[i]: [c] for i, c in enumerate(self.cards)}
        df = pd.DataFrame(d, columns=self.co.order)

        for l, stages in self.stages.items():
            for s in stages:
                dftmp = s.to_df(self.co.order)
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

    def estimate_stage_parameters(self, data, method="BDeu", alpha_tot=1):
        # Set stage probabilities

        for lev, stages in self.stages.items():            
            print("Level", lev)
            if lev == self.p-1:
                continue

            stage_counts = sc.counts_at_level(self, lev+1, data) # lev = node?
            # printing stage counts 
            for key, value in stage_counts.items():
                print(str(key), value)
            
            for i, stage in enumerate(stages):
                probs = sc.estimate_parameters(self, stage, stage_counts, method, alpha_tot)
                stage.probs = probs
                stage.color = self.colors[i]    


    def set_random_parameters(self, alpha=1):
        """ This is dependent on if one has sampled from the tree already.
            I.e., if a probablity is already set for an edge, it
            should not be overwritten.
        """

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

                if stage != None: # not a singleton stage
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
            probs = sc.estimate_parameters(self, stage, stage_counts, method, alpha_tot)
            
            # print(list(children))
            # print(probs)

            for i, ch in enumerate(children):
                stage = self.get_stage(node)

                if stage != None:
                    prob = stage.probs[i] # the probs ar already set here..
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
    def to_minimal_context_graphs(self, labels=None): 
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

        cdags = csi_relations_to_dags(minl_csis_by_context, self.co)

        return cdags

    def plot_minl_dags(self):
        pass

    def likelihood(self, data):
        pass

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

        xs.append(self.cards) # cardinalities at the first row

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
            
        
        return np.array(xs)

    def pdf(self, x):
        """Density function evaluated at x.

        Args:
            x (array type): a vector.
        """

    def plot(self, fill=False):

        if fill or (self.tree is None):
            self.create_tree()
            #self.set_random_parameters()

        return plot(self.tree)
        #agraph = nx.nx_agraph.to_agraph(self.tree)
        # agraph.layout("dot")
        # return agraph
        # agraph.draw(filename)


class CI_relation:
    def __init__(self, a, b, sep) -> None:
        self.a = a
        self.b = b
        self.sep = sep

    def __eq__(self, o: object) -> bool:
        return ((((self.a == o.a) & (self.b == o.b)) |
                ((self.a == o.b) & (self.b == o.a)))
                & (self.sep == o.sep))

    def __str__(self) -> str:
        s1 = ""
        for i in self.a:
            s1 += "X{}, ".format(i)
        s1 = s1[:-2]
        s2 = ""
        for i in self.b:
            s2 += "X{}, ".format(i)
        s2 = s2[:-2]
        s3 = ""
        if sum(self.sep) > 0:
            for i in self.sep:
                s3 += "X{}, ".format(i)
            s3 = s3[:-2]
            return "{} ⊥ {} | {}".format(s1, s2, s3)
        return "{} ⊥  {}".format(s1, s2)

#    def __contains__(self, v):
#        return (v in self.a) or (v in self.b) or (v in self.sep)


class CausalOrder:
    def __init__(self, order) -> None:
        self.p = len(order)
        self.order = order
        self.order_inv = {j: i for i, j in enumerate(order)}

    def get_level(self, var):
        return self.order_inv[var]

    def at_level(self, k):
        return self.order[k]


class Stage:
    """
       A Stage is a CSI of form 
       X_a:b _|_ Xj | S_s, X_c=x_c, 
       with an order and mayube an associated cond prob dist.

    """

    def __init__(self, list_repr) -> None:
        self.level = len(list_repr)-1
        self.list_repr = list_repr
        self.color = None
        self.csi = self.to_csi()

    def __hash__(self) -> int:
        #return hash(tuple([tuple(i) for i in self.list_repr]))
        return hash(self.csi.context)

    def __eq__(self, __o: object) -> bool:
        return hash(__o) == hash(self)

    def __contains__(self, node):
        """Checks is a node is contained in a stage. 

        Args:
            node (_type_): A vector of values for each level.

        Returns:
            _type_: _description_
        """

        if (len(node) == 0):
            if (len(self.list_repr) == 0):
                return True
        for i, val in enumerate(self.list_repr):
            # Must check if list
            if (type(val) is list) and (node[i] not in val):
                return False
            if (type(val) is int) and (node[i] != val):
                return False

        return True

    def size(self):
        """
        The number of nodes this stage has in the CStree.
        """
        s = 1
        for e in self.list_repr:
            if type(e) is set:
                s *= len(e)
        return s

    def to_df(self, column_labels):
        """Write sthe stage to dataframe. columns is..?

        Args:
            columns (_type_): _description_

        Returns:
            _type_: _description_
        """
        import pandas as pd

        d = {}

        for i in range(len(column_labels)):
            if i < len(self.list_repr):
                if type(self.list_repr[i]) == set:
                    d[column_labels[i]] = ["*"]
                else:
                    d[column_labels[i]] = [self.list_repr[i]]
            else:
                d[column_labels[i]] = ["-"]

        df = pd.DataFrame(d, columns=column_labels)

        return df

    def set_random_params(self, cards):
        self.probs = np.random.dirichlet(
            [1] * cards[self.level])  # Need to fix this

    def from_df(self):
        pass

    def __sub__(self, stage):
        """ b is typically a sample from the space self.

        Args:
            csi (CSI_rel): The CSI relation to subract.

        Returns:
            list: A list of CSI relations representing the new space.
        """
        a = self
        b = stage
        p = self.level
        cards = [2] * (p+1)
        # Keep all context vars from a. (this is already ok if b was sampled on a).
        # For each created csi, keep 1 of the context vars from b,
        # vary the rest outside the context vars of b (or opposite??) (exept from those that were restricetd by a).

        result = []
        a_list = a.list_repr
        b_list = b.list_repr

        for level, val in enumerate(b_list):
            if type(val) is not int:
                # If not a context variable.
                continue
            if level in a.csi.context.context:
                # if the variable is also in the original space.
                continue
            else:
                # Cor the context varibles we create new csis.
                for v in range(cards[level]):
                    if v == val:
                        continue
                    # Create the new space
                    # This takes care of the fixed ones.
                    l = b_list[:level] + [v] + a_list[level+1:]
                    result.append(Stage(l))

        return result

    def to_csi(self):
        sepseta = set()
        cond_set = set()
        context = {}
        sepsetb = {self.level+1}

        for i, el in enumerate(self.list_repr):
            if type(el) is set:
                sepseta.add(i)
            else:
                context[i] = el

        ci = CI_relation(sepseta, sepsetb, cond_set)
        context = Context(context)
        return CSI_relation(ci, context)

    def intersects(self, stage):
        """ Checks if the paths of two stages intersect.
            Since then they cannot be in the same CStree.
        """

        # Suffcient with some level with no overlap to return False.
        for i, lev in enumerate(stage.list_repr):
            s_lev = self.list_repr[i]
            # Either same number of both sets. It is always the same sets,
            # the whole outcome space.
            if (lev == s_lev):
                continue
            if type(s_lev) is list:
                if (lev in s_lev):
                    continue
            if type(lev) is list:
                if (s_lev in lev):
                    continue
            return False
        return True

    def to_cstree_paths(self):
        tmp = [[i] if type(i) is int else i for i in self.list_repr]
        return list(itertools.product(*tmp))

    def __str__(self) -> str:
        # return str(self.to_csi()) + "; probs: " + str(self.probs)
        # return str(self.list_repr) + "; probs: " + str(self.probs)
        return str(self.list_repr)


class Context:
    def __init__(self, context: dict) -> None:
        self.context = context

    def __str__(self) -> str:
        context_str = ""
        for key, val in self.context.items():
            context_str += "X{}={}, ".format(key, val)
        if context_str != "":
            context_str = context_str[:-2]
        if context_str == "":
            context_str = "None"
        return context_str

    def __contains__(self, key):

        return key in self.context

    def __getitem__(self, key):
        return self.context[key]

    def __eq__(self, __o: object) -> bool:
        return hash(__o) == hash(self)

    def __hash__(self) -> int:
        m = 1  # special case when context is emtpy
        if len(self.context) == 0:
            return hash(())

        m = max(self.context)
        tmp = [None] * (m+1)
        for i in range(m+1):
            if i in self.context:
                tmp[i] = self.context[i]

        return hash(tuple(tmp))


class CSI_relation:
    """This is a context specific relation. Itshould be implemented
       as a context and a CI relation.
    """

    def __init__(self, ci: CI_relation, context: Context, cards=None) -> None:
        self.ci = ci
        self.context = context
        self.cards = cards

    def to_stages(self):
        """Returns a list of stages defining the CSI relation.
        """
        stages = []
        pass

    def __and__(self, other):
        a = self.as_list()
        b = other.as_list()
        c_list = []

        for el in zip(a, b):
            pass

        return CSI_relation(c_list)

    def as_list(self):
        """Should work for pairwise CSIs.

        Returns:
            _type_: _description_
        """
        # Get the level as the max element-1
        # The Nones not at index 0 encode the CI variables.
        def mymax(s):

            if (type(s) is set) and (len(s) > 0):
                return max(s)
            else:
                return 0

        if not ((len(self.ci.a) == 1) and (len(self.ci.b) == 1)):
            print("This only works for pairwise csis.")
            return None
        # print(print(self.ci.sep))
        levels = max(mymax(self.ci.a), mymax(self.ci.b), mymax(
            self.ci.sep), mymax(self.context.context)) + 1
        cards = [2] * levels
        csilist = [None] * levels
        for l in range(levels):
            if (l in self.ci.a) or (l in self.ci.b):
                csilist[l] = None  # None to indicate the CI variables.
            elif l in self.ci.sep:
                csilist[l] = set(range(cards[l]))
            elif l in self.context:
                csilist[l] = {self.context[l]}

        return csilist

    def to_cstree_paths(self, cards: list, order: list):
        """Genreate the set(s) of path defining the CSI relations.
        note that it can be defined by several stages (set of paths).

        Args:
            cards (list): _description_
            order (list): _description_

        Returns:
            _type_: _description_
        """

        level = len(self.ci.a) + len(self.context.context) + 1
        vals = []*level
        for i in self.ci.a:
            vals[i] = range(cards[i])
        for i in self.ci.b:
            vals[i] = range(cards[i])
        for i, j in self.context.context.items():
            vals[i] = j

        return 

    def __add__(self, o):
        """Adding two objects by adding their set of paths and create a new
            CSI_relation.

        Args:
            o (CSI_relation): A CSI relation.

        Returns:
            CSI_relation: A new CSI relation, created by joining the paths in both.
        """
        return CSI_relation(self.to_cstree_paths() + o.to_cstree_paths())

    def __hash__(self) -> int:
        """TODO: Check that the order is correct, so tht not 1 CSI can
        be represented in 2 ways.

        Returns:
            int: hash of the string representation.
        """
        return hash(str(self))

    def __str__(self) -> str:
        if len(self.context.context) == 0:
            return "{}".format(self.ci)
        return "{}, {}".format(self.ci, self.context)


def sample_random_stage(cards: list, level: int, max_contextvars: int, prob: float) -> Stage:
    """Sample a random non-singleton stage.

    Args:
        cards (list): _description_
        level (int): _description_
        max_contextvars (int): _description_
        prob (float): _description_

    Returns:
        Stage: a random stage.
    """
    # The last level cannot contain stages.

    # If the number is smaller than the level, then level is max.
    ncont = max_contextvars
    if max_contextvars > level-1:  # Since not all can be context variables. (i.e. singleton stage)
        ncont = level - 1

    possible_context_vars = np.random.choice(
        range(level+1), ncont, replace=False)

    context_vars = []
    # among the possible context variables, choose some of them.
    for i, val in enumerate(possible_context_vars):
        if np.random.multinomial(1, [prob, 1-prob], size=1)[0][0] == 1:
            context_vars.append(val)

    # for each of the context variables, choose a random value. For the rest, use the whole set.
    vals = [None]*len(cards[:level+1])
    for i, _ in enumerate(cards[:level+1]):
        if i in context_vars:  # changed
            vals[i] = np.random.randint(cards[i])
        else:
            vals[i] = set(range(cards[i]))  # use set here!
    s = Stage(vals)
    return s


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


def csi_relations_to_dags(csi_relations, causal_order, labels=None):

    p = causal_order.p
    graphs = {context: None for context in csi_relations}
    for context, csis in csi_relations.items():
        
        adjmat = np.zeros(p*p).reshape(p, p)
        for j in range(p):
            for i in range(j):
                # This will anyway be disregarded in the matrix slice?
                if (i in context) | (j in context):
                    continue
                # create temp CI relation to compare with
                a = {i}
                b = {j}

                s = {k for k in range(j) if (
                    k != i) and (k not in context)}

                ci_tmp = CI_relation(a, b, s)

                # Check is the ci is in some of the cis of the context.
                # 1. i<j
                # 2. no edge if Xi _|_ Xj | Pa1:j \ i

                cis = []
                for csi in csis:
                    cis += [csi.ci]
                    cis += decomposition(csi.ci)
                    cis += weak_union(csi.ci)

                if ci_tmp in cis:
                    adjmat[i, j] = 0
                else:
                    adjmat[i, j] = 1

        context_els = set(context.context.keys())
        inds = sorted(set(causal_order.order) - context_els)
        adjmat = adjmat[np.ix_(inds, inds)]

        graph = nx.from_numpy_array(adjmat, create_using=nx.DiGraph())

        # Label from 1 instead of 0
        labels = {}
        for i, j in enumerate(inds):
            labels[i] = j
        graph = nx.relabel_nodes(graph, labels)
        graphs[context] = graph

    return graphs


def minimal_context(csi_relations: set) -> set:
    """_summary_

    Args:
        csi_relations (set): _description_

    Returns:
        set: _description_
    """


def get_minimal_dag_imaps(causal_order, csi_relations):

    pass


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
    co = CausalOrder(range(p))
    ct = CStree(co)

    ct.set_cardinalities(cards)

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
            mc = level + 1 # should allow for "singleton" stages at level 0 as this
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


def generate_random_clusters():
    """Generate random stages by merging existing ones.
       Stages can may have only one node.
    """
    # sample two nodes. ()
    # check their stages.
    # (since a stage is sample with prob psoportional to the #nodes in it
    #  this is like sampling a stage, but we dont have to enumerate the stages).
    # merge their stages.
    # Repeat n times.

    nodes_to_stage = {}

    # uniform sampling
    probs = [[1.0/card]*card for card in cardinalities]


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def df_to_cstree(df):

    co = CausalOrder([int(x[0]) for x in df.columns])

    cards = [int(x[1]) for x in df.columns]

    stages = {i: [] for i in range(len(cards)+1)}
    stages[-1] = [Stage([])]
    cstree = CStree(co)
    cstree.set_cardinalities([None] + cards)

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
    cstree.co = co

    # cstree.set_random_stage_parameters()

    return cstree


def multivariate_multinomial(probs):
    x = []
    for dist in probs:
        s = np.random.multinomial(1,  dist, size=1)
        x.append(s)
    return x

    # make sure the last entry is the same


def decomposition(ci: CI_relation):

    cilist = []
    for x in itertools.product(ci.a, ci.b):
        new_ci = CI_relation({x[0]}, {x[1]}, ci.sep)
        if new_ci == ci:
            continue
        cilist.append(new_ci)

    return cilist


def weak_union(ci: CI_relation):

    cis = []
    for d in powerset(ci.b):
        d = set(d)
        if (len(d) == 0) | (d == ci.b):
            continue

        BuD = ci.b
        cis.append(CI_relation(ci.a, BuD-d, ci.sep | d))

    for d in powerset(ci.a):
        d = set(d)
        if (len(d) == 0) | (d == ci.a):
            continue

        d = set(d)
        AuD = ci.a
        cis.append(CI_relation(AuD - d, ci.b, ci.sep | d))

    return cis


def pairwise_cis(ci: CI_relation):
    """ Using weak union just to get pairwise indep relations.

        X_a _|_ X_b | X_d
    Args:
        ci (CI_relation): CI relation
    """
    cis = []
    A = ci.a
    B = ci.b  # This will probably just contain one element.
    for x in itertools.product(A, B):
        rest = (A - {x[0]}) | (B - {x[1]})
        cis.append(CI_relation({x[0]}, {x[1]}, ci.sep | rest))
    return cis


def pairwise_csis(csi: CSI_relation):
    context = csi.context
    ci_pairs = pairwise_cis(csi.ci)
    csis = []
    for ci in ci_pairs:
        csi = CSI_relation(ci, context=context)
        csis.append(csi)
    return csis


def do_mix(csilist_tuple, l, cards):
    p = len(csilist_tuple[0])
    mix = [None] * p

    # Going through all the levels and mix at all levels.
    # The result should be stored somewhere.
    for i, a in enumerate(zip(*csilist_tuple)):
        #print(i, a)
        # None means that a[0] is some of the CI tuple. So just skip.
        if a[0] is None:
            continue
        if i == l:  # if at the current level, the values are joined.
            mix[i] = set(range(cards[l]))
        else:  # do the intersection
            mix[i] = set.intersection(*a)
            if len(mix[i]) == 0:
                return None  # The CSIs are disjoint, so return None.

    return mix


def partition_csis(pair, csilist_list, l, cards):

    #print("level {}".format(l))
    # Put the csis in different sets that can
    # possibly be mixed to create new csis.
    csis_to_mix = [[] for _ in range(cards[l])]
    for csilist in csilist_list:
        if len(csilist[l]) > 1:  # Only consider those with single value
            continue
        var_val = list(csilist[l])[0]  # just to get the value from the set
        csis_to_mix[var_val].append(csilist)
    return csis_to_mix


def csi_set_to_list_format(csilist):
    tmp = []
    # print(csilist)
    for i, e in enumerate(csilist[1:p]):

        if e is None:
            tmp.append(list(range(cards[i])))
        elif len(e) == 1:
            tmp.append(list(e)[0])
        elif len(e) > 1:
            tmp.append(list(e))  # this should be == range(cards[i]))

    return tmp


def csilist_to_csi(csilist):
    """The independent variables are represented by None.

    Args:
        csilist (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print(csilist)
    context = {}
    indpair = []
    sep = set()
    for i, e in enumerate(csilist):
        if e is None:
            indpair.append(i)
        elif len(e) == 1:
            context[i] = list(e)[0]
        elif len(e) > 1:

            sep.add(i)  # this should be == range(cards[i]))

    context = Context(context)

    ci = CI_relation({indpair[0]}, {indpair[1]}, sep)
    csi = CSI_relation(ci, context)
    # print(csi)
    return csi


def csilist_subset(a, b):
    """True if a is a sub CSI of b, i.e. if at each level l,
        a[l] <= b[l].

    Args:
        a (list): list representation of a CSI
        b (list): list representation of a CSI
    """
    a = [x[0] <= x[1] if x[0]
         is not None else True for x in zip(a, b)]  # O(p*K)
    return all(a)


def minimal_csis(paired_csis, cards):
    """ TODO

        Loop through all levels l
        1. For each stage in the level do weak union to get the pairs Xi _|_ Xj | something, and group.
        2. For each such pair go through all levels and try to find mixable CSI by partition on value.
        3. If mixable, mix and put the result in a set woth newly created.

        When we loop through al levels again by where the old CSI are not mixed with each other
        that is, each tuple needs at least one CSI from the new CSIs.

        Does this stop automatically? Will the set with new CSI eventually be empty?

    Args:
        paired_csis (dict): Dict of csis grouped by pairwise indep rels as Xi _|_ Xj.

    Returns:
        _type_: _description_
    """

    p = len(cards)

    ret = [{} for _ in range(p)]
    for level in range(p):
        # initiate newbies in the first run to be
        logging.debug("\n#### Level {}".format(level))
        for pair, csilist_list in paired_csis[level].items():
            # print("\n#### CI pair {}".format(pair))
            oldies = []
            # This must be here, since we dont know on which variable new mixed will be mixed
            newbies = csilist_list

            iteration = 1
            while len(newbies) > 0:
                logging.debug("\n#### Iteration {}".format(iteration))
                # print(pair)
                #print("going through the levels for partitions")
                # should join with csi_list the oldies?

                fresh = []  # list of created csis
                csis_to_absorb = []  # remove from the old ones due to mixing
                # Go through all levels, potentially many times.
                for l in range(level+1):  # Added +1 after refactorization
                    logging.debug("level {}".format(l))
                    if l in pair:
                        continue

                    csis_to_mix = partition_csis(
                        pair, newbies + oldies, l, cards)
                    #logging.debug("csis to mix")
                    # logging.debug(csis_to_mix)

                    # Need to separate the newly created csis from the old ones.
                    # The old should not be re-mixed, i.e., the mixes must contain
                    # at least one new csi. How do we implement that? Just create new to mix
                    # and skip if not containing a new?

                    # E.g. combinations like {a, b, c} X {d, e} X ...
                    for csilist_tuple in itertools.product(*csis_to_mix):

                        # logging.debug(csilist_tuple)
                        # Check that at least one is a newbie
                        no_newbies = True
                        for csi in csilist_tuple:

                            if csi in newbies:
                                no_newbies = False
                                break
                        if no_newbies:
                            #print("no newbies, so skip")
                            continue

                        # Mix
                        mix = do_mix(csilist_tuple, l, cards)
                        if mix is None:
                            #print("Not mixeable")
                            continue
                        else:
                            # print(mix)
                            #assert(sum([len(el)==1 for el in mix if el is not None]) <= 3)
                            if mix not in (oldies + newbies):
                                logging.debug("mixing")
                                logging.debug(csilist_tuple)
                                logging.debug("mix result: ")
                                logging.debug(mix)
                                logging.debug(
                                    "Adding {} as newly created ******".format(mix))
                                fresh.append(mix)
                                # Check if some csi of the oldies should be removed.
                                # I.e. if some in csilist_tuple is a subset of mix.
                                for csilist in csilist_tuple:
                                    # print wher the csi is from, oldies, or newbies.
                                    if csilist_subset(csilist, mix):  # This sho
                                        logging.debug(
                                            "will later absorb {}".format(csilist))
                                        csis_to_absorb.append(csilist)
                logging.debug(
                    "##### Iterated through all levels. Prepare for next round. ### \n ")
                # Update the lists
                logging.debug(
                    "Adding the following newbies (just used for mixing) to the oldies.")
                for nn in newbies:
                    logging.debug(nn)
                oldies += newbies

                # remove duplicates
                res_list = []
                for item in oldies:
                    if item not in res_list:
                        res_list.append(item)
                oldies = res_list

                logging.debug(
                    "CSI to absorb/remove after having been mixed (can be duplicates)")
                for csi in csis_to_absorb:
                    # BUG: this is maybe not ok. Feels bad to alter here. Maybe an absorbtion step after instead.
                    if csi in oldies:  # Shouldnt it be here? Or somewhere else maybe.. Shouldnt we remove it whereever it is?
                        # Maybe make this removal after appending the newbies?
                        logging.debug(csi)
                        oldies.remove(csi)
                    # Maybe remove from new mixes as well then.

                # filter duplicates
                res_list = []
                for item in fresh:
                    if item not in res_list:
                        res_list.append(item)
                fresh = res_list
                logging.debug("New mix results:")
                for t in fresh:
                    logging.debug(t)

                # Added this to see if it fixes the bug..
                # logging.debug("Updating newbies with the unique new mix results")
                logging.debug(
                    "Updating mix results by removing if they already are in oldies, or a subset of an oldie.")
                newbies = []  # check that the newbies are not in oldies!
                for csi in fresh:  # O( #tmp)
                    #logging.debug("REMOVING {}".format(csi))
                    if (csi not in oldies) and (csi not in csis_to_absorb):  # O(#oldies)
                        newbies.append(csi)
                    else:
                        newbies.append(csi)  # Add and then remove maybe :)
                        for o in oldies:  # O(#oldies)
                            if csilist_subset(csi, o):
                                # logging.debug("FOUND A SUBSET OF AN OLDIE############")
                                # logging.debug(csi)
                                #logging.debug("is a subset of")
                                # logging.debug(o)
                                newbies.remove(csi)
                                break

                logging.debug("newbies (new mixes after the filtering):")
                for nb in newbies:
                    logging.debug(nb)
                logging.debug("oldies")
                for o in oldies:
                    logging.debug(o)
                # check the diplicates here somewhere.
                fresh = []
                iteration += 1
            ret[level][pair] = oldies
    return ret


def csis_by_levels_2_by_pairs(rels):

    paired_csis = [None] * len(rels)

    for l, val in rels.items():
        #print("level: {}".format(l))
        csi_pairs = []  # X_i _|_ X_j | something
        for v in val:
            # print(v)
            csis = pairwise_csis(v)
            csi_pairs = csi_pairs + csis

            # Loop though all levels for each of these and try to mix.
            # print("pairwise")
            # print("")
        #print("All pairs")
        cis_by_pairs = {}
        for c in csi_pairs:
            # print(c)
            clist = c.as_list()
            # print(clist)

            pair = tuple([i for i, j in enumerate(clist) if j is None])
            # print(pair)
            if pair in cis_by_pairs:
                cis_by_pairs[pair].append(clist)
            else:
                cis_by_pairs[pair] = [clist]

        # print(csi_pairs)
        # print(cis_by_pairs)
        paired_csis[l] = cis_by_pairs

    return paired_csis


def rels_by_level_2_by_context(rels_at_level):
    rels = {}
    for l, r in rels_at_level.items():
        for rel in r:
            if rel.context in rels:
                rels[rel.context].append(rel)
            else:
                rels[rel.context] = [rel]
    return rels


def csi_lists_to_csis_by_level(csi_lists, p):
    stages = {l: [] for l in range(p)}
    for l, csilist in enumerate(csi_lists):
        #print("level {}:".format(l))
        tmp = []
        # Convert formats
        for pair, csil in csilist.items():
            # print(pair)
            # print(csil)
            for csi in csil:
                #tmplist = csi_set_to_list_format(csi)
                csiobj = csilist_to_csi(csi)
                # print(tmplist)
                #stage = ct.Stage(tmplist)
                # stage.set_random_params(cards)
                # tmp.append(stage)
                tmp.append(csiobj)
        stages[l] = tmp
        # print(csilist)
    return stages


def sample_stage_restr_by_stage(stage: Stage, max_cvars: int, cvar_prob: float, cards: list):
    """
        Samples a Stage on the space restricted by the arggument stage.
        Not allow singleton stages.

    Args:
        stage (Stage): _description_
        n_context_vars (int): _description_
        cvar_prob (float): probability of a randomly picked variable to be a Context variable.
        cards (list): _description_

    Returns:
        _type_: _description_
    """

    space = stage.list_repr
    levelplus1 = len(space) # this is not the full p?    
    #print("levelplus1: {}".format(levelplus1))

    assert (max_cvars <= levelplus1)  # < Since at least one cannot be a cvar.
    # This may not be true if wa are at very low levels where the level in sthe constraint.
    fixed_cvars = len(stage.csi.context.context)
    csilist = [None] * levelplus1

    cont_var_counter = 0
    # random order here, to not favor low levels.
    randorder = list(range(levelplus1))
    random.shuffle(randorder)  

    # TODO: at level 0, it should be possible to have two singleton stages.
    for i in range(levelplus1):
        ind = randorder[i]
        s = space[ind] # a context value (int) or the full set of values.

        if type(s) is int:  # This is a restriction of the space.
            csilist[ind] = s
            cont_var_counter += 1
        else:            
            if cont_var_counter < max_cvars-fixed_cvars:  # Make sure not too many context vars
                # (i.e. a cond var), pick either one or all.

                b = np.random.multinomial(
                    1, [cvar_prob, 1-cvar_prob], size=1)[0][0]
                if b == 0:  # TODO: this should be able to happen anyway?
                    csilist[ind] = set(range(cards[ind]))
                else:
                    v = np.random.randint(cards[ind]) # choose a random context value
                    cont_var_counter += 1
                    csilist[ind] = v
            else: # no more context vars allowed.                
                csilist[ind] = set(range(cards[ind]))

    return Stage(csilist)


def csilist_to_csi_2(csilist):
    a = set()
    b = set()
    s = set()
    c = {}
    for i, e in enumerate(csilist):
        # if i == 0:
        #    continue
        if e is None:
            a |= {i}
        else:
            c[i] = list(e)[0]

    b = {len(csilist)}
    ci = CI_relation(a, b, s)
    cont = Context(c)
    return CSI_relation(ci, cont)


def all_stagings(p, cards, l, max_cvars=1):
    assert(max_cvars == 1)
    
    if l == -1:
        yield [Stage([])]
        return
    
    
    # all possible values for each variable
    vals = [list(range(cards[l])) for l in range(p)]
    for k in range(l+1):  # all variables up to l can be context variables
        # When we restrict to max_cvars = 1, we have two cases:
        # Either all are in one color or all are in different colors.
        stlist = []
        for v in vals[k]:  # Loop through the values of the context variables.
            left = [set(vals[i]) for i in range(k)]
            right = [set(vals[j]) for j in range(k+1, l+1)]
            # For example: [[0,1], [0,1], 0, [0, 1]]
            stagelistrep = left + [v] + right
            st = Stage(stagelistrep)
            stlist += [st]
        yield stlist
    
    # no context variables
    stagelistrep = [set(v) for v in vals][:l+1]
    
    st = Stage(stagelistrep)
    yield [st]
    
def n_stagings(order, cards, l, max_cvars=1):
    stagings = all_stagings(len(order), cards, l, max_cvars)
    return sum(len(staging) for staging in stagings)

def optimal_staging_at_level(order, cards, data, l, max_cvars=1, alpha_tot=None, method="BDeu"):
    p = len(order)
    co = CausalOrder(range(p))
    tree = CStree(co)
    tree.set_cardinalities(cards)
    
    stagings = all_stagings(p, cards, l, max_cvars) # at level l
    max_staging = None
    max_staging_score = -np.inf
    #logging.debug("level: {}".format(l))
    
    for stlist in stagings:
        #logging.debug("scoring staging: {}".format([str(ss) for ss in stlist]))        
        tree.update_stages({l: stlist})
        level_counts = sc.counts_at_level(tree, l+1, data) # This needs the stages to be set at the line above.
        #logging.debug("all counts in the staging")
        #for k, v in level_counts.items():
        #    logging.debug("{}: {}".format(k, v))
        score = sc.score_level(tree, l+1, level_counts, alpha_tot, method)        
        #logging.debug("score: {}".format(score))
        if score > max_staging_score:
            #logging.debug("{} is current max".format(score))
            max_staging_score = score
            max_staging = stlist

    return max_staging, max_staging_score

def optimal_cstree(order, data, max_cvars=1, alpha_tot=1.0, method="BDeu"):
    
    
    # Order should use the same labels as data?
    # Order is only intresting when data is involved. Then its the order of the data columns
    # Otherwise we only talk about levels.
    
    
    cards = data[0]
    p = len(order)
    co = CausalOrder(order)
    tree = CStree(co)
    tree.set_cardinalities(cards)
    
    stages = {}
    stages[-1] = [Stage([])]
    for level in range(-1, p-1): # dont stage the last level
        max_staging, max_staging_score = optimal_staging_at_level(order, cards, data, level, max_cvars, alpha_tot, method)
        stages[level] = max_staging
        print("max staging: {}".format([str(s) for s in max_staging]) )

    tree.update_stages(stages) 
    # maybe set level labels here?       
    return tree
    
    
def find_optimal_order(data, strategy="max", max_cvars=1, alpha_tot=1, method="BDeu"):
    """ Find the optimal causal order for the data.
    """
    
    p = data.shape[1]
    perms = permutations(list(range(p)))
    
    cards = data[0] 
    optimal_order = None
    max_score = -np.inf
    
    # iterate over all permutations
    for perm in list(perms):
        order = list(perm) # dont stage the last variable
        #print("scoring order: {}".format(order))
        score = sc.score_order(order, data, strategy="max", max_cvars=max_cvars, alpha_tot=alpha_tot, method=method)
        print("order: {}, score: {}".format(order, score))
        
        #if score == max_score:
        #    optimal_orders.append(order)
        if score > max_score:
            max_score = score            
            optimal_orders = order
            
    return optimal_orders, max_score
        
        
def find_optimal_cstree(data, max_cvars=1, alpha_tot=1, method="BDeu"):
    opt_order = find_optimal_order(data, max_cvars=max_cvars, alpha_tot=alpha_tot, method=method)
    print("optimal order: {}".format(opt_order))
    
    optimal_staging = optimal_cstree(opt_order, data, max_cvars=max_cvars, alpha_tot=alpha_tot, method=method)
    print("optimal staging: {}".format(optimal_staging))    