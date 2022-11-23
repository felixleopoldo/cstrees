
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
        >>> tree.add_stages({
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
        random.seed(1)
        self.colors = list(mcolors.cnames.keys())
        self.color_no = 0
        random.shuffle(self.colors)
        

    def set_cardinalities(self, cards):
        self.cards = cards

    def add_stages(self, stages: dict):
        """Adds a stage.

        Example:
            >>> tree.add_stages({
            >>>     0: [],
            >>>     1: [],
            >>>     2: [{(0, 0), (1, 0)}],  # green
            >>>     3: [{(0, 0, 0), (0, 1, 0)},  # blue
            >>>         {(0, 0, 1), (0, 1, 1)},  # orange
            >>>         {(1, 0, 0), (1, 1, 0)}]  # red
            >>> })
        """
        self.stages = stages
        #self.stage_probs = {key: [None]*len(val)
        #                    for key, val in stages.items()}

    def get_stages(self, level: int):
        """ Get all the stages in one level.

        Args:
            level (int): A level corresponds to variable in the causal ordering.
        """
        pass

    def get_stage(self, node: tuple):
        """ Get the stages of node.


        Args:
            node (int): node.
        """
        if self.stages is None:
            return None
        
        lev = len(node)
        
        
        if lev in self.stages:
            for s in self.stages[lev]:
                if node in s: 
                    return s
        return None
    
 
    def to_df(self):
        
        # cardinalities header
        d = {self.co.order[i]:[c] for i,c in enumerate(self.cards[1:])}
        df = pd.DataFrame(d, columns=self.co.order)
        
        for l, stages in self.stages.items():
            for s in stages:            
                dftmp = s.to_df(self.co.order)
                df = pd.concat([df, dftmp])

        return df
    
    def from_df(self, df):
        
        for row in df.iterrows():
            pass

    
    def set_random_stage_parameters(self):
        # Set stage probabilities

        cols = ["red", "blue", "green", "purple", "yellow", "grey"]
        self.colors = {key: cols[:len(val)]
                       for key, val in self.stages.items()}

        for lev, stages in self.stages.items():
            for i, stage in enumerate(stages):
                probs = np.random.dirichlet([1] * self.cards[lev+1])
                stage.probs = probs
                stage.color = cols[i]
        
    
    def set_random_parameters(self):
        """ This is dependent on if one has sampled from the tree already.
            I.e., if a probablity is already set for an edge, it 
            should not be overwritten.
        """
        

        # Check if the node is part part of a context (stage?) 
        # if so we may overwrite probs. Otherwise, generate new ones.
        for node in self.tree.nodes():
            if len(node) == self.p:
                continue
            lev = len(node)

            children = self.tree.successors(node)
            probs = np.random.dirichlet([1] * self.cards[len(node)+1])

            for i, ch in enumerate(children):
                stage = self.get_stage(node)

                if stage != None:
                    prob = stage.probs[i]
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

        ## All the levels of the firs variable
        tovisit = [(i,) for i in range(self.cards[1])]

        while len(tovisit) != 0:
            # Visit/create node in the tree
            node = tovisit.pop()
            lev = len(node)
            fr = node[:lev-1]
            to = node
            if not self.tree.has_edge(fr,to):
                self.tree.add_edge(fr, to) # check if exists first
            else:                
                pass
            self.tree.nodes[to]["label"] = to[-1]
            # Add more nodes to visit
            if lev < self.p:
                #np.random.dirichlet([1] * lev)
                for i in range(self.cards[lev + 1]):
                    tovisit.append(to + (i,))

    def to_minimal_context_graphs(self):
        """ This returns a sequence of minimal context graphs (minimal I-maps).
        """

        #csi_rels = self.csi_relations()
        #dags = get_minimal_dag_imaps(csi_rels)

        pass

    def likelihood(self, data):
        pass

    def to_csv(self):
        pass
    
    def read_csv(self,filename):
        pass
    
    def csi_relations(self):
        """ Returns all the context specific indepencende (CSI) relations.
            These should normally be thinned out using absorption, and then we would extract
            the minmal contexts based on that.
        """
        csi_rels = {}
        # print(self.stages)
        for key, stage_list in self.stages.items():
            # print(stage_list)
            for stage in stage_list:
                #       print(stage)
                csi_rel = stage.to_csi()

                if csi_rel.context not in csi_rels.keys():  # should be a list
                    csi_rels[csi_rel.context] = [csi_rel]
                else:
                    csi_rels[csi_rel.context].append(csi_rel)

        #print(set([ str(k)  for k in csi_rels.keys()]))

        return csi_rels

    def minimal_contexts(self):
        """ Returns the minimal contexts.
        """
        pass

    def sample(self, n):
        """Draws n random samples from the CStree.
            Dymanocally generates nodes in the underlying tree
            and associated parameters on the fly in order to avoid
            creating the whole tree, which is O(2^p), just to sample data.

        Args:
            n (int): number of random samples.
        """
        
        if self.tree is None:
            self.tree = nx.DiGraph()        
        xs = []
        
        for _ in range(n):
            node = ()
            x = []
            while len(x) < self.p:
                #print(node, x)
                # while self.tree.out_degree(node) != 0:
                if (node not in self.tree) or len(self.tree.out_edges(node)) == 0:
                    lev = len(node)
                    edges = [(node, node + (ind,)) for ind in range(self.cards[lev+1])]
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
                    
     
                    edges = list(self.tree.out_edges(node)) # Hope this gets in the right order
                    #print(edges)
                    # Set parameters
                    for i, e in enumerate(edges):
                        self.tree[e[0]][e[1]]["cond_prob"] = probs[i]
                        self.tree[e[0]][e[1]]["label"] = round(probs[i], 2)
                        self.tree.nodes[e[1]]["label"] = e[1][-1]
                        self.tree[e[0]][e[1]]["color"] = color
                        self.tree.nodes[e[0]]["color"] = color
 
                edges = list(self.tree.out_edges(node))
                #print(self.tree[()][(0,)]["cond_prob"])
                probabilities = [self.tree[e[0]][e[1]]["cond_prob"]
                                for e in edges]
                #print(probabilities)
                #elements = [str(e[1]) for e in edges]
                # ind is the index or the outcome of the sampled variable
                vals = len(edges)
                #print(vals)
                ind = np.random.choice(len(edges), 1, p=probabilities)[0]
                node = edges[ind][1] # This is a typle like (0,1,1,0)
                x.append(node[-1]) # Take the last element, 0, above
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
            self.set_random_parameters()

        return plot(self.tree)
        #agraph = nx.nx_agraph.to_agraph(self.tree)
        #agraph.layout("dot")
        #return agraph
        # agraph.draw(filename)



class CI_relation:
    def __init__(self, a, b, sep) -> None:
        self.a = a
        self.b = b
        self.sep = sep
        pass

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
        return "{} ⊥ {}".format(s1, s2)


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
    def __init__(self, list_repr) -> None:
        self.level = len(list_repr)
        self.list_repr = list_repr
        self.color = None

    def __hash__(self) -> int:
        return hash(tuple([tuple(i) for i in self.list_resp]))

    def __contains__(self, node):

        for i, val in enumerate(self.list_repr):
             # Must chec if list
            if (type(val) is list) and (node[i] not in val):
                return False
            if (type(val) is int) and (node[i] != val):                
                return False
            
        return True

    def to_df(self, columns):
        import pandas as pd
        
        d = {}
        
        for i in range(len(columns)):
            if i < len(self.list_repr):
                if type(self.list_repr[i]) == list:
                    d[columns[i]] = ["*"]
                else:
                    d[columns[i]] = [self.list_repr[i]]
            else:
                d[columns[i]] = ["-"]
                
                
        df = pd.DataFrame(d, columns=columns)
        
        return df

    def set_random_params(self, cards):
        self.probs = np.random.dirichlet([1] * cards[self.level]) # Need to fix this

    def from_df(self):
        pass
        

    def to_csi(self):
        sepseta = set()
        cond_set = set()
        context = {}
        sepsetb = {self.level+1}

        for i, el in enumerate(self.list_repr):
            if type(el) is list:
                sepseta.add(i+1)
            else:
                context[i+1] = el

        ci = CI_relation(sepseta, sepsetb, cond_set)
        context = Context(context)
        return CSI_relation(ci, context)

    def intersects(self, stage):
        """ Checks if tha paths of two stages intersect.
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
        return str(self.list_repr) + "; probs: " + str(self.probs)


class Context:
    def __init__(self, context: dict) -> None:
        self.context = context

    def __str__(self) -> str:
        context_str = ""
        for key, val in self.context.items():
            context_str += "X{}={}, ".format(key, val)
        if context_str != "":
            context_str = context_str[:-2]
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

    def __init__(self, ci: CI_relation, context: Context) -> None:
        self.ci = ci
        self.context = context

    def to_stages(self):
        """Returns a list of stages defining the CSI relation.
        """
        stages = []
        pass

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

        return itertools.product(*vals)

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


def sample_random_stage(cards: list, level: int) -> Stage:
    # The las level cannot contain stages
    
    #if level >= len(cards):
    #    return None
    vals = [None]*len(cards[:level])
    # Just to make sure not all variables are context variables.
    while not any(map(lambda x: type(x) is list, vals)):
        for i, j in enumerate(cards[:level]):
            b = np.random.randint(2)
            if b == 0:
                vals[i] = np.random.randint(cards[i])
            if b == 1:
                vals[i] = list(range(cards[i]))

    s = Stage(vals)
    
    
    #s.probs = np.random.dirichlet([1] * cards[level]) # Need to fix this
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


def csi_relations_to_dags(csi_relations, causal_order):

    p = causal_order.p
    graphs = {context: None for context in csi_relations}
    for context, csis in csi_relations.items():
        #print("\nContext: {}".format(context))
        adjmat = np.zeros(p*p).reshape(p, p)

        for j in range(1, p+1):
            for i in range(1, j):
                # This will anyway be disregarded in the matrix slice?
                if (i in context) | (j in context):
                    continue
                # create temp CI relation to compare with
                a = {i}
                b = {j}
                s = {k for k in range(1, j) if (
                    k != i) and (k not in context)}
                ci_tmp = CI_relation(a, b, s)

                #print("checking ci: {}".format(ci_tmp))
                # Check is the ci is in some of the cis of the context.
                # 1. i<j
                # 2. no edge if Xi _|_ Xj | Pa1:j \ i

                cis = []
                for csi in csis:
                    cis += [csi.ci]
                    cis += decomposition(csi.ci)
                    cis += weak_union(csi.ci)

                if ci_tmp in cis:
                    #print("No edprint(s1)ge ({},{})".format(i,j))
                    adjmat[i-1, j-1] = 0
                else:
                    adjmat[i-1, j-1] = 1

        context_els = {i-1 for i in context.context.keys()}
        allels = np.array(causal_order.order)-1
        inds = sorted(set(allels) - context_els)
        adjmat = adjmat[np.ix_(inds, inds)]

        graph = nx.from_numpy_array(adjmat, create_using=nx.DiGraph())
        labels = {}
        for i, j in enumerate(inds):
            labels[i] = j+1
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


def sample_cstree(p: int) -> CStree:
    """
       Sample a random CStree with given cardinalities.
       Since the tree is sampled the order shouldn't matter?

    Args:
        cardinalities (list): cardinalities of the variables.

    Returns:
        CStree: a CStree.
    """
    co = CausalOrder(range(1, p+1))
    ct = CStree(co)
    cards = [2] * p
    ct.set_cardinalities([None] + cards)

    stages = {}
    for key, val in enumerate(cards): # not the last level

        stages[key] = []
        for i in range(key):  # Number of trees incrases as O(p*level)
            if np.random.randint(2): 
                s = sample_random_stage(cards, level=key)
                #s.set_random_params(cards)

                if all(map(lambda x: x.intersects(s) is False, stages[key])):
                    stages[key].append(s)

    ct.add_stages(stages)
    
    #ct.set_random_stage_parameters()
    
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
    
    co = CausalOrder([x[0] for x in df.columns])
    
    cards = [int(x[1]) for x in df.columns]
    
    stages = {i:[] for i in range(len(cards)+1)}
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
    
    cstree.add_stages(stages)

    #cstree.set_random_stage_parameters()
    
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


def weak_union(ci):

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
