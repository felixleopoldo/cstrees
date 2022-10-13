
from matplotlib.style import context
import networkx as nx
import numpy as np
import matplotlib

import itertools


class CStree(nx.Graph):
    """ Naive implementation of a CStree for testing purposes and sanity checks.

        One of the main problems is to implement this efficiently, avoiding 
        the O(2^p) space complexity.

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
        self.p = len(causal_order)

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
        self.stage_probs = {key: [None]*len(val)
                            for key, val in stages.items()}

    def get_stage(self, level: int):
        """ Get all the stages in one level.

        Args:
            level (int): A level corresponds to variable in the causal ordering.
        """
        pass

    def set_random_parameters(self):
        # Set stage prbabilities

        cols = ["red", "blue", "green", "purple"]
        self.colors = {key: cols[:len(val)]
                       for key, val in self.stages.items()}
        for lev, stages in self.stages.items():
            for i, stage_dict in enumerate(stages):
                probs = np.random.dirichlet([1] * self.cards[lev+1])
                self.stage_probs[lev][i] = probs

        # Check if the node is part part of a context
        # if so we may overwrite probs. Otherwise, generate new ones.
        for node in self.tree.nodes():
            if len(node) == self.p:
                continue
            lev = len(node)

            children = self.tree.successors(node)
            probs = np.random.dirichlet([1] * self.cards[len(node)+1])

            for i, ch in enumerate(children):
                node_stage_no = self.get_stage_no(node)
                if node_stage_no != None:
                    prob = self.stage_probs[lev][node_stage_no][i]
                    self.tree[node][ch]["cond_prob"] = prob
                    self.tree[node][ch]["label"] = round(prob, 2)
                    self.tree[node][ch]["color"] = self.colors[lev][node_stage_no]
                    self.tree.nodes[node]["color"] = self.colors[lev][node_stage_no]
                else:
                    self.tree[node][ch]["cond_prob"] = probs[i]
                    self.tree[node][ch]["label"] = round(probs[i], 2)

    def get_stage_no(self, node):
        lev = len(node)
        for lev, stages in self.stages.items():
            for i, stage_dict in enumerate(stages):
                if node in stage_dict:
                    return i
        return None

    def create_tree(self):
        self.tree = nx.DiGraph()

        tovisit = [(i,) for i in range(self.cards[1])]

        while len(tovisit) != 0:
            # Visit/create node in the tree
            node = tovisit.pop()
            lev = len(node)
            fr = node[:lev-1]
            to = node
            self.tree.add_edge(fr, to)
            self.tree.nodes[to]["label"] = to[-1]
            # Add more nodes to visit
            if lev < self.p:
                np.random.dirichlet([1] * lev)
                for i in range(self.cards[lev + 1]):
                    tovisit.append(to + (i,))

    def to_minimal_context_graphs(self):
        """ This returns a sequence of minimal context graphs (minimal I-maps).
        """

        #csi_rels = self.csi_relations()
        #dags = get_minimal_dag_imaps(csi_rels)

        pass

    def csi_relations(self):
        """ Returns all the context specific indepencende (CSI) relations. 
            These should normally be thinned out using absorption, and then we would extract
            the minmal contexts based on that.
        """
        csi_rels = {}
        for key, stage_list in self.stages.items():
            for stage in stage_list:
                context_path = comp_bit_strings(stage)
                csi_rel = CSI_relation(context_path)
                # print(csi_rel.context)
                # print(csi_rel)
                if csi_rel.context not in csi_rels:
                    csi_rels[csi_rel.context] = csi_rel
                else:
                    print("context already here")

        return csi_rels

    def minimal_contexts(self):
        """ Returns the minimal contexts.
        """
        pass

    def sample(self, n):
        """Draws n random samples from the CStree.

        Args:
            n (int): number of random samples.
        """

        xs = []
        for _ in range(n):
            node = ()
            x = []
            while self.tree.out_degree(node) != 0:
                edges = list(self.tree.out_edges(node))
                probabilities = [self.tree[e[0]][e[1]]["cond_prob"]
                                 for e in edges]
                elements = [str(e[1]) for e in edges]
                ind = np.random.choice(
                    range(len(edges)), 1, p=probabilities)[0]
                node = edges[ind][1]
                x.append(node[-1])
            xs.append(x)
        return np.array(xs)

    def pdf(self, x):
        """Density function exaluated at x

        Args:
            x (array type): a vector.
        """

    def plot(self, filename="cstree.png"):
        agraph = nx.nx_agraph.to_agraph(self.tree)
        agraph.layout("dot")
        agraph.draw(filename)


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


class Context:
    def __init__(self, context: dict) -> None:
        self.context = context 

    def __str__(self) -> str:
        context_str = ""
        for key, val in self.context.items():   
             context_str += "X{}={}, ".format(key, val)
        return context_str

    def __contains__(self, key):
        return key in self.context

    def __getitem__(self, key):
        return self.context[key]
    
    def __hash__(self) -> int:
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

    def __init__(self, path) -> None:
        sepseta = set()
        cond_set = set()
        context = {}#[None]*(len(path)+3)
        sepsetb = {len(path)+1}

        for i, el in enumerate(path):
            if el is False:
                sepseta.add(i+1)
            else:
                context[i+1] = el

        self.ci = CI_relation(sepseta, sepsetb, cond_set)
        self.context = Context(context)

    def to_cstree_paths(cards: list):
        k = len(sepseta)
        vals = []*k
        for i in range(k):
            if sepset[a]:
                pass
        return itertools.product(*vals)

    def __str__(self) -> str:        
        return "{}, {}".format(self.ci, self.context)


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

    return levels


def csi_relations_to_dags(csi_relations, causal_order):
    p = len(causal_order)
    graphs = {context: None for context in csi_relations}
    for context, csi in csi_relations.items():

        adjmat = np.zeros(p*p).reshape(p, p)

        for j in range(1, p+1):
            for i in range(1, j):
                if (i in context) | (j in context):
                    continue
                # create temp CI relation to compare with
                a = {i}
                b = {j}
                s = {k for k in range(1, j) if (
                    k != i) and (k not in context)}
                ci_tmp = CI_relation(a, b, s)

                csi = csi_relations[context]
                # 1. i<j
                # 2. no edge if Xi _|_ Xj | Pa1:j \ i
                if ci_tmp == csi.ci:
                    adjmat[i-1, j-1] = 0
                else:
                    adjmat[i-1, j-1] = 1

        context_els = {i-1 for i in context.context.keys()}
        allels = np.array(causal_order)-1
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


def sample_cstree(cardinalities: list) -> CStree:
    """ 
       Sample a random CStee with given cardinalities. 
       Since the tree is sampled the order shouldn't matter?

    Args:
        cardinalities (list): cardinalirties of the variables.

    Returns:
        CStree: a CStree.
    """
    order = range(len(cardinalities))
    ct = CStree(order)
    return ct
