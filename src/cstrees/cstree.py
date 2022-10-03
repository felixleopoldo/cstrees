import networkx as nx
import numpy as np


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

    """

    def __init__(self, causal_order, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)
        self.p = len(causal_order)

    def set_cardinalities(self, cards):
        self.cards = cards

    def add_stages(self, stages: dict):
        """Adds a stage.
        """
        self.stages = stages
        self.stage_probs = {s:[] for s in stages}

    def get_stage(self, level: int):
        """ Get all the stages in one level.

        Args:
            level (int): A level corresponds to variable in the causal ordering.
        """
        pass

    def set_random_parameters(self):
        for node in self.tree.nodes():
            if len(node) == self.p:
                continue
            lev = len(node)
            for i, stage_dict in enumerate(self.stage[lev]):
                if node in stage_dict:
                    if self.stage_probs[lev][i] != []:
                      
            
            children = self.tree.successors(node)
                                
            probs = np.random.dirichlet([1] * self.cards[len(node)+1])
            for i, ch in enumerate(children):
                self.tree[node][ch]["cond_prob"] = probs[i]
                self.tree[node][ch]["label"] = round(probs[i], 2)
                      
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
        pass

    def minimal_contexts(self):
        """ Returns the minimal contexts.
        """
        pass

    def sample(self, n):
        """Draws n random samples from the CStree.

        Args:
            n (int): number of random samples.
        """

        x = np.zeros(self.p)

        x[0] = self.get_stage(0).cond_sample()

        for i in range(self.p):
            x[i] = self.get_stage(i).cond_sample(x[i-1])

        return x

    def pdf(self, x):
        """Density function exaluated at x

        Args:
            x (array type): a vector.
        """


class CSI_relation:

    def __init__(self) -> None:

        pass


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
