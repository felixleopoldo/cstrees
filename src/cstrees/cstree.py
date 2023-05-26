import math
from math import comb
from random import uniform
from re import S
import networkx as nx
import numpy as np
import matplotlib
from itertools import chain, combinations

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import pandas as pd
import logging
import sys
import cstrees.scoring as sc
from cstrees.stage import * 
from cstrees.csi_relation import *
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


def plot(graph, layout="dot"):
    """Plots a graph using graphviz.
    
    Args:
        graph (nx.Graph): The graph to plot.
        layout (str, optional): The layout to use. Defaults to "dot".

    """
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout(layout)
    return agraph


class CStree(nx.Graph):
    """ A CStree class. The levels are enumerated from 0,...,p-1. 
    You may provide labels for the lavels, corresponding to the 
    random variable they represent.

    Args:
        cards (list): A list of integers representing the cardinality of each level.
        labels (list, optional): A list of strings representing the labels of each level. Defaults to [0,1,...,p-1].
        
    Example:
        >>> # Figure 1. from (Duarte & Solus 2022)
        >>> import cstrees.cstree as ct
        >>> tree = ct.CStree([2,2,2,2]])
        >>> tree.update_stages({
        >>>     0: [ct.Stage([0]), ct.Stage([1])],
        >>>     1: [ct.Stage([{0, 1}, 0], color="green"), ct.Stage([0, 1]), ct.Stage([1, 1])],
        >>>     2: [ct.Stage([0, {0, 1}, 0], color="blue"),
        >>>         ct.Stage([0, {0, 1}, 1], color="orange"),
        >>>         ct.Stage([1, {0, 1}, 0], color="red"),
        >>>         ct.Stage([1, 1, 1]),
        >>>         ct.Stage([1, 0, 1])]  
        >>> })
        >>> tree.sample_stage_parameters()
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
      
        #self.colors = list(set(mcolors.cnames.keys()))
        self.colors = ['blueviolet', 'orange', 'navy', 'rebeccapurple', 'darkseagreen', 'darkslategray', 'lightslategray', 'aquamarine', 'lightgoldenrodyellow', 'cornsilk', 'azure', 'chocolate', 'red', 'darkolivegreen', 'chartreuse', 'turquoise', 'olive', 'crimson', 'goldenrod', 'orchid', 'firebrick', 'lawngreen', 'deeppink', 'wheat', 'teal', 'mediumseagreen', 'peru', 'salmon', 'palegreen', 'navajowhite', 'yellowgreen', 'mediumaquamarine', 'darkcyan', 'dodgerblue', 'brown', 'powderblue', 'mistyrose', 'violet', 'darkslategrey', 'midnightblue', 'aliceblue', 'dimgrey', 'palegoldenrod', 'black', 'darkgrey', 'olivedrab', 'linen', 'lightblue', 'thistle', 'greenyellow', 'indianred', 'khaki', 'lightslategrey', 'slateblue', 'purple', 'deepskyblue', 'magenta', 'yellow', 'ivory', 'darkorchid', 'mediumpurple', 'snow', 'dimgray', 'palevioletred', 'darkslateblue', 'sandybrown', 'lightgray', 'lemonchiffon', 'gray', 'silver', 'aqua', 'tomato', 'lightyellow', 'seagreen', 'darkmagenta', 'beige', 'cornflowerblue', 'peachpuff', 'ghostwhite', 'cyan', 'lightcoral', 'hotpink', 'lightpink', 'lightskyblue', 'slategrey', 'tan', 'oldlace', 'steelblue', 'springgreen', 'fuchsia', 'lime', 'papayawhip', 'mediumblue', 'mediumspringgreen', 'darkorange', 'lightgreen', 'blue', 'slategray', 'white', 'saddlebrown', 'mediumturquoise', 'paleturquoise', 'darkblue', 'plum', 'lightseagreen', 'lightgrey', 'blanchedalmond', 'lavenderblush', 'darkkhaki', 'gainsboro', 'lightsalmon', 'darkturquoise', 'moccasin', 'darkgoldenrod', 'mediumorchid', 'honeydew', 'mediumslateblue', 'maroon', 'forestgreen', 'darkgray', 'floralwhite', 'darkgreen', 'lightcyan', 'darksalmon', 'pink', 'royalblue', 'sienna', 'green', 'orangered', 'bisque', 'antiquewhite', 'rosybrown', 'whitesmoke', 'darkred', 'burlywood', 'skyblue', 'mediumvioletred', 'mintcream', 'limegreen', 'lightsteelblue', 'grey', 'coral', 'indigo', 'gold', 'cadetblue']
        #random.shuffle(self.colors)
        self.color_no = 0
        self.stages = {i: [] for i in range(self.p)}


    def update_stages(self, stages: dict):
        """ Update the stages of the CStree.
        
        Args:
            stages (dict): A dictionary of stages. The keys are the levels, and the values are lists of stages.
            
        Example:
            >>> import cstrees.cstree as ct
            >>> tree = ct.CStree([2,2,2,2]])
            >>> tree.update_stages({
            >>>     0: [ct.Stage([0]), ct.Stage([1])],
            >>>     1: [ct.Stage([{0, 1}, 0], color="green"), ct.Stage([0, 1]), ct.Stage([1, 1])],
            >>>     2: [ct.Stage([0, {0, 1}, 0], color="blue"),
            >>>         ct.Stage([0, {0, 1}, 1], color="orange"),
            >>>         ct.Stage([1, {0, 1}, 0], color="red"),
            >>>         ct.Stage([1, 1, 1]),
            >>>         ct.Stage([1, 0, 1])]  
            >>> })

        """
        self.stages.update(stages)
        self.stages[-1] = [Stage([], color="black")]

        # Add support for the set format too
        # self.stage_probs = {key: [None]*len(val)
        #                    for key, val in stages.items()}

    def get_stage(self, node: tuple):
        """ Get the stage of a node in the cstree.

        Args:
            node (tuple): A node in the CStree.
        Example:
            >>> # tree is the fig. 1 CStree
            >>> st = tree.get_stage([0, 0])
            >>> print(st)
            [{0, 1}, 0]; probs: [0.57628561 0.42371439]; color: green
        """
        assert(self.stages is not None)
        lev = len(node)-1

        stage = None
        if lev in self.stages:
            for s in self.stages[lev]:
                # print(s)
                if node in s:
                    stage = s
                    break

        if stage is None:
            print("No stage found for {}".format(node))

        assert(stage is not None)
        return stage

    def to_df(self):
        """ Converts the CStree to a pandas dataframe.
        
        Returns:
            df (pd.DataFrame): A pandas dataframe with the stages of the CStree.
            
        Example:
            >>> tree.to_df()
            0  1  2  3
            2  2  2  2
            0  -  -  -
            1  -  -  -
            *  0  -  -
            0  1  -  -
            1  1  -  -
            0  *  0  -
            0  *  1  -
            1  *  0  -
            1  1  1  -
            1  0  1  -
            -  -  -  -    
        """

        # cardinalities header
        d = {self.labels[i]: [c] for i, c in enumerate(self.cards)}
        df = pd.DataFrame(d, columns=self.labels)

        for l, stages in self.stages.items():
            if l == -1:
                continue
            for s in stages:
                dftmp = s.to_df(self.labels)
                df = pd.concat([df, dftmp])
        df.reset_index(drop=True, inplace=True)
        
        return df

    def sample_stage_parameters(self, alpha=1):
        """ Set the parameters of the stages of the CStree to be random samples 
        from a Dirichlet distribution with hyper parameter alpha.
        
        Args:
            alpha (float): The hyper parameter for the Dirichlet distribution.
        
        Example:
            >>> t = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
            >>> t.sample_stage_parameters()        
        """
        for lev, stages in self.stages.items():
            for i, stage in enumerate(stages):
                probs = np.random.dirichlet([alpha] * self.cards[lev+1])                
                stage.probs = probs
                if stage.color is None:
                    stage.color = self.colors[i] # Set color from stage if possible
                
        self.set_tree_probs()

    def estimate_stage_parameters(self, data, method="BDeu", alpha_tot=1):
        """ Estimate the parameters of the stages of the CStree under a Dirichlet model.
        Args:
            data (pd.DataFrame): A pandas dataframe with the data.
            method (str): The method to use for estimating the parameters. 
                Currently only "BDeu" is implemented.
            alpha_tot (float): The total alpha value to use for the Dirichlet model.
        Example:
            >>> t = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
            >>> t.estimate_stage_parameters(x, alpha_tot=1.0, method="BDeu")
        
        """
        
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
        #print("nodes: ", self.tree.nodes())
        for node in self.tree.nodes():
            if len(node) == self.p:
                continue

            children = sorted(list(self.tree.successors(node)))
            probs = np.random.dirichlet([alpha] * self.cards[len(node)])
            # Seems like a stage at level l holds the probtable for the variabel at level l+1.
            for i, ch in enumerate(children):
                #print("i {i}: {node} -> {ch}".format(i=i, ch=ch, node=node))
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


    def get_stage_no(self, node):
        """ Get the stage number of a node in a level of the cstree.
        This is used when coloring the nodes.
            
        Args:
            node (tuple): A node in the CStree.    
        """
        lev = len(node)
        for stages in self.stages[lev]:
            for i, stage_dict in enumerate(stages):
                if node in stage_dict:
                    return i
        return None

    def create_tree(self):
        if self.tree is None:
            self.tree = nx.DiGraph()

        # All the levels of the first variable.
        tovisit = [(i,) for i in range(self.cards[0])]
        #print("tovisit: ", tovisit)
        while len(tovisit) != 0:
            # Visit/create node in the tree
            node = tovisit.pop()
            #print("node: ", node)
            lev = len(node)-1  # added -1
            fr = node[:lev]
            to = node
            #print("{} -> {} ".format(fr, to))

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

    # Here is the only place we need labels/orders.
    def to_minimal_context_graphs(self):
        """ This returns a sequence of minimal context graphs (minimal I-maps).
        
        Example:
            >>> # tree is the Figure 1 CStree
            >>> gs = tree.to_minimal_context_graphs()
            >>> for key, graph in gs.items():
            >>>     print("{}: Edges {}".format(key, graph.edges()))
            X0=0: Edges [(1, 2), (2, 3)]
            X1=0: Edges [(0, 3), (2, 3)]
            X2=0: Edges [(0, 1), (0, 3)]            

        """
        
        minl_csis_by_context = self.to_minimal_context_csis()
        
        cdags = csi_relations_to_dags(
            minl_csis_by_context, self.p, labels=self.labels)

        return cdags

    def to_minimal_context_csis(self):
        """ This returns a sequence of minimal context graphs (minimal I-maps).
        
        Example:
            >>> # tree is the Figure 1 CStree
            >>> gs = tree.to_minimal_context_graphs()
            >>> for key, graph in gs.items():
            >>>     print("{}: Edges {}".format(key, graph.edges()))
            X0=0: Edges [(1, 2), (2, 3)]
            X1=0: Edges [(0, 3), (2, 3)]
            X2=0: Edges [(0, 1), (0, 3)]            

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
                
        return minl_csis_by_context

    def csi_relations_per_level(self, level="all"):

        return {l: [s.to_csi() for s in stages] for l, stages in self.stages.items()}

    def csi_relations(self, level="all"):
        """ Returns all the context specific indepencende (CSI) relations.
        These should normally be thinned out using absorption, and then we would extract
        the minmal contexts based on that.

        Examples:
            >>> rels = tree.csi_relations()
            >>> for cont, rels in rels.items():
            >>>     for rel in rels:
            >>>         print(rel)
            X0 ⊥  X2, X1=0
            X1 ⊥  X3, X0=0, X2=0
            X1 ⊥  X3, X0=0, X2=1
            X1 ⊥  X3, X0=1, X2=0
        """
        csi_rels = {}
        # print(self.stages)
        for key, stage_list in self.stages.items():
            #print("{} {}".format(key, stage_list))
            for stage in stage_list:
                #       print(stage)
                if stage.is_singleton():
                    continue # As these dont encode any independence relations.
                csi_rel = stage.to_csi()

                if csi_rel.context not in csi_rels.keys():  # should be a list
                    csi_rels[csi_rel.context] = [csi_rel]
                else:
                    csi_rels[csi_rel.context].append(csi_rel)

        return csi_rels

    def sample(self, n):
        """Draws n random samples from the CStree.
        When singletons are allowed (maybe deprecated) it dynamically generates nodes in the underlying tree
        and associated parameters on the fly in order to avoid
        creating the whole tree, which is O(2^p), just to sample data.

        Args:
            n (int): number of random samples.
            
        Returns:
            pandas.DataFrame: A pandas dataframe with the sampled data. 
            The first row contains the labels of the variables and the second row contains the cardinalities.
        Examples:
            >>> df = tree.sample(5)
            >>> print(df)
            0  1  2  3
            0  2  2  2  2
            1  0  0  1  1
            2  0  0  0  0
            3  0  0  0  1
            4  0  0  1  1
            5  0  0  0  1
        """

        if self.tree is None:
            print("Creating tree on the fly while sampling to save space when allowing for singleton stages.")
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
                    print("should not happen")
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

    def plot(self, fill=False):
        """Plot the CStree. Make sure to set the parameters first.
        
        Args:
            fill (bool): If True, the tree is filled with parameters.
        
        Examples:
            >>> tree.sample_stage_parameters()
            >>> agraph = tree.plot()
            >>> agraph.draw("cstree.png")
        """
    
        if fill or (self.tree is None):
            self.create_tree()
            # self.set_random_parameters()

        return plot(self.tree)
        #agraph = nx.nx_agraph.to_agraph(self.tree)
        # agraph.layout("dot")
        # return agraph
        # agraph.draw(filename)


def sample_cstree(cards: list, max_cvars: int, prob_cvar: int, prop_nonsingleton: float) -> CStree:
    """
       Sample a random CStree with given cardinalities.

    Args:
        cards (list): cardinalities of the levels.  
        max_cvars (int): maximum number of context variables.
        prob_cvar (float): probability a potential context variable (in the algorithm) will a context variable.
        prop_nonsingleton (float): proportion of non-singleton stages.
                
    Returns:
        CStree: A CStree.
    Examples:
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)

    """
    p = len(cards)
    ct = CStree(cards)

    stagings = {}
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
            stagings[level] = []

            if b == 0:
                print(b)
                # it should be the whole space to begin with
                stage_restr = stage_space.pop(0)
                new_stage = sample_stage_restr_by_stage(
                    stage_restr, mc, 1.0, cards)
                stagings[level].append(new_stage)
            continue  # cant add anything anyway so just go to the next level.

        # print(mc)
        #max_n_stages = comb(level+1, mc) * (cards[level]**mc)
        # logging.debug("Max # stages with {} context variables: {}".format(mc, max_n_stages))
        stagings[level] = []
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
                stagings[level].append(new_stage)
            #print("proportion left")
            #print(space_left / full_state_space_size)

    stagings[-1] = [Stage([], color="black")]
    ct.update_stages(stagings)

    return ct


def df_to_cstree(df):
    """ Convert a dataframe to a CStree. The dataframe should have the following format:
    The labels should be the level labels.
    The first row should be the cards, the second row should be the first stage, the third row the second stage etc.
    
    Args:
        df (pd.DataFrame): The dataframe to convert.
    Example:
        >>> df = tree.()
        >>> print(df)
        >>> t2 = ct.df_to_cstree(df)
        >>> df2 = t2.()
        >>> print("The same tree:")
        >>> print(df)
           0  1  2  3  
        0  2  2  2  2
        0  0  -  -  -
        0  1  -  -  -
        0  *  0  -  -
        0  0  1  -  -
        0  1  1  -  -
        0  0  *  0  -
        0  0  *  1  -
        0  1  *  0  -
        0  1  1  1  -
        0  1  0  1  -
        1  2  3  4
        The same tree:
           0  1  2  3  
        0  2  2  2  2
        0  0  -  -  -
        0  1  -  -  -
        0  *  0  -  -
        0  0  1  -  -
        0  1  1  -  -
        0  0  *  0  -
        0  0  *  1  -
        0  1  *  0  -
        0  1  1  1  -
        0  1  0  1  -
    """
    cards = cards = df.iloc[0].values  # [int(x[1]) for x in df.columns]

    stagings = {i: [] for i in range(len(cards))}
    stagings[-1] = [Stage([])]
    cstree = CStree(cards)
    cstree.labels = df.columns

    for row in df[1:].iterrows():
        stage_list = []
        for level, val in enumerate(row[1]):
            if level == len(cards): 
                break
            if val == "*":
                stage_list.append(set(range(cards[level])))
            elif val == "-":
                s = Stage(stage_list)
                stagings[level-1].append(s)
                break
            else:
                stage_list.append(int(val))

    cstree.update_stages(stagings)

    return cstree

