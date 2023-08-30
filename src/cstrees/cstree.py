from math import comb
from random import uniform
import networkx as nx
import numpy as np
import pandas as pd

import cstrees.stage as st
from cstrees import csi_relation

import logging
import sys
from importlib import reload  # Not needed in Python 2

reload(logging)
FORMAT = '%(filename)s:%(funcName)s (%(lineno)d):  %(message)s'
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format=FORMAT)
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

def write_minimal_context_graphs_to_files(context_dags, prefix="mygraphs"):
    for key, val in context_dags.items():
        agraph = nx.nx_agraph.to_agraph(val)
        agraph.layout("dot")
        agraph.draw(prefix+str(key) + ".png", args='-Glabel="'+str(key)+'"   ')

def plot(graph, layout="dot"):
    """Plots a graph using graphviz.

    Args:
        graph (nx.Graph): The graph to plot.
        layout (str, optional): The layout to use. Defaults to "dot".

    """
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout(layout)
    return agraph


class CStree:
    """ A CStree class. The levels are enumerated from 0,...,p-1.
    You may provide labels for the lavels, corresponding to the
    random variable they represent.

    Args:
        cards (list): A list of integers representing the cardinality of each level.
        labels (list, optional): A list of strings representing the labels of each level. Defaults to [0,1,...,p-1].

    Example:
        >>> # Figure 1. from (Duarte & Solus 2022)
        >>> import cstrees.cstree as ct
        >>> import cstrees.stage as st
        >>> tree = ct.CStree([2, 2, 2, 2])
        >>> tree.update_stages({
        >>>     0: [st.Stage([0]), st.Stage([1])],
        >>>     1: [st.Stage([{0, 1}, 0], color="green"), st.Stage([0, 1]), st.Stage([1, 1])],
        >>>     2: [st.Stage([0, {0, 1}, 0], color="blue"),
        >>>         st.Stage([0, {0, 1}, 1], color="orange"),
        >>>         st.Stage([1, {0, 1}, 0], color="red"),
        >>>         st.Stage([1, 1, 1]),
        >>>         st.Stage([1, 0, 1])]
        >>> })
        >>> tree.sample_stage_parameters()
    """

    def __init__(self, cards, labels=None):
        self.tree = None
        self.stages = None
        self.cards = cards
        self.p = len(cards)
        if labels is None:
            self.labels = list(range(self.p))
        else:
            self.labels = labels

        #self.colors = list(set(mcolors.cnames.keys()))
        self.colors = ['peru','blueviolet', 'orange', 'navy', 'rebeccapurple', 'darkseagreen', 'aquamarine', 'goldenrodyellow', 'cornsilk', 'azure', 'chocolate', 'red', 'darkolivegreen', 'chartreuse', 'turquoise', 'olive', 'crimson', 'goldenrod', 'orchid', 'firebrick', 'lawngreen', 'deeppink', 'wheat', 'teal', 'mediumseagreen',  'salmon', 'palegreen', 'navajowhite', 'yellowgreen', 'mediumaquamarine', 'darkcyan', 'dodgerblue', 'brown', 'powderblue', 'mistyrose', 'violet', 'darkslategrey', 'midnightblue', 'aliceblue', 'dimgrey', 'palegoldenrod', 'black', 'darkgrey', 'olivedrab', 'linen', 'lightblue', 'thistle', 'greenyellow', 'indianred', 'khaki', 'lightslategrey', 'slateblue', 'purple', 'deepskyblue', 'magenta', 'yellow', 'ivory', 'darkorchid', 'mediumpurple', 'snow', 'dimgray', 'palevioletred', 'darkslateblue', 'sandybrown', 'lightgray', 'lemonchiffon', 'gray', 'silver', 'aqua', 'tomato', 'lightyellow', 'seagreen', 'darkmagenta', 'beige', 'cornflowerblue', 'peachpuff', 'ghostwhite', 'cyan', 'lightcoral', 'hotpink', 'lightpink', 'lightskyblue', 'slategrey', 'tan', 'oldlace', 'steelblue', 'springgreen', 'fuchsia', 'lime', 'papayawhip', 'mediumblue', 'mediumspringgreen', 'darkorange', 'lightgreen', 'blue', 'slategray', 'white', 'saddlebrown', 'mediumturquoise', 'paleturquoise', 'darkblue', 'plum', 'lightseagreen', 'lightgrey', 'blanchedalmond', 'lavenderblush', 'darkkhaki', 'gainsboro', 'lightsalmon', 'darkturquoise', 'moccasin', 'darkgoldenrod', 'mediumorchid', 'honeydew', 'mediumslateblue', 'maroon', 'forestgreen', 'darkgray', 'floralwhite', 'darkgreen', 'lightcyan', 'darksalmon', 'pink', 'royalblue', 'sienna', 'green', 'orangered', 'bisque', 'antiquewhite', 'rosybrown', 'whitesmoke', 'darkred', 'burlywood', 'skyblue', 'mediumvioletred', 'mintcream', 'limegreen', 'lightsteelblue', 'grey', 'coral', 'indigo', 'gold', 'cadetblue']
        self.color_no = 0
        self.stages = {i: [] for i in range(self.p)}

    def stage_proportion(self, stage: st.Stage):
        """The proportion of the space this stage represents.
        
        Args:
            cards (list): A list of the cardinalities of the variables in the space.
        
        Returns:
            float: A number between 0 and 1.
            
        Example:
            # assuming all variables are binary
            >>> s = Stage([0, {0, 1}, 1])
            >>> cstree.stage_proportion(s)
            0.25        
        """
        prop = 1
        for i, val in enumerate(stage.list_repr):
            if not isinstance(val, set):
                prop *= 1/self.cards[i]
        
        return prop

    def update_stages(self, stages: dict):
        """ Update the stages of the CStree.

        Args:
            stages (dict): A dictionary of stages. The keys are the levels, and the values are lists of stages.

        Example:
            >>> import cstrees.cstree as ct
            >>> import cstrees.stage as st
            >>> tree = ct.CStree([2, 2, 2, 2]])
            >>> tree.update_stages({
            >>>     0: [st.Stage([0]), st.Stage([1])],
            >>>     1: [st.Stage([{0, 1}, 0], color="green"), st.Stage([0, 1]), st.Stage([1, 1])],
            >>>     2: [st.Stage([0, {0, 1}, 0], color="blue"),
            >>>         st.Stage([0, {0, 1}, 1], color="orange"),
            >>>         st.Stage([1, {0, 1}, 0], color="red"),
            >>>         st.Stage([1, 1, 1]),
            >>>         st.Stage([1, 0, 1])]
            >>> })

        """
        for lev, stage_list in stages.items():
            for stage in stage_list:
                stage.cards = self.cards#[:lev+1] # Or full cards?
            
            
        self.stages.update(stages)
        if -1 not in self.stages:
            self.stages[-1] = [st.Stage([], color="black")]


    def get_stage(self, node: tuple):
        """ Get the stage of a node in the cstree.

        Args:
            node (tuple): A node in the CStree.
        Example:
            >>> # tree is the fig. 1 CStree
            >>> stage = tree.get_stage([0, 0])
            >>> print(stage)
            [{0, 1}, 0]; probs: [0.57628561 0.42371439]; color: green
        """
        assert(self.stages is not None)
        lev = len(node)-1

        stage = None
        if lev in self.stages:
            for s in self.stages[lev]:
                
                if node in s:
                    stage = s
                    break
        
        if stage is None:
            print("No stage found for {}".format(node))

        assert(stage is not None)
        return stage

    def to_df(self, write_probs=False):
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
        """

        # cardinalities header
        d = {self.labels[i]: [c] for i, c in enumerate(self.cards)}
        max_card = max(self.cards)
        
        if write_probs:
            labs = self.labels + ["PROB_"+str(i) for i in range(max_card)]
        else:
            labs = self.labels
        df = pd.DataFrame(d, columns=labs)
            
        for l, stages in self.stages.items():
            for s in stages:                
                dftmp = s.to_df(labs, max_card=max_card, write_probs=write_probs)
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
        import cstrees.scoring as sc
        # Set stage probabilities

        for lev, stages in self.stages.items():
            if lev == self.p-1:
                continue

            stage_counts = sc.counts_at_level(self, lev+1, data)

            for stage in stages:
                probs = sc.estimate_parameters(
                    self, stage, stage_counts, method, alpha_tot)
                stage.probs = probs


    def _set_tree_probs(self, alpha=1):
        """ This is dependent on if one has sampled from the tree already.
            I.e., if a probablity is already set for an edge, it
            should not be overwritten.
        """
        # First create the tree.
        # This has to do with that, if we allow for singleton stages, we need to
        # make sure it is not too big sort of.

        self._create_tree()

        # Check if the node is part part of a context (stage?)
        # if so we may overwrite probs. Otherwise, generate new ones.
        for node in self.tree.nodes():
            if len(node) == self.p:
                continue
            children = sorted(list(self.tree.successors(node)))
            probs = np.random.dirichlet([alpha] * self.cards[len(node)])
            # Seems like a stage at level l holds the probtable for the variabel at level l+1.
            for i, ch in enumerate(children):
                stage = self.get_stage(node)
                
                if (stage != None):  # No singleton stages allowed!
                    prob = stage.probs[i]
                    self.tree[node][ch]["cond_prob"] = prob
                    self.tree[node][ch]["label"] = round(prob, 2)
                    self.tree[node][ch]["color"] = stage.color
                    self.tree.nodes[node]["color"] = stage.color
                else: # This should never happen as all nodes should be part of a stage.
                    self.tree[node][ch]["cond_prob"] = probs[i]
                    self.tree[node][ch]["label"] = round(probs[i], 2)


    def _get_stage_no(self, node):
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

    def _create_tree(self):
        if self.tree is None:
            self.tree = nx.DiGraph()

        # All the levels of the first variable.
        tovisit = [(i,) for i in range(self.cards[0])]
        while len(tovisit) != 0:
            # Visit/create node in the tree
            node = tovisit.pop()
            lev = len(node)-1  # added -1
            fr = node[:lev]
            to = node
            stage = self.get_stage(fr)
            self.tree.add_edge(fr, to)  # check if exists first
            
            if (stage != None):  # No singleton stages allowed!
                if stage.probs is not None:                       
                    prob = stage.probs[to[-1]] # The last digit/element in the node is the variable value
                    self.tree[fr][to]["cond_prob"] = prob
                    self.tree[fr][to]["label"] = round(prob, 2)
                self.tree[fr][to]["color"] = stage.color
                self.tree.nodes[fr]["color"] = stage.color

            if fr == ():
                self.tree.nodes[fr]["label"] = "ø"
            self.tree.nodes[to]["label"] = to[-1]
            # Add more nodes to visit
            if lev < self.p-1:
                for i in range(self.cards[lev + 1]):
                    tovisit.append(to + (i,))

    # Here is the only place we need labels/orders.
    def to_minimal_context_graphs(self):
        """ This returns a sequence of minimal context NetworkX graphs dsadas
        (minimal I-maps).

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
        cdags = csi_relation.csi_relations_to_dags(
            minl_csis_by_context, self.p, labels=self.labels)

        return cdags

    def to_minimal_context_agraphs(self, layout="dot"):
        """This returns a sequence of minimal context DAGs as Graphviz AGraphs.

        Args:
            layout (str, optional): Graphviz engine. Defaults to "dot".

        Returns:
           pygraphviz.agraph.AGraph: pygraphviz graphs
        """

        graphs = self.to_minimal_context_graphs()
        agraphs = {}
        for key, graph in graphs.items():
            agraphs[key] = nx.nx_agraph.to_agraph(graph)
            agraphs[key].layout(layout)

        return agraphs


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
        
        logging.debug("Stages")
        
        for key, val in self.stages.items():
            logging.debug("level {}".format(key))
            for s in val:
                logging.debug(s)
        logging.debug("Getting csi rels per level")
        rels = self.csi_relations_per_level()
        logging.debug("CSI relations per level")
        for key, rel in rels.items():
            
            logging.debug("level {}: ".format(key))
            
            for r in rel:
                logging.debug("the CSI")
                logging.debug(r)
                logging.debug("cards: {}".format(r.cards))
                
        paired_csis = csi_relation._csis_by_levels_2_by_pairs(rels, cards=self.cards) # this could still be per level?
        logging.debug("\n###### Paired_csis")
        logging.debug(paired_csis)

        logging.debug("\n ######### minl cslisist")
        # The pairs may change during the process this is why we have by pairs.
        # However, the level part
        # should always remain the same actually, so may this is not be needed.
        minl_csislists = csi_relation.minimal_csis(paired_csis, self.cards) # this could be by level still?
        logging.debug(minl_csislists)

        logging.debug("\n ############### get minl csis in list format")
        minl_csis = csi_relation._csi_lists_to_csis_by_level(minl_csislists, self.p, labels=self.labels) # this would not be needed
        logging.debug(minl_csislists)
        for key in minl_csislists:
            for pair, val in key.items():
                logging.debug("{}: {}".format(pair, val))

        logging.debug("#### minimal csis")
        minl_csis_by_context = csi_relation._rels_by_level_2_by_context(minl_csis)
        logging.debug(minl_csis_by_context)
        for pair, val in minl_csis_by_context.items():
            for csi in val:
                logging.debug(csi)

        return minl_csis_by_context

    def csi_relations_per_level(self):
        """Get the context specific independence relations per level.

        Returns:
            dict: The CSI relations per level. The keys are the levels, and the values are lists of CSI relations.
        """

        return {l: [s.to_csi() for s in stages] for l, stages in self.stages.items() if l>=0}

    def csi_relations(self, level="all"):
        """ Returns all the context specific indepencende (CSI) relations.
        These should normally be thinned out using absorption, and then we would extract
        the minmal contexts based on that.

        Examples:
            >>> rels = tree.csi_relations()
            >>> for cont, rels in rels.items():
            >>>     for rel in rels:
            >>>         print(rel)
            X0 ⊥ X2, X1=0
            X1 ⊥ X3, X0=0, X2=0
            X1 ⊥ X3, X0=0, X2=1
            X1 ⊥ X3, X0=1, X2=0
        """
        csi_rels = {}

        for key, stage_list in self.stages.items():

            for stage in stage_list:

                if stage.is_singleton():
                    continue # As these dont encode any independence relations.
                csi_rel = stage.to_csi(labels=self.labels)

                if csi_rel.context not in csi_rels.keys():
                    csi_rels[csi_rel.context] = [csi_rel]
                else:
                    csi_rels[csi_rel.context].append(csi_rel)

        return csi_rels

    def sample(self, n):
        """Draws n random samples from the CStree.
        When singletons are allowed (maybe deprecated) it dynamically generates
        nodes in the underlying tree and
        associated parameters on the fly in order to avoid creating the whole
        tree, which is O(2^p), just to sample
        data.

        Args:
            n (int): number of random samples.

        Returns:
            pandas.DataFrame: A pandas dataframe with the sampled data. The first row contains the labels of the
            variables and the second row contains the cardinalities.
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
            print("Creating tree on the fly while sampling to save space "
                  "when allowing for singleton stages.")
            self.tree = nx.DiGraph()
        xs = []

        xs.append(self.cards)  # cardinalities at the first row

        for _ in range(n):
            node = ()
            x = []
            while len(x) < self.p:
                # Create tree dynamically if isnt already crated.
                if (node not in self.tree) or len(self.tree.out_edges(node)) == 0:
                    lev = len(node)-1
                    edges = [(node, node + (ind,))
                             for ind in range(self.cards[lev+1])]

                    self.tree.add_edges_from(edges)

                    # We set the parametres at to whats specifies in the stage.
                    s = self.get_stage(node)
                    color = ""
                    if s is not None: # not singleton
                        probs = s.probs
                        if s.color is None: # Color if the color isnt set already
                            s.color = self.colors[self.color_no]
                            self.color_no += 1

                        color = s.color

                    # Hope this gets in the right order
                    edges = list(self.tree.out_edges(node))

                    # Set parameters
                    for i, e in enumerate(edges):
                        self.tree[e[0]][e[1]]["cond_prob"] = probs[i]
                        self.tree[e[0]][e[1]]["label"] = round(probs[i], 2)
                        self.tree.nodes[e[1]]["label"] = e[1][-1]
                        self.tree[e[0]][e[1]]["color"] = color
                        self.tree.nodes[e[0]]["color"] = color
                        
                        # Coloring the child too                        
                        if len(e[1]) < self.p:
                            child_stage = self.get_stage(e[1])
                            self.tree.nodes[e[1]]["color"] = child_stage.color
                        
                        # We should color all children here.

                edges = list(self.tree.out_edges(node))

                probabilities = [self.tree[e[0]][e[1]]["cond_prob"]
                                 for e in edges]

                # ind is the index or the outcome of the set_d variable
                vals = len(edges)

                ind = np.random.choice(len(edges), 1, p=probabilities)[0]
                node = edges[ind][1]  # This is a typle like (0,1,1,0)
                x.append(node[-1])  # Take the last element, 0, above

            xs.append(x)

        df = pd.DataFrame(xs)
        df.columns = self.labels
        return df

    def plot(self, full=False):
        """Plot the CStree. Make sure to set the parameters first.

        Args:
            fill (bool): If True, the tree is filled with parameters.

        Examples:
            >>> tree.sample_stage_parameters()
            >>> agraph = tree.plot()
            >>> agraph.draw("cstree.png")
        """

        # If no samples has been drawn, create the full tree.
        if full:# or (self.tree is None):            
            self._create_tree()
        else:
            print("Use plot(full=True) to draw the full tree.")

        if self.tree is None:
            return
        agraph = plot(self.tree)
        #agraph.node_attr["shape"] = "circle"
        
        ###############################################################
        ### This is a hack to plot labels. Just an invisible graph. ###
        for i, lab in enumerate(self.labels):
            agraph.add_node(lab, color="white")

        agraph.add_node("-", color="white")
        agraph.add_edge("-",
                        self.labels[0],
                        color="white")

        for i, lab in enumerate(self.labels[:-1]):
            agraph.add_edge(self.labels[i],
                            self.labels[i+1],
                            color="white")

        agraph.layout("dot")
        return agraph


def sample_cstree(cards: list, max_cvars: int, prob_cvar: int,
                  prop_nonsingleton: float) -> CStree:
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
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.to_df()
            0	1	2	3
        0	2	2	2	2
        1	0	-	-	-
        2	1	-	-	-
        3	1	*	-	-
        4	0	*	-	-
        5	*	*	0	-
        6	*	*	1	-
    """
    p = len(cards)
    ct = CStree(cards)

    stagings = {}
    for level, val in enumerate(cards[:-1]):  # not the last level
        tmpstage =  st.Stage([set(range(cards[l])) for l in range(level+1)], cards=cards)
        
        stage_space = [tmpstage]

        full_stage_space_size = stage_space[0].size()        
        singleton_space_size = full_stage_space_size

        # Need to adjust the max_cvars for the low levels.
        mc = max_cvars
        if level < mc:
            mc = level + 1  # should allow for "singleton" stages at level 0 as this
            # this correponds to the half of the space.
        # in general, when the level is the liming factor, the number of stages
        # should be as many as possible. Hence, mc = level +1
        # It should not make abig difference. But in this way, we actually show
        # all the stages expicitly, even when saving to file, without risking
        # to blow up the space.

        minimal_stage_size = np.prod(sorted(cards[:level+1])[:-mc], dtype=int) # take the product of all cards, expept for the mc largest ones

        # special case when the granularity is to coarse. Randomly add anyway?
        # Setting the upper boudn likje this may generalize. However, the stage
        # should not always be accepted.. prop_nonsingleton =
        # max(prop_nonsingleton, minimal_stage_size / full_state_space_size)

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
                new_stage = st.sample_stage_restr_by_stage(
                    stage_restr, mc, 1.0, cards)
                stagings[level].append(new_stage)
            continue  # cant add anything anyway so just go to the next level.
        stagings[level] = []

        while (1 - (singleton_space_size / full_stage_space_size)) < prop_nonsingleton:
            colored_size_old = full_stage_space_size - singleton_space_size
            # Choose randomly a stage space
            
            space_int = np.random.randint(len(stage_space))
            stage_restr = stage_space.pop(space_int)
            new_stage = st.sample_stage_restr_by_stage(
                stage_restr, mc, prob_cvar, cards)

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


    stagings[-1] = [st.Stage([], color="black")]

    # Color each stage in the optimal staging. Singletons are black.
    # This should be done somewhere else probably.
    colors = ['peru','blueviolet', 'orange', 'navy', 'rebeccapurple', 'darkseagreen',
              'darkslategray', 'lightslategray', 'aquamarine',
              'lightgoldenrodyellow', 'cornsilk', 'azure', 'chocolate',
              'red', 'darkolivegreen']

    for level, staging in stagings.items():
        for i, stage in enumerate(staging):
            if (level==-1) or ((level>0) and all([isinstance(i, int) for i in stage.list_repr])):
                stage.color = "black"
            else:
                stage.color = colors[i]
                
    ct.update_stages(stagings)

    return ct


def df_to_cstree(df, read_probs=True):
    """ Convert a dataframe to a CStree. The dataframe should have the following format:
    The labels should be the level labels.
    The first row should be the cards, the second row should be the first stage, the third row the second stage etc.

    Args:
        df (pd.DataFrame): The dataframe to convert.
    Example:
        >>> df = tree.to_df()
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
    collabs= list(df.columns)
    
    # Gets the number of varaibles. 
    # If no probs are given, then the number of variables is 
    # the number of columns.
    if "PROB_0" in collabs:
        has_probs = True    
        nvars = collabs.index("PROB_0")
    else:
        has_probs = False
        nvars = len(collabs)
    
    cards = df.iloc[0,:nvars].values  # [int(x[1]) for x in df.columns]
    
    stagings = {i: [] for i in range(-1,len(cards))}
        
    cstree = CStree(cards)
    cstree.labels = list(df.columns[:nvars])
    
    color_ind = 0
    for row in df.iloc[1:,:nvars].iterrows():
        stage_list = []
        for level, val in enumerate(row[1]):
            if level == len(cards):
                break
            if val == "*":
                stage_list.append(set(range(cards[level])))
            elif val == "-":
                # Reached stop mark "-", so create a stage of stage_list.
                s = st.Stage(stage_list)
                
                if s.size() == 1: # Singleton stage? Or maybe root stage?.
                    s.color = "black"
                else:
                    s.color = cstree.colors[color_ind]
                    color_ind = (color_ind + 1) % len(cstree.colors)
                # Now we reed the probabilities, if there are any                
                if has_probs:
                    s.probs = df.iloc[row[0], nvars:].values

                stagings[level-1].append(s)

                break
            else:
                stage_list.append(int(val))
        

    cstree.update_stages(stagings)
    
    if has_probs:
        cstree._set_tree_probs()

    return cstree

