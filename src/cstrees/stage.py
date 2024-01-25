import itertools
import random 

import numpy as np

import cstrees.cstree as ct
from cstrees import csi_relation

import logging
import sys
from importlib import reload  # Not needed in Python 2

reload(logging)
FORMAT = '%(filename)s:%(funcName)s (%(lineno)d):  %(message)s'
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format=FORMAT)


class Stage:
    """
       A Stage is a CSI of form 
       X_a:b _|_ Xj | S_s, X_c=x_c, 
       with an order and mayube an associated cond prob dist.

    """

    def __init__(self, list_repr, color=None, cards=None) -> None:
        self.level = len(list_repr)-1
        self.list_repr = list_repr
        # Check if singleton, if so set color black
        if all([isinstance(i, int) for i in list_repr]):
            self.color = "black"
        else:
            self.color = color        
        self.probs = None
        self.cards=cards
        self.csi = self.to_csi()

    def __hash__(self) -> int:
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
    
    def is_singleton(self):
        """
        Checks if the stage is a singleton.
        """
        return self.size() == 1

    def to_df(self, column_labels, max_card=None, write_probs=False):
        """Write sthe stage to dataframe. columns is..?

        Args:
            columns (_type_): _description_

        Returns:
            _type_: _description_
        """
        import pandas as pd

        d = {}
        if write_probs:
            cols = range(len(column_labels) - max_card)
        else:
            cols = range(len(column_labels))

        for i in cols:
            if i < len(self.list_repr):
                if type(self.list_repr[i]) == set:
                    d[column_labels[i]] = ["*"]
                else:
                    d[column_labels[i]] = [self.list_repr[i]]
            else:
                d[column_labels[i]] = ["-"]
        
        if (self.probs is not None) and write_probs:            
            df = pd.DataFrame(d, columns=column_labels[:-max_card])
            df_prop = pd.DataFrame({"PROB_"+str(i):[prob] for i, prob in enumerate(self.probs)})
            df = pd.concat([df, df_prop], axis=1)
        else:
            df = pd.DataFrame(d, columns=column_labels)
        return df

    def set_random_params(self, cards):
        self.probs = np.random.dirichlet(
            [1] * cards[self.level])  # Need to fix this


    def __sub__(self, stage):
        """ b is typically a sample from the space self.

        Args:
            csi (CSI_rel): The CSI relation to subract.

        Returns:
            list: A list of CSI relations representing the new space.
        """
        assert stage.cards is not None # Shouldnt use assert here
        assert self.cards is not None
              
        a = self
        b = stage
        p = self.level

        cards = self.cards
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
                    result.append(Stage(l, cards=cards))

        return result

    def to_csi(self, labels=None):
        sepseta = set()
        cond_set = set()
        context = {}
        sepsetb = {self.level+1}

        for i, el in enumerate(self.list_repr):
            if type(el) is set:
                sepseta.add(i)
            else:
                context[i] = el

        ci = csi_relation.CI(sepseta, sepsetb, cond_set, labels=labels)
        context = csi_relation.Context(context, labels=labels)
        
        return csi_relation.CSI(ci, context, cards=self.cards)

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
        if self.probs is not None:
            return str(self.list_repr) + "; probs: " + str(self.probs)+ "; color: " + str(self.color)
        return str(self.list_repr)


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

    return Stage(csilist, cards=stage.cards)
  
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