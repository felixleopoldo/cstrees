import itertools
import logging
from itertools import chain, combinations

import networkx as nx
import numpy as np

import logging
import sys
from importlib import reload  # Not needed in Python 2

reload(logging)
FORMAT = '%(filename)s:%(funcName)s (%(lineno)d):  %(message)s'
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format=FORMAT)
logging.basicConfig(stream=sys.stderr, level=logging.ERROR, format=FORMAT)

def mymax(s):
    if (type(s) is set) and (len(s) > 0):
        return max(s)
    else:
        return 0

class Context:
    """ A class for the context of a CSI. It takes a dictionary as input, where
    e.g. {0:0, 3:1} means that X0=0 and X3=1, where Xl is the variable at level
    l.

    Args:
        context (dict): A dictionary of the context.
    Examples:

        >>> c = Context({0:0, 3:1})
        >>> print(c)
        X0=0, X3=1
    """

    def __init__(self, context: dict, labels=None) -> None:
    
        self.context = context
        if (labels is None) and (len(self.context) > 0):
            levels = max(self.context) + 1
            self.labels = list(range(levels))
        else:
            self.labels = labels

    def __str__(self) -> str:
        
        context_str = ""
        for key, val in self.context.items():
            context_str += "{}={}, ".format(self.labels[key], val)
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


class CI:
    """ This is a contitional independence relation.

    Args:
        a (set): The first set of variables.
        b (set): The second set of variables.
        sep (set): The set of variables that separate a and b.

    Examples:
        >>> ci = csi_relation.CI({1}, {2}, {4, 5})
        >>> print(ci)
        X1 ⊥ X2 | X4, X5
    """

    def __init__(self, a, b, sep, labels=None) -> None:
        self.a = a
        self.b = b
        self.sep = sep
        
        # Just set the labels to [0,1,2,3,..]
        if labels is None:
            levels = max(mymax(self.a), mymax(self.b), mymax(
            self.sep)) + 1        
            self.labels = range(levels)
        else:
            self.labels = labels

    def __eq__(self, o: object) -> bool:
        return ((((self.a == o.a) & (self.b == o.b)) |
                ((self.a == o.b) & (self.b == o.a)))
                & (self.sep == o.sep))

    def __str__(self) -> str:
        s1 = ""
        for i in self.a:
            s1 += "{}, ".format(self.labels[i])
        s1 = s1[:-2]
        s2 = ""
        for i in self.b:
            s2 += "{}, ".format(self.labels[i])
        s2 = s2[:-2]
        s3 = ""
        if len(self.sep) > 0: # BUG: sum instead of len ???
            for i in self.sep:
                s3 += "{}, ".format(self.labels[i])
            s3 = s3[:-2]
            return "{} ⊥ {} | {}".format(s1, s2, s3)
        return "{} ⊥ {}".format(s1, s2)


class CSI:
    """This is a context specific relation. Itshould be implemented
       as a context and a CI relation.

    Args:
        ci (CI): The CI relation.
        context (Context): The context.
        cards (list): The list of cardinalities of the variables.

    Examples:
        >>> from cstrees import csi_relation
        >>> c = csi_relation.Context({0:0, 3:1})
        >>> ci = csi_relation.CI({1}, {2}, {4, 5})
        >>> csi = csi_relation.CSI(ci, c)
        >>> print(csi)
        X1 ⊥ X2 | X4, X5, X0=0, X3=1
    """

    def __init__(self, ci: CI, context: Context, cards=None) -> None:
        self.ci = ci
        self.context = context
        self.cards = cards

    def __and__(self, other):
        a = self.as_list()
        b = other.as_list()
        c_list = []

        for el in zip(a, b):
            pass

        return CSI(c_list, cards=self.cards)

    def as_list(self):
        """ List representation. Important: only for pairwise CSIs.

        Returns:
            list: list representation of the CSI.
        Examples:
            >>> from cstrees import csi_relation
            >>> c = csi_relation.Context({0:0, 3:1})
            >>> ci = csi_relation.CI({1}, {2}, {4, 5}) # CI between a pair of variables
            >>> csi = csi_relation.CSI(ci, c)
            >>> csi.as_list()
            [{0}, None, None, {1}, {0, 1}, {0, 1}]
        """
        
        logging.debug("Pairwise CSI as a list ")
        assert self.cards is not None
                
       
        # Get the level as the max element-1
        # The Nones not at index 0 encode the CI variables.
       

        if not ((len(self.ci.a) == 1) and (len(self.ci.b) == 1)):
            print("This only works for pairwise csis (Xi _|_ Xj | ...).")
            return None
        # print(print(self.ci.sep))
        levels = max(mymax(self.ci.a), mymax(self.ci.b), mymax(
            self.ci.sep), mymax(self.context.context)) + 1
        
        #cards = [2] * levels
        cards = self.cards[:levels+1]
         
        csilist = [None] * levels
        for l in range(levels):
            if (l in self.ci.a) or (l in self.ci.b):
                csilist[l] = None  # None to indicate the CI variables.
            elif l in self.ci.sep:
                csilist[l] = set(range(cards[l]))
            elif l in self.context:
                csilist[l] = {self.context[l]}

        return csilist

    def __hash__(self) -> int:
        """TODO: Check that the order is correct, so tht not 1 CSI can
        be represented in 2 ways.

        Returns:
            int: hash of the string representation.
        """
        return hash(str(self))

    def __str__(self) -> str:

        if len(self.context.context) == 0:
            # No context
            return "{}".format(self.ci)

        if len(self.context.context) != 0:
            if len(self.ci.sep) == 0:
                # Adding the |
                return "{} | {}".format(self.ci, self.context)
            else:
                # | is already there
                return "{}, {}".format(self.ci, self.context)

def decomposition(ci: CI):
    """Generate all possible pairwise CI relations that are implied by
    decomposition rule.

    Args:
        ci (CI): A CI relation.

    Returns:
        list: List of pairwise CI relations.
        
    Examples:
        >>> from cstrees import csi_relation
        >>> ci = csi_relation.CI({1,2}, {3,4},{5})
        >>> print(ci)
        >>> dec = csi_relation.decomposition(ci)
        >>> for d in dec:
        >>>     print(d)
        X1, X2 ⊥ X3, X4 | X5
        X1 ⊥ X3 | X5
        X1 ⊥ X4 | X5
        X2 ⊥ X3 | X5
        X2 ⊥ X4 | X5
    """

    cilist = []
    for x in itertools.product(ci.a, ci.b):
        new_ci = CI({x[0]}, {x[1]}, ci.sep)
        if new_ci == ci:
            continue
        cilist.append(new_ci)

    return cilist


def _powerset(iterable):
    """Returns the set of all subsets of a set.

    Args:
        iterable (list): Lest of elements.

    Returns:
        list: List of all subsets.

    Example:
        >>> powerset([1,2,3])
        () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def weak_union(ci: CI):
    """ Using weak union just to get pairwise indep relations from a CSI.
    
    Args:
        ci (CI): CI relation
    
    Returns:
        list: List of pairwise CI relations.
    
    Examples:
        >>> from cstrees import csi_relation
        >>> ci = csi_relation.CI({1,2}, {3,4},{5})
        >>> print("Original CI:")
        >>> print(ci)        
        >>> dec = csi_relation.weak_union(ci)
        >>> print("Pairwise CI relations:")
        >>> for d in dec:
        >>>     print(d)
        Original CSI:
        X1, X2 ⊥ X3, X4 | X5
        Pairwise CI relations:
        X1, X2 ⊥ X4 | X3, X5
        X1, X2 ⊥ X3 | X4, X5
        X2 ⊥ X3, X4 | X1, X5
        X1 ⊥ X3, X4 | X2, X5
    """
    cis = []
    for d in _powerset(ci.b):
        d = set(d)
        if (len(d) == 0) | (d == ci.b):
            continue

        BuD = ci.b
        cis.append(CI(ci.a, BuD-d, ci.sep | d))

    for d in _powerset(ci.a):
        d = set(d)
        if (len(d) == 0) | (d == ci.a):
            continue

        d = set(d)
        AuD = ci.a
        cis.append(CI(AuD - d, ci.b, ci.sep | d))

    return cis


def pairwise_cis(ci: CI):
    """ Using weak union just to get pairwise indep relations from a CI.
    X_a _|_ X_b | X_d
        
    Args:
        ci (CI): CI relation
        
    Examples:
        >>> from cstrees import csi_relation
        >>> ci = csi_relation.CI({1,2}, {3,4},{5})        
        >>> print("Original CI: ", ci)
        >>> print("Pairwise CIs:")
        >>> for x in pw:
        >>>    print(x)
        Original CI:  X1, X2 ⊥ X3, X4 | X5
        Pairwise CIs:
        X1 ⊥ X3 | X2, X4, X5
        X1 ⊥ X4 | X2, X3, X5
        X2 ⊥ X3 | X1, X4, X5
        X2 ⊥ X4 | X1, X3, X5
    """
    cis = []
    A = ci.a
    B = ci.b  # This will probably just contain one element.
    for x in itertools.product(A, B):
        rest = (A - {x[0]}) | (B - {x[1]})
        cis.append(CI({x[0]}, {x[1]}, ci.sep | rest))
    return cis


def pairwise_csis(csi: CSI, cards=None):
    """ Using weak union just to get pairwise indep relations from a CSI.

    Args:
        csi (CSI): CSI relation
                
    Examples:
        >>> from cstrees import csi_relation
        >>> c = csi_relation.Context({6:0})
        >>> csi = csi_relation.CSI(ci, c)
        >>> print("Original CSI: ", csi)
        >>> pw = csi_relation.pairwise_csis(csi)
        >>> print("Pairwise CSIs:")
        >>> for x in pw:
        >>>    print(x)
        X1 ⊥ X3 | X2, X4, X5, X6=0
        X1 ⊥ X4 | X2, X3, X5, X6=0
        X2 ⊥ X3 | X1, X4, X5, X6=0
        X2 ⊥ X4 | X1, X3, X5, X6=0
    """
    logging.debug("Pairwise CSIs")
    context = csi.context
    ci_pairs = pairwise_cis(csi.ci)
    csis = []
    for ci in ci_pairs:
        csi = CSI(ci, context=context, cards=cards)
        csis.append(csi)
    return csis


def mix(csilist_tuple, level, cards):
    """Mix two pairwise CI relations represented as lists.
    A mix is the intersection at each level except for the current level l
    where the values are joined.

    Args:
        csilist_tuple (tuple): Two pairwise CSI lists
        l (int): the level
        cards (list): cardinalities of the levels.

    Returns:
        list: A mixed CSI list.

    Example:
        >>> a = [0, None, None, {0,1}, 1]
        >>> b = [1, None, None, 0, 1]
        >>> c = do_mix((a,b), 0, [2,2,2,2,2])
        >>> c
        [{0, 1}, None, None, 0,, 1]
    """

    p = len(csilist_tuple[0])
    mix = [None] * p

    # Going through all the levels and mix at all levels.
    # The result should be stored somewhere.
    for i, a in enumerate(zip(*csilist_tuple)):
        #print(i, a)
        # None means that a[0] is some of the CI tuple. So just skip.
        if a[0] is None:
            continue
        if i == level:  # if at the current level, the values are joined.
            mix[i] = set(range(cards[level]))
        else:  # do the intersection
            mix[i] = set.intersection(*a)
            if len(mix[i]) == 0:
                return None  # The CSIs are disjoint, so return None.

    return mix


def partition_csis(csilist_list, level, cards):
    """ Put the csis in different sets that can possibly be mixed to create 
    new csis. It is assumed that all are pairwise csis with the same 
    "indepedent" variables.

    Args:
        csilist_list (list): List of pairwise CSI lists.
        level (int): The level up to which the mixing is done.
        cards (list): Cardinalities of the levels.

    Returns:
        list: list of disjoint lists of pairwise CSI lists that can possibly be mixed.
    
    Example:
        >>> pairwise_csis
        [{0}, None, {0}, None]
        [{0}, None, {1}, None]
        [{1}, None, {0}, None]
        >>> # partition based on values at level 0
        >>> partitioned_csis = partition_csis(pairwise_csis, 0, [2,2,2,2])
        >>> for i, csis in enumerate(partitioned_csis):
        >>>     print("{}: {}".format(i, csis))
        0: [[{0}, None, {0}, None], [{0}, None, {1}, None]]
        1: [[{1}, None, {0}, None]]
    """
    logging.debug("Partitioning CSIs")
    logging.debug("level {}".format(level))
    logging.debug("cards {}".format(cards))
    logging.debug(csilist_list)

    csis_to_mix = [[] for _ in range(cards[level])]
    for csilist in csilist_list:
        if len(csilist[level]) > 1:  # Only consider those with single value
            continue
        var_val = list(csilist[level])[0]  # just to get the single value from the set
        logging.debug("var_val {}".format(var_val))
        csis_to_mix[var_val].append(csilist)
      
    return csis_to_mix


def _csilist_to_csi(csilist, labels=None): #This could probably take labels as well
    """ The independent variables are represented by None. 
    Only for pairwise CSIs.

    Args:
        csilist (list): List representation of a CSI.

    Returns:
        CSI: A CSI object.
    """

    context = {}
    indpair = []
    sep = set()
    for i, values in enumerate(csilist):
        if values is None:  # None means that the variable is independent.
            indpair.append(i)
        elif len(values) == 1:
            context[i] = list(values)[0]
        elif len(values) > 1:

            sep.add(i)  # this should be == range(cards[i]))

    context = Context(context, labels=labels)

    ci = CI({indpair[0]}, {indpair[1]}, sep, labels=labels)
    csi = CSI(ci, context)
    return csi


def _csilist_subset(a, b):
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
    """ Loop through all levels l
        1. For each stage in the level do weak union to get the pairs
            Xi _|_ Xj | something, and group.
        2. For each such pair go through all levels and try to find mixable
            CSI by partition on value.
        3. If mixable, mix and put the result in a set woth newly created.

        When we loop through al levels again by where the old CSI are not mixed
        with each other that is, each tuple needs at least one CSI from the new
        CSIs.

    Args:
        paired_csis (dict): Dict of csis grouped by pairwise indep rels as Xi _|_ Xj.
        
    Example:
        >>> rels = tree.csi_relations()
        >>> for cont, rels in rels.items():
        >>>     for rel in rels:
        >>>        print(rel)
        X0 ⊥ X2, X1=0
        X1 ⊥ X3, X0=0, X2=0
        X1 ⊥ X3, X0=0, X2=1
        X1 ⊥ X3, X0=1, X2=0
        >>> minl_csis = tree.to_minimal_context_csis()
        >>> for cont, csis in minl_csis.items():
        >>>    for csi in csis:
        >>>        print(csi)
        X0 ⊥ X2, X1=0
        X1 ⊥ X3, X2=0
        X1 ⊥ X3 | X2, X0=0

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
            # This must be here, since we dont know on which variable new mixed
            # will be mixed
            newbies = csilist_list

            iteration = 1
            while len(newbies) > 0:
                logging.debug("\n#### Iteration {}".format(iteration))
                # print(pair) print("going through the levels for partitions")
                # should join with csi_list the oldies?

                fresh = []  # list of created csis
                csis_to_absorb = []  # remove from the old ones due to mixing
                # Go through all levels, potentially many times.
                for l in range(level+1):  # Added +1 after refactorization
                    logging.debug("level {}".format(l))
                    if l in pair:
                        continue

                    csis_to_mix = partition_csis(newbies + oldies, l, cards)
                    #logging.debug("csis to mix")
                    # logging.debug(csis_to_mix)

                    # Need to separate the newly created csis from the old
                    # ones. The old should not be re-mixed, i.e., the mixes
                    # must contain at least one new csi. How do we implement
                    # that? Just create new to mix and skip if not containing a
                    # new?

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
                        mixed_csi = mix(csilist_tuple, l, cards)
                        if mixed_csi is None:
                            #print("Not mixeable")
                            continue
                        else:
                            # print(mix) assert(sum([len(el)==1 for el in mix
                            # if el is not None]) <= 3)
                            if mixed_csi not in (oldies + newbies):
                                logging.debug("mixing")
                                logging.debug(csilist_tuple)
                                logging.debug("mix result: ")
                                logging.debug(mixed_csi)
                                logging.debug(
                                    "Adding {} as newly created ******".format(mixed_csi))
                                fresh.append(mixed_csi)
                                # Check if some csi of the oldies should be
                                # removed. I.e. if some in csilist_tuple is a
                                # subset of mix.
                                for csilist in csilist_tuple:
                                    # print wher the csi is from, oldies, or
                                    # newbies.
                                    if _csilist_subset(csilist, mixed_csi):  # This sho
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
                            if _csilist_subset(csi, o):
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


def _csis_by_levels_2_by_pairs(rels,cards=None):

    paired_csis = [None] * len(rels)

    for l, val in rels.items():
        #print("level: {}".format(l))
        csi_pairs = []  # X_i _|_ X_j | something
        for v in val:
            # print(v)
            csis = pairwise_csis(v, cards=cards) # Using weak unions
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


def _rels_by_level_2_by_context(rels_at_level):
    rels = {}
    for l, r in rels_at_level.items():
        for rel in r:
            if rel.context in rels:
                rels[rel.context].append(rel)
            else:
                rels[rel.context] = [rel]
    return rels


def _csi_lists_to_csis_by_level(csi_lists, p, labels):
    stages = {l: [] for l in range(p)}
    for l, csilist in enumerate(csi_lists):

        tmp = []
        # Convert formats
        for pair, csil in csilist.items():
            for csi in csil:
                csiobj = _csilist_to_csi(csi, labels=labels) # TODO: add labels?
                tmp.append(csiobj)
        stages[l] = tmp
    return stages


def csi_relations_to_dags(csi_relations, p, labels=None):
    """Converts the CSI relations to dags.
    
    Args:
        csi_relations (dict): A dictionary with contexts as keys and lists of csi relations as values.  
        p (int): The number of variables.
        labels (list, optional): A list of labels for the variables. Defaults to None.
    
    Returns:
        dict: A dictionary with contexts as keys and dags as values.
    
    """

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

                ci_tmp = CI(a, b, s)

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
        inds = sorted(set(range(p)) - context_els)
        adjmat = adjmat[np.ix_(inds, inds)]

        graph = nx.from_numpy_array(adjmat, create_using=nx.DiGraph())

        # TODO: the context should also be relabeled
        # accordingly. Maybe in Context directly.
        labs = {}
        for i, j in enumerate(inds):
            labs[i] = labels[j]
        graph = nx.relabel_nodes(graph, labs)
        graphs[context] = graph

    return graphs
