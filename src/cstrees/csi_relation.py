from cstrees.cstree import *


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

    def to_cstree_paths(self, cards: list):
        """Genreate the set(s) of path defining the CSI relations.
        note that it can be defined by several stages (set of paths).

        Args:
            cards (list): _description_

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


def decomposition(ci: CI_relation):

    cilist = []
    for x in itertools.product(ci.a, ci.b):
        new_ci = CI_relation({x[0]}, {x[1]}, ci.sep)
        if new_ci == ci:
            continue
        cilist.append(new_ci)

    return cilist

def powerset(iterable):
    # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


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


def csi_relations_to_dags(csi_relations, p, labels=None):

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
        inds = sorted(set(range(p)) - context_els)
        adjmat = adjmat[np.ix_(inds, inds)]

        graph = nx.from_numpy_array(adjmat, create_using=nx.DiGraph())

        # Label from 1 instead of 0
        labs = {}
        for i, j in enumerate(inds):
            labs[i] = labels[j]
        graph = nx.relabel_nodes(graph, labs)
        graphs[context] = graph

    return graphs