import networkx as nx

""" Naive implementation of a CStree for testing purposes and sanity checks.
"""

class CStree(nx.Graph):
  """ A CStree class.

  Args:
      causal_order (list): A causal order of the variables.

  """
  
  def __init__(self, causal_order, data=None, **attr):
    nx.Graph.__init__(self, data, **attr)

  def add_stage(self):
    """_summary_
    """
    pass

  def get_stage(self, level):
    """ Get all the stages in one level.

    Args:
        level (int): A level corresponds to variable in the causal ordering.
    """
    pass

  def to_minimal_context_graphs(self):
    """ This returns a sequence of minimal context graphs (minimal I-maps).
    """
    pass

  def csi_relations(self):
    """ Returns all the context specific indepencende (CSI) relations. 
        These should normally be thinned out using absorption.        
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
    pass

  def pdf(self, x):
    """Density function exaluated as x

    Args:
        x (array type): a vector.
    """


    