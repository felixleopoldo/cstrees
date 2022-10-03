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

  def add_stage(self, level: int, stage: dict):
    """Adds a stage.
    """
    pass

  def get_stage(self, level: int):
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
