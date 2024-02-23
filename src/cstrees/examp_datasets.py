from importlib import resources
import pandas as pd

from . import datasets


def sachs_observational() -> pd.DataFrame:
    """Binarized observational dataset extracted from [1]_

    See Appendix C.3 of [2]_ and Section 5.2 of [3]_ for details of how
    the raw data is processed.

    References
    ----------

    .. [1] K. Sachs, O. Perez, D. Pe’er, D. A. Lauffenburger, and G.
    P. Nolan. Causal protein-signaling networks derived from
    multiparameter single-cell data. Science, 308(5721):523–529, 2005.

    .. [2] F. L. Rios, A. Markham, and L. Solus. Scalable structure
    learning for sparse context-specific causal systems. 2024.
    arXiv:2402.07762.

    .. [3] Y. Wang, L. Solus, K. Yang, and C. Uhler. Permutation-based
    causal inference algorithms with interventions. Advances in Neural
    Information Processing Systems, 30, 2017.
    """
    data_path = resources.files(datasets) / "sachs_observational.csv"
    return pd.read_csv(data_path)
