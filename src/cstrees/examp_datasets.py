from importlib import resources
import pandas as pd
import numpy as np

from . import datasets


def sachs_observational(binarized: bool = True) -> pd.DataFrame:
    """(Binarized) observational dataset extracted from [1]_

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
    df = pd.read_csv(data_path)
    if binarized:
        sachsnp = df.to_numpy()
        sachs2 = np.zeros([len(df), len(list(df.columns))], int)
        for i in range(len(list(df.columns))):
            sachs2[:, i] = pd.cut(sachsnp[:, i], 2, labels=False)
        df = pd.DataFrame(sachs2, columns=list(df.columns))
    return df


def alarm():
    """Discrete observational (realistically sythetic) dataset from [1]_

    Also availale from the R package `bnlearn <https://www.bnlearn.com/documentation/man/alarm.html>`_.

    References
    ----------
    .. [1] I. Beinlich, H. J. Suermondt, R. M. Chavez, G. F. Cooper.
    The ALARM Monitoring System: A Case Study with Two Probabilistic
    Inference Techniques for Belief Networks. Proceedings of the 2nd
    European Conference on Artificial Intelligence in Medicine,
    247–256, 1989.
    """
    data_path = resources.files(datasets) / "alarm.csv"
    df = pd.read_csv(data_path)
    return df
