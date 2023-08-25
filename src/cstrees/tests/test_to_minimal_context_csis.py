
import pandas as pd
import matplotlib.pyplot as plt
import cstrees.cstree as ct
import cstrees.scoring as sc
import cstrees.stage as st



import src.cstrees.learning as ctl
import networkx as nx
import numpy as np
import pp



def test_to_minimal_context_csis():
    # 3 variables, 2 outcomes each
    tree = ct.CStree([2] * 3, labels=["a", "b", "c"])
    
    # V-structure
    tree.update_stages({
        0: [st.Stage([{0, 1}])],
        1: [st.Stage([0, 0]), st.Stage([0, 1]), st.Stage([1, 0]), st.Stage([1, 1])]})
    minl_csis = tree.to_minimal_context_csis()

    for cont, csis in minl_csis.items():
        for csi in csis:
            print(csi)