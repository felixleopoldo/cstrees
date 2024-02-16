import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))+"/src" ))

import pandas as pd
import matplotlib.pyplot as plt
import cstrees.cstree as ct
import cstrees.scoring as sc
import cstrees.stage as st

import cstrees.learning as ctl
import networkx as nx
import numpy as np
import pp

import unittest


class TestingToMinimalCSIs(unittest.TestCase):
    def test_v_structure(self):
        # 3 variables, 2 outcomes each
        tree = ct.CStree([2] * 3, labels=["a", "b", "c"])

        # V-structure
        tree.update_stages(
            {
                0: [st.Stage([{0, 1}])],
                1: [
                    st.Stage([0, 0]),
                    st.Stage([0, 1]),
                    st.Stage([1, 0]),
                    st.Stage([1, 1]),
                ],
            }
        )
        minl_csis = tree.to_minimal_context_csis()

        csi_strings = set([])
        for cont, csis in minl_csis.items():
            for csi in csis:
                csi_strings.add(str(csi))

        # Testing against the string representation of the CSIs.
        # Maybe not the best, but quite good anyway since it
        # is easy to read and understand.
        correct_csis = {"a ⊥ b"}
        self.assertEqual(csi_strings, correct_csis)

    def test_figure1(self):
        tree = ct.CStree([2, 2, 2, 2], labels=["X" + str(i) for i in range(1, 5)])
        tree.update_stages(
            {
                0: [st.Stage([0]), st.Stage([1])],
                1: [
                    st.Stage([{0, 1}, 0], color="green"),
                    st.Stage([0, 1]),
                    st.Stage([1, 1]),
                ],
                2: [
                    st.Stage([0, {0, 1}, 0], color="blue"),
                    st.Stage([0, {0, 1}, 1], color="orange"),
                    st.Stage([1, {0, 1}, 0], color="red"),
                    st.Stage([1, 1, 1]),
                    st.Stage([1, 0, 1]),
                ],
            }
        )

        csi_strings = set([])
        minl_csis = tree.to_minimal_context_csis()
        for cont, csis in minl_csis.items():
            for csi in csis:
                csi_strings.add(str(csi))

        correct_csis = {"X1 ⊥ X3 | X2=0", "X2 ⊥ X4 | X1, X3=0", "X2 ⊥ X4 | X3, X1=0"}
        self.assertEqual(csi_strings, correct_csis)


if __name__ == "__main__":
    unittest.main()
