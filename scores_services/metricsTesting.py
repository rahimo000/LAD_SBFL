import unittest
from math import inf

import numpy as np

from serverapp.services.model.g_item_set import GItemSet
from serverapp.services.model.g_score import GScore
from serverapp.services.services.scores_services.dstar import calculate_dstar_scores
from serverapp.services.services.scores_services.gp13 import gp13
from serverapp.services.services.scores_services.jaccard import jaccard
from serverapp.services.services.scores_services.ochiai import ochiai
from serverapp.services.services.scores_services.topk import calculate_instruction_scores
from serverapp.services.services.scores_services.trantula import Tarontula
# {{change 1}}
from serverapp.services.services.scores_services.scores import ProdCartXor, ProdCartNP


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.p = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        ])
        self.n = np.array([
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.instruction_scores = []
        dstar_output = [3.0, 6.0, inf, 2.5, 1.0, 1.3333333333333333, 0.5, 0.4, 0.4, 3.0]
        gp13_output = [6.6, 6.75, 7.0, 5.71, 4.0, 4.67, 3.0, 2.5, 2.5, 6.6]
        jaccard_output = [0.75, 0.86, 1.0, 0.71, 0.5, 0.57, 0.33, 0.29, 0.29, 0.75]
        ochiai_output = [0.87, 0.93, 1.0, 0.83, 0.71, 0.73, 0.58, 0.47, 0.47, 0.87]
        topk_output = [6.33, 6.67, 7.0, 5.67, 4.0, 4.67, 3.0, 2.67, 2.67, 6.33]
        tarantula_output = [0.5, 0.67, 1.0, 0.62, 1.0, 0.57, 1.0, 0.4, 0.4, 0.5]
        ProdCartXor_output = [0.0, 6.0, 12.0, 6.0, 6.0, 6.0, 4.0, 6.0, 6.0, 0.0]
        ProdCartNP_output = [0.0, 6.0, 12.0, 5.0, 6.0, 4.0, 4.0, 2.0, 2.0, 0.0]

        for i in range(10):
            gscore = GScore()
            gscore.dstar = dstar_output[i]
            gscore.gp = gp13_output[i]
            gscore.jac = jaccard_output[i]
            gscore.och = ochiai_output[i]
            gscore.topk = topk_output[i]
            gscore.tar = tarantula_output[i]
            gscore.xor = ProdCartXor_output[i]
            gscore.np = ProdCartNP_output[i]
            self.instruction_scores.append(gscore)

    def test_dstar(self):
        expected_output = [18.0, 36.0, inf, 12.5, 3.0, 5.333333333333333, 1.0, 0.8, 0.8, 18.0]
        actual_output = calculate_dstar_scores(self.p, self.n, 2)
        print(actual_output)
        self.assertTrue(np.allclose(actual_output, expected_output))

    def test_gp13(self):
         expected_output = [6.6, 6.75, 7.0, 5.71, 4.0, 4.67, 3.0, 2.5, 2.5, 6.6]
         actual_output = gp13(self.p, self.n)
         print(actual_output)
         self.assertTrue(np.allclose(actual_output, expected_output))

    def test_jaccard(self):
        expected_output = [0.75, 0.86, 1.0, 0.71, 0.5, 0.57, 0.33, 0.29, 0.29, 0.75]
        actual_output = jaccard(self.p, self.n)
        print(actual_output)
        self.assertTrue(np.allclose(actual_output, expected_output))

    def test_ochiai(self):
        expected_output = [0.87, 0.93, 1.0, 0.83, 0.71, 0.73, 0.58, 0.47, 0.47, 0.87]
        actual_output = ochiai(self.p, self.n)
        print(actual_output)
        self.assertTrue(np.allclose(actual_output, expected_output))

    def test_topk(self):
        expected_output = [6.33, 6.67, 7.0, 5.67, 4.0, 4.67, 3.0, 2.67, 2.67, 6.33]
        actual_output = calculate_instruction_scores(self.p, self.n)
        print(actual_output)
        self.assertTrue(np.allclose(actual_output, expected_output))

    def test_tarantula(self):
        expected_output = [0.5, 0.67, 1.0, 0.62, 1.0, 0.57, 1.0, 0.4, 0.4, 0.5]
        actual_output = Tarontula(self.p, self.n)
        print(actual_output)
        self.assertTrue(np.allclose(actual_output, expected_output))

    def test_ProdCartXor(self):
        expected_matrix = np.array(
            [[0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
             [0, 1, 1, 1, 0, 1, 1, 0, 0, 0],
             [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
             [0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
             [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]]
        )
        expected_scores = [0.0, 6.0, 12.0, 6.0, 6.0, 6.0, 4.0, 6.0, 6.0, 0.0]
        result = ProdCartXor(self.p, self.n)
        print("----xor --")
        print(result.Matrix)
        print([float(i) for i in result.scores])
        self.assertTrue(np.allclose(result.Matrix, expected_matrix))
        self.assertTrue(np.allclose(result.scores, expected_scores))

    def test_ProdCartNP(self):
        expected_matrix = np.array(
            [[0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
             [0, 1, 1, 1, 0, 1, 1, 0, 0, 0],
             [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
        )
        expected_scores = [0.0, 6.0, 12.0, 5.0, 6.0, 4.0, 4.0, 2.0, 2.0, 0.0]
        result = ProdCartNP(self.p, self.n)
        print("----np--")
        print(result.Matrix)
        print([float(i) for i in result.scores])
        self.assertTrue(np.allclose(result.Matrix, expected_matrix))
        self.assertTrue(np.allclose(result.scores, expected_scores))

    
    def calculate_and_print_itemsets(self):
        item_sets_to_test = [
            [2, 3, 4, 5, 6, 8, 9],
            [2, 3, 4, 5, 6, 7],
            [2, 3, 4, 6, 8, 9],
            [2, 3, 4, 6, 7],
            [2, 3, 4, 5],
            [2, 3],
            [3, 5],
            [3, 5, 7, 8, 9],
            [3],
            [3, 7, 8, 9],
            [3, 5, 6, 8, 9],
            [3, 4, 6, 8, 9]
        ]
        print("------")
        for itemset_indices in item_sets_to_test:
            itemset = GItemSet()
            itemset.itemset = itemset_indices
            itemset.calculateScore(self.instruction_scores)
            print(f"Itemset {itemset_indices}: dstar={itemset.score.dstar}, gp={itemset.score.gp}, jac={itemset.score.jac}, och={itemset.score.och}, topk={itemset.score.topk}, tar={itemset.score.tar}, xor={itemset.score.xor}, np={itemset.score.np}")

    def test_calculate_itemsets(self):
        self.calculate_and_print_itemsets()

if __name__ == '__main__':
    unittest.main()