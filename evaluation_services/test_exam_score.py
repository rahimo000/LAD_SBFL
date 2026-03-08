import unittest
import numpy as np
from evaluation_services.exam_score import best_case_exam_score, worst_case_exam_score

class TestExamScore(unittest.TestCase):

    def setUp(self):
        # 4 instructions, fault is at index 1 (the second instruction)
        self.fault_index = 1
        # rows = instructions, cols = metrics
        # metrics = ['tar', 'och']
        self.score_matrix = np.array([
            [0.8, 0.9],  # Index 0: Higher score than fault
            [0.5, 0.7],  # Index 1: FAULTY instruction (Score: 0.5)
            [0.5, 0.4],  # Index 2: Same score as fault for 'tar', lower for 'och'
            [0.2, 0.1]   # Index 3: Lower score than fault
        ])
        self.metric_keys = ['tar', 'och']
        # Total instructions = 4

    def test_best_case_exam_score(self):
        """
        For 'tar':
        - Index 0 has 0.8 (> 0.5) -> 1 instruction higher
        - Index 1 and 2 have 0.5 (== 0.5) -> 2 instructions equal
        Best Case Rank: 1 (higher) + 1 = 2
        EXAM: 2 / 4 = 0.5
        
        For 'och':
        - Index 0 has 0.9 (> 0.7) -> 1 instruction higher
        - Index 1 has 0.7 (== 0.7) -> 1 instruction equal
        Best Case Rank: 1 (higher) + 1 = 2
        EXAM: 2 / 4 = 0.5
        """
        results = best_case_exam_score(self.score_matrix, self.fault_index, self.metric_keys)
        
        self.assertEqual(results["tar"], 0.5)
        self.assertEqual(results["och"], 0.5)

    def test_worst_case_exam_score(self):
        """
        For 'tar':
        - Index 0 has 0.8 (> 0.5) -> 1 instruction higher
        - Index 1 and 2 have 0.5 (== 0.5) -> 2 instructions equal
        Worst Case Rank: 1 (higher) + 2 (equal) = 3
        EXAM: 3 / 4 = 0.75
        
        For 'och':
        - Index 0 has 0.9 (> 0.7) -> 1 instruction higher
        - Index 1 has 0.7 (== 0.7) -> 1 instruction equal
        Worst Case Rank: 1 (higher) + 1 (equal) = 2
        EXAM: 2 / 4 = 0.5
        """
        results = worst_case_exam_score(self.score_matrix, self.fault_index, self.metric_keys)
        
        self.assertEqual(results["tar"], 0.75)
        self.assertEqual(results["och"], 0.5)

    def test_fault_at_top(self):
        """Test when the fault is the clear winner."""
        # rows = instructions, cols = metrics
        matrix = np.array([
            [1.0], # Fault
            [0.5],
            [0.0]
        ])
        keys = ["tar"]
        # Fault is index 0
        best = best_case_exam_score(matrix, 0, keys)
        worst = worst_case_exam_score(matrix, 0, keys)
        self.assertEqual(best["tar"], 1/3)
        self.assertEqual(worst["tar"], 1/3)

    def test_ten_instructions_with_ties(self):
        """
        Scenario: 10 instructions, fault has score 0.5.
        - 2 instructions have score 0.8 (> 0.5)
        - 4 instructions (including fault) have score 0.5 (== 0.5)
        - 4 instructions have score 0.2 (< 0.5)
        
        Best Case Rank: 2 (higher) + 1 = 3. EXAM: 3/10 = 0.3
        Worst Case Rank: 2 (higher) + 4 (equal) = 6. EXAM: 6/10 = 0.6
        """
        # rows = instructions, cols = metrics
        matrix = np.array([
            [0.8], [0.8], # 2 higher
            [0.5], # Fault at index 2
            [0.5], [0.5], [0.5], # 3 others tied (Total 4)
            [0.2], [0.2], [0.2], [0.2] # 4 lower
        ])
        keys = ["tar"]
        fault_idx = 2
        
        best = best_case_exam_score(matrix, fault_idx, keys)
        worst = worst_case_exam_score(matrix, fault_idx, keys)
        
        self.assertAlmostEqual(best["tar"], 0.3)
        self.assertAlmostEqual(worst["tar"], 0.6)

if __name__ == '__main__':
    unittest.main()
