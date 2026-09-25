"""Working unit tests for the SBFL scores_services package.

Run with:  python -m unittest scores_services.metricsTesting -v
"""
import unittest

import numpy as np

import scores_services as ss
from scores_services import metric_service as ms
from scores_services.metric_service import calculate_all_metrics


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

    # --- Classic metrics: known values ------------------------------------
    def test_tarantula(self):
        expected = [0.5, 0.67, 1.0, 0.62, 1.0, 0.57, 1.0, 0.4, 0.4, 0.5]
        self.assertTrue(np.allclose(ms.Tarontula(self.p, self.n), expected))

    def test_ochiai(self):
        expected = [0.87, 0.93, 1.0, 0.83, 0.71, 0.73, 0.58, 0.47, 0.47, 0.87]
        self.assertTrue(np.allclose(ms.ochiai(self.p, self.n), expected))

    def test_jaccard(self):
        expected = [0.75, 0.86, 1.0, 0.71, 0.5, 0.57, 0.33, 0.29, 0.29, 0.75]
        self.assertTrue(np.allclose(ms.jaccard(self.p, self.n), expected))

    def test_gp13(self):
        expected = [0.943, 0.964, 1.0, 0.952, 1.0, 0.933, 1.0, 0.833, 0.833, 0.943]
        self.assertTrue(np.allclose(ms.gp13(self.p, self.n), expected))

    # --- New metrics: known values (column 2: ef=6, ep=0 -> all 1.0) -------
    def test_new_metrics_known_values(self):
        self.assertEqual(ms.op2(self.p, self.n)[2], 1.0)
        self.assertEqual(ms.kulczynski2(self.p, self.n)[2], 1.0)
        self.assertEqual(ms.zoltar(self.p, self.n)[2], 1.0)
        self.assertEqual(ms.ample(self.p, self.n)[2], 1.0)
        # Column 0: ef=6, ep=2, np_=0 -> op2=(6-2/3+1)/7=0.9, kul2=0.88, ample=|1-1|=0
        self.assertAlmostEqual(ms.op2(self.p, self.n)[0], 0.9)
        self.assertAlmostEqual(ms.kulczynski2(self.p, self.n)[0], 0.88)
        self.assertAlmostEqual(ms.ample(self.p, self.n)[0], 0.0)

    # --- New metrics: always bounded in [0, 1] -----------------------------
    def test_new_metrics_bounded(self):
        rng = np.random.default_rng(42)
        for func in (ms.op2, ms.kulczynski2, ms.zoltar, ms.ample):
            # Fixture, all-zero coverage, all-covered, and fuzz matrices
            cases = [
                (self.p, self.n),
                (np.zeros((2, 4), dtype=np.int8), np.zeros((3, 4), dtype=np.int8)),
                (np.ones((2, 4), dtype=np.int8), np.ones((3, 4), dtype=np.int8)),
                (rng.integers(0, 2, size=(5, 20)).astype(np.int8),
                 rng.integers(0, 2, size=(7, 20)).astype(np.int8)),
            ]
            for p, n in cases:
                with self.subTest(func=func.__name__, p=p.tolist(), n=n.tolist()):
                    scores = func(p, n)
                    self.assertTrue(np.all(scores >= 0.0))
                    self.assertTrue(np.all(scores <= 1.0))
                    self.assertFalse(np.any(np.isnan(scores)))

    # --- Edge cases ---------------------------------------------------------
    def test_zoltar_zero_ef_is_zero(self):
        p = np.ones((2, 3), dtype=np.int8)
        n = np.zeros((3, 3), dtype=np.int8)
        self.assertTrue(np.all(ms.zoltar(p, n) == 0.0))

    def test_zero_coverage_scores_zero(self):
        p = np.zeros((2, 3), dtype=np.int8)
        n = np.zeros((3, 3), dtype=np.int8)
        self.assertTrue(np.all(ms.kulczynski2(p, n) == 0.0))
        self.assertTrue(np.all(ms.zoltar(p, n) == 0.0))
        self.assertTrue(np.all(ms.ample(p, n) == 0.0))

    # --- No duplication: modules are the single source of truth -------------
    def test_service_reexports_match_modules(self):
        import importlib
        # NOTE: importlib (not `import scores_services.x as x`) because the
        # package also exposes same-named functions (e.g. ss.ochiai) which
        # would shadow the submodule attribute.
        tar_mod = importlib.import_module('scores_services.trantula')
        ochiai_mod = importlib.import_module('scores_services.ochiai')
        jaccard_mod = importlib.import_module('scores_services.jaccard')
        gp13_mod = importlib.import_module('scores_services.gp13')
        op2_mod = importlib.import_module('scores_services.op2')
        kul2_mod = importlib.import_module('scores_services.kulczynski2')
        zoltar_mod = importlib.import_module('scores_services.zoltar')
        ample_mod = importlib.import_module('scores_services.ample')
        pairs = [
            (ms.Tarontula, tar_mod.Tarontula), (ms.ochiai, ochiai_mod.ochiai),
            (ms.jaccard, jaccard_mod.jaccard), (ms.gp13, gp13_mod.gp13),
            (ms.op2, op2_mod.op2), (ms.kulczynski2, kul2_mod.kulczynski2),
            (ms.zoltar, zoltar_mod.zoltar), (ms.ample, ample_mod.ample),
        ]
        for via_service, via_module in pairs:
            with self.subTest(func=via_module.__name__):
                self.assertIs(via_service, via_module)

    def test_registry_wires_real_functions(self):
        for key, func, _name in ss.ACTIVE_METRICS:
            with self.subTest(key=key):
                self.assertTrue(callable(func))
                self.assertIs(func, ss.get_metric_function(key))
        self.assertEqual(ss.get_all_metric_keys(),
                         ['tar', 'och', 'jac', 'gp', 'op2', 'kul2', 'zol', 'amp', 'mj', 'apv'])

    # --- Orchestration ------------------------------------------------------
    def test_calculate_all_metrics_shape_and_keys(self):
        matrix, keys = calculate_all_metrics(self.p, self.n)
        self.assertEqual(keys, ss.get_all_metric_keys())
        self.assertEqual(matrix.shape, (10, 10))
        idx = {k: i for i, k in enumerate(keys)}
        self.assertTrue(np.allclose(matrix[:, idx['och']], ms.ochiai(self.p, self.n)))
        self.assertTrue(np.allclose(matrix[:, idx['zol']], ms.zoltar(self.p, self.n)))


if __name__ == '__main__':
    unittest.main()
