"""Working unit tests for the SBFL scores_services package.

Run with:  python -m unittest scores_services.metricsTesting -v
"""
import os
import tempfile
import unittest
import zipfile

import numpy as np

import scores_services as ss
from scores_services import metric_service as ms
from scores_services.metric_service import calculate_all_metrics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_VERSION = os.path.join(REPO_ROOT, 'eventbuslite', '1-fault', 'Datasets', 'v1')
HAS_SAMPLE_DATA = os.path.isdir(os.path.join(SAMPLE_VERSION, 'Base'))


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

    # --- Selection: parsing & validation --------------------------------------
    def test_parse_metrics(self):
        from scores_services.selection import parse_metrics, parse_evals
        self.assertEqual(parse_metrics(None), ss.get_all_metric_keys())
        self.assertEqual(parse_metrics("jac,tar"), ['tar', 'jac'])  # canonical order
        self.assertEqual(parse_evals("pexam,oexam"), ['oexam', 'pexam'])
        with self.assertRaises(ValueError):
            parse_metrics("tar,bogus")
        with self.assertRaises(ValueError):
            parse_evals("oexam,bogus")

    def test_selection_validation(self):
        from scores_services.selection import Selection
        with self.assertRaises(ValueError):
            Selection(metrics=[], evals=['oexam'])
        with self.assertRaises(ValueError):
            Selection(metrics=['mj'], evals=['oexam'])  # aggregator, no base
        with self.assertRaises(ValueError):
            Selection(metrics=['tar'], evals=[])
        sel = Selection(metrics=['tar', 'och', 'jac'], evals=['oexam', 'pexam'])
        self.assertTrue(sel.show_delta)
        self.assertFalse(sel.needs_topk)
        sel2 = Selection(metrics=['tar'], evals=['oexam', 'l-Top1'])
        self.assertFalse(sel2.show_delta)
        self.assertTrue(sel2.needs_topk)

    def test_subset_matrix_shape_and_aggregator_scope(self):
        from scores_services.selection import Selection
        sel = Selection(metrics=['tar', 'och', 'jac', 'mj', 'apv'], evals=['oexam'])
        matrix, keys = calculate_all_metrics(self.p, self.n, metrics=sel.metrics)
        self.assertEqual(keys, ['tar', 'och', 'jac', 'mj', 'apv'])
        self.assertEqual(matrix.shape, (10, 5))
        # mj/apv aggregate the selected subset only: verify against direct calls.
        idx = {k: i for i, k in enumerate(keys)}
        sub = matrix[:, [idx['tar'], idx['och'], idx['jac']]]
        self.assertTrue(np.allclose(
            matrix[:, idx['mj']],
            ss.calculate_majority_judgment_score(matrix, [idx['tar'], idx['och'], idx['jac']])))
        self.assertTrue(np.allclose(
            matrix[:, idx['apv']],
            ss.calculate_approval_voting_score(matrix, [idx['tar'], idx['och'], idx['jac']])))
        self.assertTrue(np.all(sub >= 0))

    # --- Exam-only mode (eval deselection) ------------------------------------
    @unittest.skipUnless(HAS_SAMPLE_DATA, "sample eventbuslite data not present")
    def test_process_version_exam_only(self):
        from main import process_version
        from scores_services.selection import Selection
        sel = Selection(evals=['oexam', 'pexam', 'lex-exam', 'rev-exam'])
        res = process_version(SAMPLE_VERSION, selection=sel)
        self.assertIsNotNone(res)
        self.assertEqual(res['topk'], {})
        self.assertIn('tar', res['best'])
        self.assertEqual(res['metric_keys'], ss.get_all_metric_keys())

    @unittest.skipUnless(HAS_SAMPLE_DATA, "sample eventbuslite data not present")
    def test_process_version_subset(self):
        from main import process_version
        from scores_services.selection import Selection
        sel = Selection(metrics=['tar', 'och', 'jac'], evals=['oexam', 'pexam'])
        res = process_version(SAMPLE_VERSION, selection=sel)
        self.assertEqual(res['metric_keys'], ['tar', 'och', 'jac'])
        self.assertEqual(set(res['topk']), set())
        self.assertIn('tar', res['best'])
        self.assertNotIn('op2', res['best'])

    @unittest.skipUnless(HAS_SAMPLE_DATA, "sample eventbuslite data not present")
    def test_printers_with_subset(self):
        import io
        from contextlib import redirect_stdout
        from main import process_version
        from printing_service.version_printing_service import print_exam_scores
        from scores_services.selection import Selection
        sel = Selection(metrics=['tar', 'och', 'jac'], evals=['oexam', 'pexam'])
        res = process_version(SAMPLE_VERSION, selection=sel)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_exam_scores("v1", res, selection=sel)
        out = buf.getvalue()
        self.assertIn("oexam", out)
        self.assertIn("delta", out)  # auto-shown with oexam+pexam
        self.assertNotIn("l-Top1", out)
        self.assertNotIn("lex-ex", out)
        self.assertNotIn("ochiai", out)


class TestExcelExport(unittest.TestCase):

    def _fake_averages(self):
        rng = np.random.default_rng(7)
        metrics = ss.get_all_metric_keys()
        topk_keys = [f"l-Top{k}" for k in [1, 3, 5]] + [f"r-Top{k}" for k in [1, 3, 5]]
        avg = {}
        for proj in ("projA", "projB"):
            avg[proj] = {m: {"oexam": rng.random(), "pexam": rng.random(),
                              "lexical": rng.random(), "reverse": rng.random(),
                              **{k: rng.integers(0, 2) for k in topk_keys}}
                         for m in metrics}
        return avg

    def _export_and_check(self, selection, expected_series):
        import re
        from unittest import mock
        from printing_service.excel_export_service import export_dataset_overall
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('fileManagment.output_dir', return_value=tmp):
                path = export_dataset_overall("overall.xlsx", "fakeDS", self._fake_averages(),
                                              selection=selection)
            self.assertEqual(path, os.path.join(tmp, "overall.xlsx"))
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                chart = z.read("xl/charts/chart1.xml").decode()
                shared = z.read("xl/sharedStrings.xml").decode()
            self.assertIn("xl/worksheets/sheet1.xml", names)
            series = re.findall(r"<c:tx><c:v>([^<]+)</c:v>", chart)
            self.assertEqual(series, expected_series)
            for m in selection.metrics:
                self.assertIn(m, shared)

    def test_export_all(self):
        from scores_services.selection import default_selection
        sel = default_selection()
        expected = ['oexam', 'pexam', 'lex-exam', 'rev-exam', 'deltaexam',
                    'l-Top1', 'l-Top3', 'l-Top5', 'r-Top1', 'r-Top3', 'r-Top5']
        self._export_and_check(sel, expected)

    def test_export_subset(self):
        from scores_services.selection import Selection
        sel = Selection(metrics=['tar', 'och', 'jac'], evals=['oexam', 'pexam'])
        self._export_and_check(sel, ['oexam', 'pexam', 'deltaexam'])


class TestExportPaths(unittest.TestCase):

    def test_bare_name_resolves_into_output(self):
        from unittest import mock
        import fileManagment as fm
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('fileManagment.output_dir', return_value=tmp):
                self.assertEqual(fm.prepare_export_path("quick.csv"),
                                 os.path.join(tmp, "quick.csv"))

    def test_nested_subpath_preserved_and_created(self):
        from unittest import mock
        import fileManagment as fm
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('fileManagment.output_dir', return_value=tmp):
                resolved = fm.prepare_export_path("issta/out.csv")
                self.assertEqual(resolved, os.path.join(tmp, "issta", "out.csv"))
                self.assertTrue(os.path.isdir(os.path.join(tmp, "issta")))

    def test_traversal_and_absolute_paths_clamped(self):
        from unittest import mock
        import fileManagment as fm
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('fileManagment.output_dir', return_value=tmp):
                for evil in ("../evil.csv", "..\\evil.csv",
                             os.path.join(tmp, "abs.csv"),
                             "C:/abs/win.csv" if os.name == "nt" else "/abs/nix.csv"):
                    resolved = fm.prepare_export_path(evil)
                    self.assertEqual(os.path.commonpath([tmp, resolved]), tmp)
                    self.assertTrue(resolved.endswith("evil.csv") or
                                    resolved.endswith("abs.csv") or
                                    resolved.endswith("win.csv") or
                                    resolved.endswith("nix.csv"))

    def test_invalid_path_rejected(self):
        import fileManagment as fm
        with self.assertRaises(ValueError):
            fm.prepare_export_path("")

    def test_csv_writer_lands_in_output(self):
        from unittest import mock
        from printing_service.csv_export_service import capture_to_csv
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('fileManagment.output_dir', return_value=tmp):
                with capture_to_csv("t/r.csv"):
                    print("A | B")
                self.assertTrue(os.path.isfile(os.path.join(tmp, "t", "r.csv")))

    def test_default_export_names(self):
        from tui import default_export_name
        self.assertEqual(default_export_name("version", "issta13", "eventbus", "v1"),
                         "output/issta13/eventbus_v1.csv")
        self.assertEqual(default_export_name("project", "issta13", "eventbus"),
                         "output/issta13/eventbus.csv")
        self.assertEqual(default_export_name("dataset", "issta13", excel=True),
                         "output/issta13/overall.xlsx")
        self.assertEqual(default_export_name("dataset", "issta13", excel=False),
                         "output/issta13/overall.csv")


if __name__ == '__main__':
    unittest.main()
