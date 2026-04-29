import os
import sys
import types
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np


class _FakeAdata:
    def __init__(self):
        self.obs = {
            "orig.ident": ["s1", "s2"],
            "type": ["case", "ctrl"],
            "major": ["ct1", "ct2"],
        }
        self.X = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.layers = {}
        self.uns = {}

    def write_h5ad(self, path):
        self.written_path = path


class TestDatasetHandlerPath(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self.project_root = Path(__file__).resolve().parents[2]
        self.code_dir = self.project_root / "code"
        self.pipeline_dir = self.code_dir / "pipeline"
        sys.path.insert(0, str(self.code_dir))
        os.chdir(self.pipeline_dir)

        self.fake_adata = _FakeAdata()
        self._saved_modules = {}
        self._install_stubs()
        sys.modules.pop("process.processer", None)

    def tearDown(self):
        os.chdir(self._old_cwd)
        sys.modules.pop("process.processer", None)
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _stub_module(self, name, module):
        self._saved_modules.setdefault(name, sys.modules.get(name))
        sys.modules[name] = module

    def _install_stubs(self):
        process_pkg = types.ModuleType("process")
        process_pkg.__path__ = [str(self.code_dir / "process")]
        self._stub_module("process", process_pkg)

        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(self.code_dir / "utils")]
        self._stub_module("utils", utils_pkg)

        pandas = types.ModuleType("pandas")
        pandas.read_csv = lambda *args, **kwargs: None
        self._stub_module("pandas", pandas)

        scipy = types.ModuleType("scipy")
        scipy_sparse = types.ModuleType("scipy.sparse")
        scipy_sparse.csr_matrix = lambda x: x
        scipy_sparse.issparse = lambda x: False
        self._stub_module("scipy", scipy)
        self._stub_module("scipy.sparse", scipy_sparse)

        scanpy = types.ModuleType("scanpy")
        scanpy.read_h5ad = lambda path: self.fake_adata
        scanpy.pp = types.SimpleNamespace(
            filter_genes=lambda *args, **kwargs: None,
            normalize_total=lambda *args, **kwargs: None,
            log1p=lambda *args, **kwargs: None,
        )
        self._stub_module("scanpy", scanpy)

        liana = types.ModuleType("liana")
        self._stub_module("liana", liana)

        liana_method = types.ModuleType("liana.method")

        class _Method:
            magnitude = "fake_score"
            specificity = None

            @staticmethod
            def by_sample(*args, **kwargs):
                return {"ok": True}

        for method_name in [
            "cellphonedb",
            "connectome",
            "cellchat",
            "scseqcomm",
            "singlecellsignalr",
            "natmi",
            "logfc",
            "rank_aggregate",
            "geometric_mean",
        ]:
            setattr(liana_method, method_name, _Method)
        self._stub_module("liana.method", liana_method)

        classify_utils = types.ModuleType("utils.classify_utils")
        classify_utils._dict_setup = lambda *args, **kwargs: None
        classify_utils.run_mofatalk = lambda *args, **kwargs: None
        classify_utils.run_tensor_c2c = lambda *args, **kwargs: None
        classify_utils._run_rf_auc = lambda *args, **kwargs: None
        classify_utils._assign_dict = lambda *args, **kwargs: None
        classify_utils._generate_splits = lambda *args, **kwargs: None
        self._stub_module("utils.classify_utils", classify_utils)

        prep_utils = types.ModuleType("process.prep_utils")
        prep_utils.filter_samples = lambda adata, **kwargs: adata
        prep_utils.filter_celltypes = lambda adata, **kwargs: adata
        prep_utils.check_group_balance = lambda adata, **kwargs: adata
        prep_utils.map_gene_symbols = lambda adata, map_df: adata
        self._stub_module("process.prep_utils", prep_utils)

        adaptive_config = types.ModuleType("utils.adaptive_config")
        adaptive_config.get_adaptive_config_for_dataset = lambda *args, **kwargs: ({}, {})
        self._stub_module("utils.adaptive_config", adaptive_config)

    def test_process_dataset_writes_to_project_data_interim(self):
        processer = import_module("process.processer")
        handler = processer.DatasetHandler("carraro", use_adaptive_config=False)

        handler.process_dataset()

        written = Path(self.fake_adata.written_path).resolve()
        expected = (self.project_root / "data" / "interim" / "carraro_processed.h5ad").resolve()
        self.assertEqual(written, expected)


if __name__ == "__main__":
    unittest.main()
