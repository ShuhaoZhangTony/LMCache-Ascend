# SPDX-License-Identifier: Apache-2.0
"""Static regression for optional-engine imports without requiring an NPU."""
import ast
from pathlib import Path


def test_vllm_connectors_do_not_import_removed_sglang_symbols():
    root = Path(__file__).resolve().parents[1] / 'lmcache_ascend/v1/npu_connector'
    tree = ast.parse((root / 'npu_connectors.py').read_text())
    imports = [alias.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) for alias in node.names]
    assert not any(name.startswith('SGLang') for name in imports)
    legacy = ast.parse((root / 'sglang_connectors.py').read_text())
    assert {node.name for node in legacy.body if isinstance(node, ast.ClassDef)} == {
        'SGLangNPUConnector', 'SGLangLayerwiseNPUConnector'}
    factory = ast.parse((root / '__init__.py').read_text())
    assert not any(isinstance(node, ast.ImportFrom) and
                   node.module.endswith('sglang_connectors') for node in factory.body)
