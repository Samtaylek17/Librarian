from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_matrix = _load_module("run_matrix_under_test", REPO_ROOT / "scripts/run_matrix.py")
vllm_with_energy = _load_module(
    "vllm_with_energy_under_test",
    REPO_ROOT / "scripts/serving/vllm_with_energy.py",
)


MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"


def _argv(*, port: int = 8000) -> list[str]:
    return [
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--port",
        str(port),
        "--enable-prefix-caching",
        "--tool-call-parser",
        "qwen3_xml",
    ]


def _environ() -> dict[str, str]:
    return {
        "VLLM_MIN_VERSION": "0.19.0",
        "VLLM_EXPECTED_VERSION": "0.19.1",
        "VLLM_EXPECTED_TOOL_PARSER": "qwen3_xml",
        "VLLM_EXPECTED_MODEL": MODEL,
        "VLLM_MODEL_REVISION": REVISION,
        "VLLM_EXPECTED_GPU_COUNT": "1",
    }


def _server(fingerprint: str = "a" * 64, *, model: str = MODEL) -> dict:
    return {
        "url": "http://worker:8000/v1",
        "model_ids": [model],
        "experiment_info": {
            "fingerprint": fingerprint,
            "gpu_count": 1,
            "image_digest": "sha256:" + "1" * 64,
            "repository_commit": "2" * 40,
            "repository_dirty": False,
            "model": model,
            "model_revision": REVISION,
            "tokenizer_revision": REVISION,
            "reproduction_status": "reconstructed reproduction using vLLM 0.19.1",
        },
    }


def test_serving_preflight_accepts_locked_contract():
    contract = vllm_with_energy._validate_preflight(
        argv=_argv(),
        environ=_environ(),
        vllm_version="0.19.1",
        tool_parsers={"hermes", "qwen3_xml"},
        visible_gpu_count=1,
    )

    assert contract == {
        "model": MODEL,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "tool_call_parser": "qwen3_xml",
        "vllm_version": "0.19.1",
    }


@pytest.mark.parametrize(
    ("version", "parsers", "gpu_count", "message"),
    [
        ("0.9.2", {"qwen3_xml"}, 1, "older than required"),
        ("0.19.1", {"hermes"}, 1, "is unavailable"),
        ("0.19.1", {"qwen3_xml"}, 2, "expected 1 visible GPU"),
    ],
)
def test_serving_preflight_rejects_drift(version, parsers, gpu_count, message):
    with pytest.raises(RuntimeError, match=message):
        vllm_with_energy._validate_preflight(
            argv=_argv(),
            environ=_environ(),
            vllm_version=version,
            tool_parsers=parsers,
            visible_gpu_count=gpu_count,
        )


def test_fingerprint_arguments_ignore_port_and_redact_secrets():
    first = vllm_with_energy._fingerprint_argv(
        _argv(port=8000) + ["--api-key", "sensitive"]
    )
    second = vllm_with_energy._fingerprint_argv(
        _argv(port=9000) + ["--api-key", "different"]
    )

    assert first == second
    assert "sensitive" not in first
    assert "<redacted>" in first


def test_server_pool_accepts_identical_fingerprints():
    first = _server()
    second = _server()
    second["url"] = "http://worker:8001/v1"

    assert run_matrix.validate_server_pool([first, second], MODEL) == "a" * 64


def test_server_pool_rejects_heterogeneous_workers():
    first = _server("a" * 64)
    second = _server("b" * 64)
    second["url"] = "http://worker:8001/v1"

    with pytest.raises(ValueError, match="fingerprints differ"):
        run_matrix.validate_server_pool([first, second], MODEL)


def test_server_pool_rejects_dirty_checkout():
    server = _server()
    server["experiment_info"]["repository_dirty"] = True

    with pytest.raises(ValueError, match="working tree is dirty"):
        run_matrix.validate_server_pool([server], MODEL)


def test_manifest_records_servers_and_refuses_mixed_resume(tmp_path):
    args = SimpleNamespace(reset_prefix_cache=True)
    kwargs = {
        "spec": {"title": "test"},
        "spec_path": "test.yaml",
        "args": args,
        "urls": ["http://worker:8000/v1"],
        "runs": [],
        "root_log": tmp_path,
        "servers": [_server()],
        "serving_fingerprint": "a" * 64,
    }
    path = run_matrix.write_manifest(**kwargs)
    manifest = json.loads(path.read_text())

    assert manifest["serving_fingerprint"] == "a" * 64
    assert manifest["dispatch_runs"][0]["servers"] == [_server()]

    kwargs["serving_fingerprint"] = "b" * 64
    with pytest.raises(SystemExit, match="Refusing to mix serving environments"):
        run_matrix.write_manifest(**kwargs)


def test_manifest_refuses_legacy_unfingerprinted_resume(tmp_path):
    (tmp_path / "_manifest.json").write_text(json.dumps({"dispatch_runs": [{}]}))
    with pytest.raises(SystemExit, match="has no serving fingerprint"):
        run_matrix.write_manifest(
            spec={"title": "test"},
            spec_path="test.yaml",
            args=SimpleNamespace(reset_prefix_cache=True),
            urls=["http://worker:8000/v1"],
            runs=[],
            root_log=tmp_path,
            servers=[_server()],
            serving_fingerprint="a" * 64,
        )


def test_lock_matches_container_base_and_contains_full_model_revisions():
    root = run_matrix.REPO_ROOT
    lock = json.loads(
        (root / "scripts/serving/reproduction_lock.json").read_text()
    )
    dockerfile = (root / "scripts/serving/Dockerfile.vllm").read_text()

    assert dockerfile.splitlines()[0] == f"FROM {lock['base_image']}"
    assert lock["vllm_version"] == "0.19.1"
    assert lock["tool_call_parser"] == "qwen3_xml"
    assert all(len(revision) == 40 for revision in lock["models"].values())
