"""CPU regressions for GPT-OSS rollout startup memory admission."""

import json
from pathlib import Path
import sys

import pytest

import gpu_runtime


ROOT = Path(__file__).resolve().parents[1]


def blackwell_cards(total_gib=94.97):
    return {
        gpu: gpu_runtime.GPUMemory(
            physical_id=gpu,
            name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            total_gib=total_gib, free_gib=total_gib)
        for gpu in range(8)
    }


@pytest.mark.parametrize("total_gib,quantization,expected_tp", [
    (94.97, "", 1), (141.0, "", 1), (48.0, "", 4),
    (94.97, "bitsandbytes", 2),
])
def test_gpt_oss_layout_and_worker_share_bounded_startup_contract(
        total_gib, quantization, expected_tp):
    from gen_workers import _vllm_engine_kwargs

    cfg = dict(model_name="openai/gpt-oss-120b", max_seq_length=131072,
               vllm_gpu_memory_utilization=.9,
               vllm_runtime_reserve_gib=5., allow_gpt_oss_tp1=True,
               vllm_quantization=quantization)
    roles = gpu_runtime.allocate_gpu_roles(range(8), "erdos")
    layout = gpu_runtime.derive_vllm_parallel_layout(
        cfg, roles, blackwell_cards(total_gib), 64)
    assert layout.tensor_parallel_size == expected_tp
    assert layout.pipeline_parallel_size == 1
    assert layout.replicas * expected_tp == 8
    kwargs = _vllm_engine_kwargs(
        cfg["model_name"], 131072, False, 32, .9, quantization=quantization,
        tensor_parallel_size=expected_tp, enable_sleep_mode=True)
    assert kwargs["enable_lora"] and kwargs["enable_sleep_mode"]
    if quantization == "":
        assert kwargs["worker_cls"] == "vllm_compat.gpt_oss_worker.GptOssWorker"
    else:
        assert "worker_cls" not in kwargs


@pytest.mark.parametrize("resume", [False, True])
def test_erdos_gpt_oss_config_uses_eight_coders(monkeypatch, tmp_path, resume):
    import train_multy_CVaR as training

    monkeypatch.setenv("AVAILABLE_GPUS", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(gpu_runtime, "query_gpu_memory", blackwell_cards)
    argv = ["train_multy_CVaR.py", "--config", str(ROOT / "configs/erdos.yaml"),
            "--backend", "hf", "--advantage-mode", "spo-rs", "--fast",
            "--strategies", "--coder-retry", "--fused-long-attention"]
    if resume:
        # Resume recomputes topology instead of retaining the TP=2 workaround.
        (tmp_path / "config.json").write_text(json.dumps({
            "allow_gpt_oss_tp1": True, "vllm_tensor_parallel_size": 2,
            "vllm_pipeline_parallel_size": 1,
        }))
        argv += ["--resume", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    cfg, _ = training.load_config()
    assert cfg.vllm_tensor_parallel_size == 1
    assert cfg.vllm_pipeline_parallel_size == 1
    assert cfg.num_gpus == cfg.num_training_gpus == 8
    assert cfg.max_seq_length == cfg.max_new_tokens == 131072
    assert cfg.coder_reasoning_effort == "low"
    assert cfg.strategy_backend == "local" and not cfg.api_strategy
    assert cfg.training_layout == "replicated"
    assert cfg.gen_micro_batch >= 8


def test_qwen_keeps_original_worker_and_parallel_layout():
    from gen_workers import _vllm_engine_kwargs

    cfg = dict(model_name="Qwen/Qwen3.8-27B", max_seq_length=131072,
               vllm_gpu_memory_utilization=.9, vllm_runtime_reserve_gib=15.)
    roles = gpu_runtime.allocate_gpu_roles(range(8), "erdos")
    layout = gpu_runtime.derive_vllm_parallel_layout(
        cfg, roles, blackwell_cards(), 24)
    assert (layout.tensor_parallel_size, layout.pipeline_parallel_size,
            layout.replicas) == (1, 1, 8)
    kwargs = _vllm_engine_kwargs(cfg["model_name"], 131072, False, 32, .9)
    assert "worker_cls" not in kwargs


def test_bounded_repack_preserves_bytes_and_uses_only_one_expert_of_scratch():
    import torch
    from vllm_compat.gpt_oss_repack import repack_experts_in_place

    weight = torch.arange(4 * 16 * 16, dtype=torch.int32).to(torch.uint8).view(4, 16, 16)
    original = weight.clone()
    calls = []
    sentinel_perm = object()

    def packing(experts, size_n, size_k, perm, is_a_8bit):
        assert experts.shape[0] == 1
        assert (size_n, size_k, perm, is_a_8bit) == (16, 32, sentinel_perm, False)
        calls.append(experts.shape[0])
        # A deterministic stand-in for the CUDA packing kernel, deliberately
        # changing both byte order and dtype/shape like the actual kernel.
        return experts.flatten().flip(0).contiguous().view(torch.int32).view(1, 4, 16)

    expected = torch.stack([
        expert.flatten().flip(0).contiguous().view(torch.int32).view(4, 16)
        for expert in original])
    actual = repack_experts_in_place(
        weight, 16, 32, sentinel_perm, False, repack=packing)
    assert torch.equal(actual, expected)
    assert actual.data_ptr() == weight.data_ptr()
    assert calls == [1] * 4


def test_bounded_repack_rejects_changed_storage_contract_before_overwriting():
    import torch
    from vllm_compat.gpt_oss_repack import repack_experts_in_place

    weight = torch.ones(2, 16, 16, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="storage contract"):
        repack_experts_in_place(
            weight, 16, 32, None, False,
            repack=lambda *args: torch.zeros(1, 16, 32, dtype=torch.uint8))
    assert torch.all(weight == 1)


@pytest.mark.parametrize("fail", [False, True])
def test_custom_worker_scopes_repacking_and_allocator_to_startup(monkeypatch, fail):
    from contextlib import contextmanager
    from types import SimpleNamespace

    def original(weight, size_n, size_k, perm, is_a_8bit):
        raise AssertionError("not called in this wiring check")

    marlin = SimpleNamespace(_repack_marlin_experts=original)
    monkeypatch.setitem(sys.modules,
        "vllm.model_executor.layers.quantization.utils",
        SimpleNamespace(marlin_utils_fp4=marlin))

    class Worker:
        split_limit = 20

        @contextmanager
        def _scoped_allocator_max_split(self, max_split_size_mb):
            previous = self.split_limit
            self.split_limit = max_split_size_mb
            try:
                yield
            finally:
                self.split_limit = previous

        def load_model(self, *, load_dummy_weights=False):
            # Match the allocator context entered by vLLM 0.28's loader.
            with self._scoped_allocator_max_split(max_split_size_mb=20):
                assert marlin._repack_marlin_experts is not original
                assert load_dummy_weights
                assert self.split_limit > 20
                if fail:
                    raise RuntimeError("startup failed")
                return "loaded"

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker",
                        SimpleNamespace(Worker=Worker))
    from vllm_compat.gpt_oss_worker import GptOssWorker
    try:
        worker = GptOssWorker()
        if fail:
            with pytest.raises(RuntimeError, match="startup failed"):
                worker.load_model(load_dummy_weights=True)
        else:
            assert worker.load_model(load_dummy_weights=True) == "loaded"
        assert worker.split_limit == 20
        assert marlin._repack_marlin_experts is original
    finally:
        sys.modules.pop("vllm_compat.gpt_oss_worker", None)


def test_cuda_marlin_packing_matches_installed_vllm():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA packing comparison requires the GPU host")
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
    from vllm_compat.gpt_oss_repack import repack_experts_in_place

    weight = torch.randint(0, 256, (3, 256, 64), dtype=torch.uint8, device="cuda")
    perm = torch.empty(0, dtype=torch.int32, device="cuda")
    original = marlin_utils_fp4._repack_marlin_experts
    expected = original(weight, 256, 128, perm, False)
    actual = repack_experts_in_place(
        weight.clone(), 256, 128, perm, False, repack=original)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


def test_cuda_gpt_oss_scratch_reuse_and_sleep_wake():
    """Exercise real expert sizes and the private allocator on the GPU host.

    With vLLM's original 20 MiB split limit, cached conversion segments grow
    every iteration. Repeated conversions must instead reuse bounded scratch,
    produce identical packed weights, and survive the normal sleep/wake path.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("private-pool memory regression requires the GPU host")
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
    from vllm_compat.gpt_oss_repack import repack_experts_in_place
    from vllm_compat.gpt_oss_worker import GptOssWorker

    # GPT-OSS-120B gate/up expert dimensions, large enough to use the 20 MiB
    # allocator segments that caused the production OOM (tiny tensors miss it).
    weight = torch.randint(0, 256, (3, 5760, 1440),
                           dtype=torch.uint8, device="cuda")
    perm = torch.empty(0, dtype=torch.int32, device="cuda")
    original = marlin_utils_fp4._repack_marlin_experts
    expected = original(weight, 5760, 2880, perm, False)
    allocator = CuMemAllocator.get_instance()
    worker = object.__new__(GptOssWorker)
    tag = "gpt_oss_repack_test"
    with (allocator.use_memory_pool(tag=tag),
          worker._scoped_allocator_max_split(max_split_size_mb=20)):
        storage = weight.clone()
        before = allocator.get_current_usage()
        for _ in range(32):
            storage.copy_(weight)
            actual = repack_experts_in_place(
                storage, 5760, 2880, perm, False, repack=original)
        torch.cuda.synchronize()
        assert allocator.get_current_usage() - before < 128 * 1024**2
    assert torch.equal(actual, expected)
    allocator.sleep(offload_tags=(tag,))
    allocator.wake_up(tags=[tag])
    assert torch.equal(actual, expected)
