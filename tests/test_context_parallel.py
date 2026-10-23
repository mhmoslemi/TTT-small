"""CPU distributed equivalence checks; no checkpoint download or GPU required."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

import context_parallel as cp


@pytest.mark.parametrize('split_k_supported', [False, True])
def test_bf16_shard_precision_restores_settings_even_on_failure(monkeypatch, split_k_supported):
    from contextlib import nullcontext
    import torch.nn.attention
    flags = SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
    if split_k_supported:
        flags.allow_bf16_reduced_precision_reduction_split_k = True
    library = ['cublas']

    def preferred(value=None):
        if value is not None:
            library[0] = value
        return library[0]

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.backends.cuda, 'matmul', flags)
    monkeypatch.setattr(torch.backends.cuda, 'preferred_blas_library', preferred)
    monkeypatch.setattr(torch.nn.attention, 'sdpa_kernel', lambda *args: nullcontext())
    with pytest.raises(RuntimeError, match='injected'):
        with cp._stable_bf16_matmul():
            assert library[0] == 'cublaslt'
            assert flags.allow_bf16_reduced_precision_reduction == (
                (False, False) if split_k_supported else False)
            raise RuntimeError('injected')
    assert library[0] == 'cublas'
    assert flags.allow_bf16_reduced_precision_reduction == (
        (True, True) if split_k_supported else True)


def test_calibration_shapes_keep_linear_values_gradients_and_restore():
    layer = nn.Linear(7, 5).double()
    value = torch.randn(2, 71, 7, dtype=torch.float64, requires_grad=True)
    expected = layer(value)
    expected.sum().backward()
    gradients = [value.grad.clone(), layer.weight.grad.clone(), layer.bias.grad.clone()]
    layer.zero_grad(set_to_none=True)
    value.grad = None
    with pytest.raises(RuntimeError, match='injected'):
        with cp._calibration_shapes(layer, 8):
            actual = layer(value)
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            for got, wanted in zip((value.grad, layer.weight.grad, layer.bias.grad), gradients):
                torch.testing.assert_close(got, wanted)
            raise RuntimeError('injected')
    assert 'forward' not in layer.__dict__


def example(key, tokens):
    return dict(_overlap_key=key, prompt_ids=SimpleNamespace(shape=(1, 17)),
                response_ids=SimpleNamespace(shape=(1, tokens - 16)))


def test_wave_sizes_use_observed_local_memory_and_reusable_groups():
    items = [example((0, i), tokens) for i, tokens in enumerate((108000, 90000, 40000, 31000))]
    first = cp.plan_context_wave(items, 8, 24000)
    assert len(first) == 1 and len(first[0][1]) == 8
    second = cp.plan_context_wave(items[1:], 8, 24000)
    assert [len(ranks) for _, ranks in second] == [4, 2, 2]
    assert {key for _, ranks in second for key in ranks} == set(range(8))
    assert all(ranks in cp.context_parallel_groups(8) for _, ranks in second)
    for tokens in (270000, 1000000):
        full = example((0, 0), tokens)
        assert cp.plan_context_wave([full], 8, 24000) == [(full, tuple(range(8)))]
    assert items[0]['response_ids'].shape[1] == 108000 - 16  # no truncation


def test_empty_or_short_tail_uses_the_available_gpus():
    assert cp.plan_context_wave([], 8, 24000) == []
    wave = cp.plan_context_wave([example((0, i), 30000) for i in range(3)], 8, 24000)
    assert sorted(len(ranks) for _, ranks in wave) == [2, 2, 4]


def test_other_models_and_objectives_do_not_enter_context_parallel():
    for profile, mode in [('gpt-oss-120b', 'spo-rs'), ('qwen3.8-27b', 'entropic')]:
        assert cp.availability(nn.Linear(2, 2), SimpleNamespace(
            coder_model_profile=profile, advantage_mode=mode), 8)[0] is False


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(31, 8)
        self.query, self.key, self.value = (nn.Linear(8, 8, bias=False) for _ in range(3))
        self.decay = nn.Linear(8, 2, bias=False)
        self.beta = nn.Linear(8, 2, bias=False)
        self.layers = nn.ModuleList([nn.Identity()])
        self.gradient_checkpointing = False
        self.context = None

    def forward(self, input_ids, position_ids=None, **kwargs):
        positions = (torch.arange(input_ids.shape[1])[None, :]
                     if position_ids is None else position_ids)
        hidden = self.embed_tokens(input_ids) + positions[..., None] * .005
        # Exercise hybrid recurrence plus attention, not attention alone.
        delta_args = [hidden.reshape(*hidden.shape[:2], 2, 4)] * 3
        delta_kwargs = dict(g=-torch.nn.functional.softplus(self.decay(hidden)),
                            beta=self.beta(hidden).sigmoid(), use_qk_l2norm_in_kernel=True)
        if self.context is None:
            recurrent, _ = _recurrent_delta(*delta_args, **delta_kwargs)
        else:
            recurrent, _ = cp.head_parallel_delta(
                *delta_args, **delta_kwargs, context=self.context, kernel=_recurrent_delta)
        hidden = hidden + recurrent.reshape_as(hidden)
        q, k, v = [module(hidden).reshape(*hidden.shape[:2], 2, 4).transpose(1, 2)
                   for module in (self.query, self.key, self.value)]
        if self.context is None:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            output = output.transpose(1, 2)
        else:
            output, _ = cp.attention(q, k, v, self.context)
        return SimpleNamespace(last_hidden_state=output.reshape_as(hidden))


class TinyModel(nn.Module):
    base_model_prefix = 'model'

    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.head = nn.Linear(8, 31, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.head


def _native_convolution(x, weight, bias=None, *, activation=None, seq_idx=None):
    # Emulate the native CUDA kernel's FP32 convolution + SiLU accumulator.
    dtype = x.dtype
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    result = torch.nn.functional.conv1d(
        x.to(compute_dtype), weight.to(compute_dtype).unsqueeze(1),
        None if bias is None else bias.to(compute_dtype),
        padding=weight.shape[-1] - 1, groups=weight.shape[0])[:, :, :x.shape[-1]]
    return torch.nn.functional.silu(result).to(dtype)


def test_bf16_halo_preserves_native_fused_rounding(monkeypatch):
    torch.manual_seed(19)
    x = torch.randn(1, 4, 128).to(torch.bfloat16)
    weight = torch.randn(4, 4).to(torch.bfloat16)
    expected = _native_convolution(x, weight, activation='silu')
    old = torch.nn.functional.silu(torch.nn.functional.conv1d(
        x, weight.unsqueeze(1), padding=3, groups=4)[:, :, :128])
    assert not torch.equal(expected, old), 'must exercise the premature BF16 rounding bug'
    tails = torch.cat((x[:, :, 61:64], x[:, :, 125:128]), dim=2)
    monkeypatch.setattr(cp, 'gather', lambda *args: tails)
    for rank in range(2):
        actual = cp.causal_convolution(
            x[:, :, rank * 64:(rank + 1) * 64], weight,
            activation='silu', kernel=_native_convolution,
            context=SimpleNamespace(group=None, rank=rank))
        torch.testing.assert_close(actual, expected[:, :, rank * 64:(rank + 1) * 64],
                                   rtol=0, atol=0)


def test_hybrid_context_keeps_native_forward_and_restores_kernels(monkeypatch):
    import model_backend

    layer = type('Qwen3_5GatedDeltaNet', (nn.Module,), {})()
    layer.causal_conv1d_fn = _native_convolution
    delta = lambda *args, **kwargs: kwargs
    layer.chunk_gated_delta_rule = delta
    native_forward = layer.forward
    model = nn.Sequential(nn.Embedding(3, 4), layer)
    model.get_input_embeddings = lambda: model[0]
    monkeypatch.setattr(cp.dist, 'get_rank', lambda *a: 0)
    monkeypatch.setattr(cp.dist, 'get_world_size', lambda *a: 2)
    checkpoint_modes = []
    monkeypatch.setattr(model_backend, 'set_gradient_checkpointing',
                        lambda model, enabled: checkpoint_modes.append(enabled))
    with pytest.raises(RuntimeError, match='test interruption'):
        with cp.hybrid_context(model, None) as context:
            assert layer.forward == native_forward
            assert layer.chunk_gated_delta_rule.func is cp.head_parallel_delta
            assert layer.chunk_gated_delta_rule.keywords == dict(context=context, kernel=delta)
            assert layer.causal_conv1d_fn.keywords['kernel'] is _native_convolution
            assert layer._ttt_context_parallel is context
            raise RuntimeError('test interruption')
    assert layer.forward == native_forward
    assert layer.causal_conv1d_fn is _native_convolution
    assert layer.chunk_gated_delta_rule is delta
    assert not hasattr(layer, '_ttt_context_parallel')
    assert checkpoint_modes == [True, False]


def _recurrent_delta(q, k, v, g, beta, *, initial_state=None,
                     output_final_state=False, cu_seqlens=None,
                     use_qk_l2norm_in_kernel=False):
    """Small independent reference, including state dependence across shards."""
    assert initial_state is None and not output_final_state and cu_seqlens is None
    if use_qk_l2norm_in_kernel:
        q, k = (torch.nn.functional.normalize(x, dim=-1) for x in (q, k))
    repeats = v.shape[2] // q.shape[2]
    q, k = (x.repeat_interleave(repeats, dim=2) for x in (q, k))
    state = v.new_zeros(v.shape[0], v.shape[2], k.shape[-1], v.shape[-1])
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        error = v[:, t] - (k[:, t, :, :, None] * state).sum(-2)
        state = state + k[:, t, :, :, None] * (beta[:, t, :, None] * error)[..., None, :]
        outputs.append((q[:, t, :, :, None] * state).sum(-2) / q.shape[-1] ** .5)
    return torch.stack(outputs, dim=1), None


def test_calibration_delta_preserves_grouped_heads_and_gradients():
    torch.manual_seed(41)
    q, k = [torch.randn(1, 9, 2, 3, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    v = torch.randn(1, 9, 4, 3, dtype=torch.float64, requires_grad=True)
    g = (-torch.rand(1, 9, 4, dtype=torch.float64)).requires_grad_()
    beta = torch.rand(1, 9, 4, dtype=torch.float64, requires_grad=True)
    expected, _ = _recurrent_delta(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    actual, _ = cp._calibration_delta(q, k, v, g=g, beta=beta, width=2,
                                     kernel=_recurrent_delta, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(actual, expected)
    inputs = (q, k, v, g, beta)
    for got, wanted in zip(torch.autograd.grad(actual.square().sum(), inputs),
                           torch.autograd.grad(expected.square().sum(), inputs)):
        torch.testing.assert_close(got, wanted)


def _distributed_checks(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        torch.manual_seed(91)
        context = SimpleNamespace(group=dist.group.WORLD, rank=rank, size=2)
        # Both equal-head and grouped-value layouts must retain the complete
        # recurrence and return later-token gradients to earlier token shards.
        for query_heads in (2, 4):
            inputs = [torch.randn(1, 128, query_heads, 3, dtype=torch.float64),
                      torch.randn(1, 128, query_heads, 3, dtype=torch.float64),
                      torch.randn(1, 128, 4, 5, dtype=torch.float64),
                      -torch.rand(1, 128, 4, dtype=torch.float64),
                      torch.rand(1, 128, 4, dtype=torch.float64)]
            full_inputs = [x.clone().requires_grad_() for x in inputs]
            local_inputs = [x[:, rank * 64:(rank + 1) * 64].clone().requires_grad_()
                            for x in inputs]
            expected, _ = _recurrent_delta(*full_inputs, use_qk_l2norm_in_kernel=True)
            actual, _ = cp.head_parallel_delta(
                *local_inputs, kernel=_recurrent_delta, context=context,
                use_qk_l2norm_in_kernel=True)
            torch.testing.assert_close(actual, expected[:, rank * 64:(rank + 1) * 64],
                                       rtol=1e-10, atol=1e-10)
            weights = torch.randn_like(expected)
            (expected * weights).sum().backward()
            (actual * weights[:, rank * 64:(rank + 1) * 64]).sum().backward()
            for full_input, local_input in zip(full_inputs, local_inputs):
                torch.testing.assert_close(local_input.grad,
                                           full_input.grad[:, rank * 64:(rank + 1) * 64],
                                           rtol=1e-9, atol=1e-9)

        source = torch.randn(1, 8, 4, dtype=torch.float64)
        weights = torch.randn_like(source)
        local = source[:, rank * 4:(rank + 1) * 4].clone().requires_grad_()
        full = source.clone().requires_grad_()
        full_heads = full.reshape(1, 8, 2, 2).transpose(1, 2)
        local_heads = local.reshape(1, 4, 2, 2).transpose(1, 2)
        expected = torch.nn.functional.scaled_dot_product_attention(
            full_heads, full_heads, full_heads, is_causal=True).transpose(1, 2).reshape_as(full)
        actual, _ = cp.attention(local_heads, local_heads, local_heads, context)
        actual = actual.reshape_as(local)
        torch.testing.assert_close(actual, expected[:, rank * 4:(rank + 1) * 4], rtol=1e-10, atol=1e-10)
        (expected * weights).sum().backward()
        (actual * weights[:, rank * 4:(rank + 1) * 4]).sum().backward()
        torch.testing.assert_close(local.grad, full.grad[:, rank * 4:(rank + 1) * 4], rtol=1e-10, atol=1e-10)

        # More ranks than KV heads: GQA replication must sum gradients back to
        # the original KV head and preserve the ordinary full-sequence kernel.
        qkv = [torch.randn(1, heads, 8, 4, dtype=torch.float64) for heads in (4, 1, 1)]
        full_qkv = [item.clone().requires_grad_() for item in qkv]
        local_qkv = [item[:, :, rank * 4:(rank + 1) * 4].clone().requires_grad_()
                     for item in qkv]
        from model_backend import _ttt_blockwise_attention_forward
        native = lambda q, k, v, mask, **kw: _ttt_blockwise_attention_forward(
            SimpleNamespace(training=False), q, k, v, mask, **kw)
        expected, _ = native(*full_qkv, None)
        actual, _ = cp.attention(*local_qkv, context, kernel=native)
        torch.testing.assert_close(actual, expected[:, rank * 4:(rank + 1) * 4], rtol=1e-10, atol=1e-10)
        head_weights = torch.randn_like(expected)
        (expected * head_weights).sum().backward()
        (actual * head_weights[:, rank * 4:(rank + 1) * 4]).sum().backward()
        for full_input, local_input in zip(full_qkv, local_qkv):
            torch.testing.assert_close(local_input.grad, full_input.grad[:, :, rank * 4:(rank + 1) * 4],
                                       rtol=1e-10, atol=1e-10)

        # The previous rank's convolution halo must also receive the gradient.
        convolution = nn.Conv1d(4, 4, 3, groups=4, padding=2, bias=False).double()
        local = source[:, rank * 4:(rank + 1) * 4].clone().requires_grad_()
        full = source.clone().requires_grad_()
        expected = torch.nn.functional.silu(convolution(full.transpose(1, 2))[:, :, :8]).transpose(1, 2)
        actual = cp.causal_convolution(
            local.transpose(1, 2), convolution.weight.squeeze(1),
            context=context, kernel=_native_convolution, activation='silu').transpose(1, 2)
        torch.testing.assert_close(actual, expected[:, rank * 4:(rank + 1) * 4], rtol=1e-10, atol=1e-10)
        (expected * weights).sum().backward()
        (actual * weights[:, rank * 4:(rank + 1) * 4]).sum().backward()
        torch.testing.assert_close(local.grad, full.grad[:, rank * 4:(rank + 1) * 4], rtol=1e-10, atol=1e-10)

        import train_multy_CVaR as training
        reference = TinyModel()
        parallel = deepcopy(reference)
        parallel.model.context = context
        from fast_distributed import reduce_trainable_gradients
        # Also exercise a padding-only rank: it must join all backward collectives.
        for response_length in (113, 5):
            reference.zero_grad(set_to_none=True)
            parallel.zero_grad(set_to_none=True)
            prompt = torch.randint(0, 31, (1, 17))
            response = torch.randint(0, 31, (1, response_length))
            cfg = SimpleNamespace(logprob_chunk=16, advantage_mode='spo-rs')
            expected, expected_entropy = training.compute_token_logprobs(
                reference, prompt, response, with_grad=True, chunk=16, return_entropy='measure')
            current, entropy = cp.token_logprobs(parallel, dict(prompt_ids=prompt, response_ids=response),
                                               SimpleNamespace(pad_token_id=0), cfg, context, measure=True)
            assert current.numel() == response.numel()
            torch.testing.assert_close(current, expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=1e-5)
            old = expected.detach() + .03
            options = dict(clip_epsilon=.2, clip_epsilon_low=.2, clip_epsilon_high=.38)
            loss, _ = training.clipped_policy_loss(cfg, expected, old, None, .73, **options)
            parallel_loss, _ = training.clipped_policy_loss(cfg, current, old, None, .73, **options)
            loss.backward()
            (parallel_loss / 2).backward()
            reduce_trainable_gradients(parallel, group=dist.group.WORLD)
            for first, second in zip(reference.parameters(), parallel.parameters()):
                if rank == 0:
                    torch.testing.assert_close(second.grad, first.grad, rtol=1e-4, atol=1e-6)
                else:
                    assert second.grad is None

        # Exercise the production calibration, including behavior-score setup.
        # Previously it discarded the supplied reference and did a second CP
        # forward for old scores, so the losses being compared were different.
        from contextlib import contextmanager
        import fast_distributed
        backend = SimpleNamespace(model=TinyModel())
        backend.set_training_mode = backend.model.train
        original_scores = cp.token_logprobs

        @contextmanager
        def tiny_context(model, group):
            model.model.context = context
            try:
                yield context
            finally:
                model.model.context = None

        def checked_scores(*args, **kwargs):
            assert kwargs.get('with_grad', True), 'calibration must reuse reference behavior scores'
            return original_scores(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(cp, 'hybrid_context', tiny_context)
            patch.setattr(cp, 'token_logprobs', checked_scores)
            patch.setattr(fast_distributed, 'set_long_rollout_memory_ceiling', lambda *args: None)
            sample = dict(prompt_ids=prompt, response_ids=response, advantage=.73, sample_weight=1.)
            assert cp.validate_context_group(backend, SimpleNamespace(pad_token_id=0), sample,
                                             cfg, dist.group.WORLD, (0, 1)) is None

            def bad_scores(*args, **kwargs):
                scores, entropy = checked_scores(*args, **kwargs)
                return scores + .2, entropy

            patch.setattr(cp, 'token_logprobs', bad_scores)
            assert cp.validate_context_group(backend, SimpleNamespace(pad_token_id=0), sample,
                                             cfg, dist.group.WORLD, (0, 1)) is not None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo unavailable')
def test_distributed_attention_halo_tokens_entropy_and_actual_spo_rs_gradients(tmp_path):
    mp.spawn(_distributed_checks, args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)


@pytest.mark.parametrize('invalid_kernel', [False, True, 'narrow'])
def test_coordinator_commits_each_trajectory_once_before_one_update(monkeypatch, tmp_path, invalid_kernel):
    import fast_distributed
    import train_multy_CVaR as training
    from exact_overflow import METRICS, pack_cpu_gradients

    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(10)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    keys = [(0, 1), (1, 1)]
    items = [example(key, 30000) for key in keys]
    stats = dict(totals={name: 1. for name in METRICS}, max_ratio=1.,
                 max_prefix_ratio=1., entropy_examples=0, backward_batches=1,
                 training_seconds=0., trained_examples=0)
    state = dict(step=2, remaining_keys=keys[:], deferred_keys=keys[:], completed_keys=[],
                 gradients={'weight': torch.full((1, 1), 2.)},
                 parameter_signature=fast_distributed.trainable_parameter_signature(model),
                 normal_stats={}, overflow_stats=deepcopy(stats), path=str(tmp_path / 'overflow.pt'))
    commands = {rank: [] for rank in range(1, 4)}
    trainer = SimpleNamespace(
        model=model, optimizer=optimizer, backend=SimpleNamespace(), tokenizer=None,
        world_size=4, _workers_idle=True, _cpu_examples=list,
        restore_after_generation=lambda: None,
        _command_queues={rank: SimpleNamespace(put=commands[rank].append) for rank in commands},
        _abort_workers=lambda: pytest.fail('unexpected abort'))

    last_width = [2]
    def compute(*args):
        if args:
            last_width[0] = len(args[-1])
        if invalid_kernel is True or (invalid_kernel == 'narrow' and last_width[0] < 4):
            return dict(completed=False, validation_issue='unsupported kernel')
        return dict(completed=True, stats=stats,
                    gradient_pack=pack_cpu_gradients({'weight': torch.full((1, 1), 4.)}))

    trainer._collect_event = lambda event, ranks: {
        rank: dict(result=compute()) for rank in ranks}
    monkeypatch.setattr(cp, 'availability', lambda *args: (True, None))
    monkeypatch.setattr(cp, 'initialize_groups', lambda size: dict.fromkeys(cp.context_parallel_groups(size)))
    monkeypatch.setattr(cp, 'compute_context_work', compute)
    monkeypatch.setattr(fast_distributed, 'broadcast_trainable_parameters', lambda *args, **kwargs: None)
    cfg = SimpleNamespace(advantage_mode='spo-rs', grad_clip=1e6)
    result = cp.complete_context_overflow(trainer, items, cfg, 2, state)
    assert trainer._warm_overflow_min_tokens == 30001  # source tokens plus final target
    if invalid_kernel is True:
        assert result is None and state['remaining_keys'] == keys
        assert model.weight.item() == 10  # calibration never applies a partial update
        assert state['gradients']['weight'].item() == 2
    else:
        assert state['remaining_keys'] == [] and set(state['completed_keys']) == set(keys)
        assert state['gradients']['weight'].item() == 10  # two leaders, not four ranks
        assert model.weight.item() == pytest.approx(9.9)
        assert result['exact_overflow_context_parallel']
        saved = torch.load(state['path'], weights_only=False)
        assert saved['completed_keys'] == state['completed_keys']
        if invalid_kernel == 'narrow':
            assert trainer._context_minimum_gpus == saved['context_minimum_gpus'] == 4
    assert trainer._workers_idle
