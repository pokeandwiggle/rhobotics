"""Tests for train-time RTC (arXiv 2512.05964): per-token flow time, the loss mask, and the prefixed sampler.

Everything runs on a tiny CPU flow model assembled without a VLM backbone: the VLM context is
passed as ``precomputed_hidden_state``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rho.policies.rho.configuration_rho import RhoConfig
from rho.policies.rho.rho_model import FlowMatchingModel
from rho.policies.rho.rho_policy import _apply_action_padding_masks

from .test_adaln_shared import BLOCKS, EXPERTS, VLM_DIM, D, make_config, make_expert_inputs
from .test_rho import _make_observation_policy, _observation_batch

CHUNK = 8
ACTION_DIM = 4
STATE_DIM = 5
DELAY = 3
BATCH = 3
N_IMG = 6
MODES = ["per_block", "shared", "shared_lora"]


def make_model(adaln_mode="shared", rtc_delay=DELAY, num_flow_samples=1, seed=0):
    """A VLM-less FlowMatchingModel with a real cross-attention expert and non-identity adaLN."""
    torch.manual_seed(seed)
    config = RhoConfig(
        device="cpu",
        dtype="float32",
        embed_dim=D,
        num_heads=4,
        num_blocks=BLOCKS,
        ff_dim=128,
        adaln_mode=adaln_mode,
        adaln_lora_rank=8,
        cross_attention_dim=VLM_DIM,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        num_steps=3,
        num_flow_samples=num_flow_samples,
        enable_gradient_checkpointing=False,
        log_hidden_state_stats=False,
        dropout_p=0.0,
        rtc_delay=rtc_delay,
    )
    model = object.__new__(FlowMatchingModel)
    nn.Module.__init__(model)
    model.config = config
    model.embed_dim = D
    model.max_action_dim = ACTION_DIM
    model.dtype = torch.float32
    model.device = "cpu"
    model.state_projector = nn.Linear(STATE_DIM, D)
    model.action_in_proj = nn.Linear(ACTION_DIM, D)
    model.action_head = nn.Linear(D, ACTION_DIM)
    model.action_time_mlp_in = nn.Linear(2 * D, D)
    model.action_time_mlp_out = nn.Linear(D, D)
    model.time_mlp_in = nn.Linear(D, D)
    model.time_mlp_out = nn.Linear(D, D)
    model.action_expert = EXPERTS["cross"][0](config)
    model.noise_policy = None
    model._use_layerwise_cross = False
    model._last_hidden_state_stats = {}
    for p in model.parameters():
        if not p.abs().any():  # zero-init modulation maps would hide the time conditioning
            nn.init.normal_(p, std=0.1)
    return model.eval()


def make_inputs(batch=BATCH, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {
        "state": torch.randn(batch, 1, STATE_DIM, generator=g),
        "actions": torch.randn(batch, CHUNK, ACTION_DIM, generator=g),
        "noise": torch.randn(batch, CHUNK, ACTION_DIM, generator=g),
        "time": torch.rand(batch, generator=g) * 0.9 + 0.05,
        "precomputed_hidden_state": (torch.randn(batch, N_IMG, D, generator=g), torch.ones(batch, N_IMG)),
    }


def run_forward(model, inputs, **kwargs):
    return model(
        None,
        None,
        inputs["state"],
        inputs["actions"],
        noise=inputs["noise"],
        time=inputs["time"],
        precomputed_hidden_state=inputs["precomputed_hidden_state"],
        **kwargs,
    )


def run_sample(model, inputs, **kwargs):
    return model.sample_actions(
        None,
        None,
        inputs["state"],
        noise=inputs["noise"],
        precomputed_hidden_state=inputs["precomputed_hidden_state"],
        **kwargs,
    )


@pytest.mark.parametrize("rtc_delay", [0, -1, CHUNK, CHUNK + 1])
def test_rtc_delay_outside_open_interval_is_refused(rtc_delay):
    with pytest.raises(ValueError, match="rtc_delay"):
        RhoConfig(chunk_size=CHUNK, n_action_steps=CHUNK, rtc_delay=rtc_delay)


@pytest.mark.parametrize("rtc_delay", [None, 1, CHUNK - 1])
def test_rtc_delay_inside_open_interval_is_kept(rtc_delay):
    assert RhoConfig(chunk_size=CHUNK, n_action_steps=CHUNK, rtc_delay=rtc_delay).rtc_delay == rtc_delay


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("attention_type", EXPERTS)
def test_expert_per_token_time_matches_per_example_when_uniform(attention_type, mode):
    expert_cls, _ = EXPERTS[attention_type]
    torch.manual_seed(0)
    expert = expert_cls(make_config(mode, attention_type)).eval()
    for p in expert.parameters():
        if not p.abs().any():
            nn.init.normal_(p, std=0.1)
    inputs, n_state = make_expert_inputs(attention_type)
    *context, time_emb = inputs[0], inputs[1], inputs[2]
    per_token = time_emb[:, None, :].expand(-1, n_state, -1)

    expected = expert(*context, time_emb)
    actual = expert(*context, per_token)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_expert_per_token_time_changes_only_what_it_conditions(mode):
    expert = make_model(mode).action_expert
    inputs, n_state = make_expert_inputs("cross")
    image_text, state, time_emb = inputs
    uniform = time_emb[:, None, :].expand(-1, n_state, -1).clone()
    shifted = uniform.clone()
    shifted[:, -1] += 1.0  # last token only

    out_uniform = expert(image_text, state, uniform)
    out_shifted = expert(image_text, state, shifted)

    assert not torch.allclose(out_uniform, out_shifted)


@pytest.mark.parametrize("num_flow_samples", [1, 3])
@pytest.mark.parametrize("mode", MODES)
def test_forward_with_delay_zero_matches_plain_path(mode, num_flow_samples):
    # Delay 0 puts every token at the example's own t: per-token time, all equal.
    plain = make_model(mode, rtc_delay=None, num_flow_samples=num_flow_samples)
    rtc = make_model(mode, rtc_delay=DELAY, num_flow_samples=num_flow_samples)
    inputs = make_inputs()

    expected = run_forward(plain, inputs)
    actual = run_forward(rtc, inputs, rtc_delay=torch.zeros(BATCH, dtype=torch.long))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_rtc_delay_none_sample_actions_ignores_prefix_machinery():
    plain = make_model(rtc_delay=None)
    rtc = make_model(rtc_delay=DELAY)
    inputs = make_inputs()

    expected = run_sample(plain, inputs)
    actual = run_sample(rtc, inputs)  # no prefix: the delay-0 mode

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_prefix_is_returned_verbatim_and_conditions_the_rest(mode):
    model = make_model(mode)
    inputs = make_inputs()
    prefix = torch.randn(BATCH, DELAY, ACTION_DIM)

    out = run_sample(model, inputs, prefix=prefix)

    assert out.shape == (BATCH, CHUNK, ACTION_DIM)
    assert torch.equal(out[:, :DELAY], prefix)
    assert not torch.allclose(out[:, DELAY:], run_sample(model, inputs)[:, DELAY:])


def test_prefix_does_not_alias_or_mutate_the_caller_noise():
    model = make_model()
    inputs = make_inputs()
    noise = inputs["noise"].clone()

    run_sample(model, inputs, prefix=torch.randn(BATCH, DELAY, ACTION_DIM))

    assert torch.equal(inputs["noise"], noise)


def test_prefix_is_refused_without_rtc_delay():
    model = make_model(rtc_delay=None)
    with pytest.raises(ValueError, match="rtc_delay"):
        run_sample(model, make_inputs(), prefix=torch.zeros(BATCH, DELAY, ACTION_DIM))


@pytest.mark.parametrize("length", [DELAY - 1, DELAY + 1])
def test_prefix_of_wrong_length_is_refused(length):
    model = make_model()
    with pytest.raises(ValueError, match="prefix must have shape"):
        run_sample(model, make_inputs(), prefix=torch.zeros(BATCH, length, ACTION_DIM))


def test_prefix_of_wrong_action_dim_is_refused():
    model = make_model()
    with pytest.raises(ValueError, match="action_dim"):
        run_sample(model, make_inputs(), prefix=torch.zeros(BATCH, DELAY, ACTION_DIM + 1))


def test_delay_override_is_refused_without_rtc_delay():
    model = make_model(rtc_delay=None)
    with pytest.raises(ValueError, match="rtc_delay"):
        run_forward(model, make_inputs(), rtc_delay=torch.ones(BATCH, dtype=torch.long))


@pytest.mark.parametrize("num_flow_samples", [1, 3])
def test_loss_ignores_prefix_steps_and_rescales_postfix(num_flow_samples):
    model = make_model(num_flow_samples=num_flow_samples)
    inputs = make_inputs()
    delays = torch.tensor([DELAY, 0, 5])
    raw = []
    model.action_head.register_forward_hook(lambda _m, _i, out: raw.append(out))

    losses = run_forward(model, inputs, rtc_delay=delays)

    # Prefix steps carry no loss, and the rest is rescaled by H / (H - d) per example.
    for i, d in enumerate(delays.tolist()):
        assert not losses[i, :d].any()
        assert losses[i, d:].all()
    v_t = raw[0].reshape(BATCH, num_flow_samples, CHUNK, ACTION_DIM)
    # Noise is shared across an example's flow samples here, and the explicit time too.
    unmasked = ((inputs["noise"] - inputs["actions"])[:, None] - v_t).square().mean(dim=1)
    for i, d in enumerate(delays.tolist()):
        expected = unmasked[i, d:] * CHUNK / (CHUNK - d)
        torch.testing.assert_close(losses[i, d:], expected, rtol=1e-5, atol=1e-6)


def test_loss_does_not_depend_on_prefix_noise():
    # At t=0 the prefix tokens carry the clean actions, so their noise enters neither x_t nor a
    # supervised target.
    model = make_model()
    inputs = make_inputs()
    delays = torch.full((BATCH,), DELAY)
    perturbed = dict(inputs, noise=inputs["noise"].clone())
    perturbed["noise"][:, :DELAY] += 10.0

    assert torch.equal(
        run_forward(model, inputs, rtc_delay=delays), run_forward(model, perturbed, rtc_delay=delays)
    )


def test_rtc_rescale_composes_with_action_padding_masks():
    model = make_model()
    inputs = make_inputs()
    delays = torch.full((BATCH,), DELAY)
    action_is_pad = torch.zeros(BATCH, CHUNK, dtype=torch.bool)
    action_is_pad[:, -2:] = True

    losses = run_forward(model, inputs, rtc_delay=delays)
    masked = _apply_action_padding_masks(losses, action_is_pad, None)

    assert not masked[:, :DELAY].any()
    assert not masked[:, -2:].any()
    assert masked[:, DELAY:-2].all()


def test_delay_is_drawn_per_example_from_the_zero_or_target_mix():
    model = make_model()
    torch.manual_seed(0)

    delays = model._draw_rtc_delay(4000, torch.device("cpu"))

    assert set(delays.unique().tolist()) == {0, DELAY}
    assert (delays == DELAY).float().mean().item() == pytest.approx(0.25, abs=0.03)


def test_policy_pads_the_prefix_to_the_model_width_and_forwards_it():
    policy = _make_observation_policy()
    seen = {}

    def fake_sample_actions(*_args, prefix=None, **_kwargs):
        seen["prefix"] = prefix
        return torch.zeros(1, CHUNK, policy.config.max_action_dim)

    policy.model = SimpleNamespace(sample_actions=fake_sample_actions)
    policy.prepare_prompt = lambda _batch: ["do the task"]  # the tokenizer backend is not built here
    batch = _observation_batch([torch.rand(1, 3, 6, 8), torch.rand(1, 3, 6, 8)])

    policy.sample_actions(batch, prefix=torch.ones(1, DELAY, 4))

    assert seen["prefix"].shape == (1, DELAY, policy.config.max_action_dim)
    assert seen["prefix"][..., :4].eq(1).all()
    assert not seen["prefix"][..., 4:].any()
