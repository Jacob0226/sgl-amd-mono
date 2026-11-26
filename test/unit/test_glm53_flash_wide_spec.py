"""GLM-5.3-Flash's wide-launch spec on CPU: the env that picks it over the narrow
spec, the widths it takes, its config and model refusal reasons, and the steps its
forward hands back to the model's forward. SGLang checks its declared requirements,
the same way for every spec."""

import importlib
from types import SimpleNamespace

import pytest
import torch

import sgl_amd_mono
from sgl_amd_mono.glm53_flash import HF_CONFIG, LINEAR_ATTN_CONFIG, wide
from sgl_amd_mono.glm53_flash import wide_spec as ws

TEXT = SimpleNamespace(**HF_CONFIG, linear_attn_config=dict(LINEAR_ATTN_CONFIG))
FLASH = SimpleNamespace(text_config=TEXT)
ARCH = ("Glm5NextForConditionalGeneration", None)


@pytest.mark.parametrize("env, path", [("1", "wide_spec"), (None, "spec")])
def test_the_env_picks_the_wide_spec(monkeypatch, env, path):
    if env is None:
        monkeypatch.delenv("SGLANG_AMD_MONO_GLM53_FLASH_WIDE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_AMD_MONO_GLM53_FLASH_WIDE", env)
    try:
        assert importlib.reload(sgl_amd_mono).SPECS[ARCH] == (
            f"sgl_amd_mono.glm53_flash.{path}:SPEC"
        )
    finally:
        monkeypatch.delenv("SGLANG_AMD_MONO_GLM53_FLASH_WIDE", raising=False)
        importlib.reload(sgl_amd_mono)


def test_the_widths_are_every_request_count_past_the_narrow_launch():
    L = wide.SUPPORTED_L[0]
    widths = ws.SPEC.devices["gfx950"].widths
    assert widths == tuple(L * r for r in range(3, max(wide.REQ_BUCKETS) + 1))
    assert min(widths) > wide.NARROW and max(widths) == L * max(wide.REQ_BUCKETS)
    assert ws.SPEC.devices["gfx950"].tp_sizes == (wide.TP,)


def _cfg(topk=1, steps=5, draft=6):
    return SimpleNamespace(
        speculative_eagle_topk=topk,
        speculative_num_steps=steps,
        speculative_num_draft_tokens=draft,
    )


def _dtypes(temporal=torch.bfloat16, conv=torch.bfloat16):
    return lambda config: SimpleNamespace(temporal=temporal, conv=conv)


@pytest.mark.parametrize("temporal", [torch.float32, torch.bfloat16])
def test_mtp5_on_either_state_dtype_is_served(monkeypatch, temporal):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes(temporal))
    monkeypatch.setattr(ws.importlib.util, "find_spec", lambda name: object())
    assert ws.SPEC.config_disable_reasons(_cfg(), FLASH) == []


@pytest.mark.parametrize(
    "cfg, temporal, conv, want",
    [
        (_cfg(steps=3, draft=4), torch.bfloat16, torch.bfloat16, "one draft chain"),
        (_cfg(topk=4, draft=8), torch.bfloat16, torch.bfloat16, "one draft chain"),
        (_cfg(draft=5), torch.bfloat16, torch.bfloat16, "one draft chain"),
        (_cfg(), torch.float16, torch.bfloat16, "KDA state in f32 or bf16"),
        (_cfg(), torch.bfloat16, torch.float32, "conv state in bf16"),
    ],
)
def test_config_reasons(monkeypatch, cfg, temporal, conv, want):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes(temporal, conv))
    monkeypatch.setattr(ws.importlib.util, "find_spec", lambda name: object())
    reasons = ws.SPEC.config_disable_reasons(cfg, FLASH)
    assert len(reasons) == 1 and want in reasons[0], reasons


def test_config_reasons_name_another_model_and_a_missing_flydsl(monkeypatch):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    monkeypatch.setattr(ws.importlib.util, "find_spec", lambda name: None)
    other = SimpleNamespace(
        text_config=SimpleNamespace(
            **{**vars(TEXT), "hidden_size": 6144},
        )
    )
    joined = " | ".join(ws.SPEC.config_disable_reasons(_cfg(), other))
    assert "hidden_size = 4096" in joined and "needs FlyDSL" in joined, joined


def test_model_reasons_refuse_other_tp_sizes(monkeypatch):
    import sglang.srt.runtime_context as runtime_context

    monkeypatch.setattr(
        runtime_context, "get_parallel", lambda: SimpleNamespace(tp_size=2)
    )
    model = SimpleNamespace(
        is_mrope_enabled=False, model=SimpleNamespace(layers=[], layers_to_capture=None)
    )
    joined = " | ".join(ws.SPEC.model_disable_reasons(model, FLASH, {}))
    assert f"built for TP{wide.TP}" in joined and "no KDA layer" in joined, joined


class _LM:
    def __init__(self):
        self.ran = []

    def forward(self, input_ids, positions, forward_batch, input_embeds, pp):
        self.ran.append("model")
        return "model"


@pytest.mark.parametrize(
    "rows, tokens, embeds",
    [(wide.NARROW, 6, None), (18, 4, None), (18, 6, torch.zeros(1))],
)
def test_narrow_steps_other_chains_and_embeddings_run_the_model_s_forward(
    rows, tokens, embeds
):
    lm = _LM()
    built = ws._Built(lm, SimpleNamespace(L=6))
    plan = SimpleNamespace(key=SimpleNamespace(rows=rows, tokens_per_request=tokens))
    out = built.forward(plan, None, None, SimpleNamespace(), embeds)
    assert out == "model" and lm.ran == ["model"]
