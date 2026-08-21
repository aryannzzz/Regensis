from __future__ import annotations

from regenesis_cmi.synthetic.causal_generator import build_oracle_model
from regenesis_cmi.synthetic.oracle import monte_carlo_cmi


def oracle(model, **kwargs) -> float:
    return monte_carlo_cmi(
        model,
        seed=17,
        precision_bits=1.0,
        min_samples=10_000,
        max_samples=10_000,
        **kwargs,
    ).estimate_bits


def test_artifact_only_collapses_under_true_c(small_config) -> None:
    model = build_oracle_model(small_config, "artifact_only", "post", seed=42)
    assert oracle(model) > 0.1
    assert abs(oracle(model, control="true_c")) < 0.02
    assert abs(oracle(model, eeg_variant="clean")) < 0.02


def test_genuine_cortical_survives_true_c(small_config) -> None:
    model = build_oracle_model(small_config, "genuine_cortical", "post", seed=42)
    assert oracle(model, control="true_c") > 0.1


def test_mixed_world_is_smaller_but_positive_after_true_c(small_config) -> None:
    model = build_oracle_model(small_config, "cortical_plus_artifact", "post", seed=42)
    raw = oracle(model)
    controlled = oracle(model, control="true_c")
    assert raw > controlled > 0.02


def test_proxy_quality_controls_residual_artifact(small_config) -> None:
    model = build_oracle_model(small_config, "artifact_only", "post", seed=42)
    perfect = oracle(model, control="proxy", proxy_quality=1.0)
    medium = oracle(model, control="proxy", proxy_quality=0.7)
    absent = oracle(model, control="proxy", proxy_quality=0.0)
    assert abs(perfect) < 0.02
    assert perfect < medium < absent

