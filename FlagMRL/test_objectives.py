"""Numerical and integration checks for interchangeable flag objectives."""

import json

import pytest
import torch
from torch.nn import functional as F

from FlagMRL import FLAG_OBJECTIVES, FlagMRLLoss, FlagProjector
from FlagMRL.flag_math import orthogonality_error
from FlagMRL.pairs import build_pair_weights
from FlagMRL.train import parse_args


@pytest.mark.parametrize("name", ["trace_ratio", "spherical"])
@pytest.mark.parametrize("far_mode", ["uniform", "hard"])
def test_objective_matches_explicit_formula_and_gradient(name, far_mode):
    torch.manual_seed(7)
    q, p = [F.normalize(torch.randn(5, 8), dim=-1) for _ in range(2)]
    projector = FlagProjector(8, [2, 4, 8], init="random")
    near, far, valid = build_pair_weights(q, p, torch.arange(5), far_mode=far_mode)
    assert valid
    with torch.no_grad():
        loss, gradient, metrics = FLAG_OBJECTIVES[name](
            projector.U, projector.a_col, (q, p, near), (q, p, far),
            eps=1e-4, lambda_neg=0.7)

    U = projector.U.double().clone().requires_grad_()
    a = projector.a_col.double()
    if name == "trace_ratio":
        delta = q.double()[:, None, :] - p.double()[None, :, :]
        c_pos = torch.einsum("ij,ijd,ije->de", near.double(), delta, delta)
        c_neg = torch.einsum("ij,ijd,ije->de", far.double(), delta, delta)
        P = (U * a) @ U.T
        reference = -torch.trace(P @ c_neg) / (torch.trace(P @ c_pos) + 1e-4)
    else:
        x, y = [(v.double() @ U) * a.sqrt() for v in (q, p)]
        x, y = [v / (v.square().sum(-1, keepdim=True) + 1e-4).sqrt() for v in (x, y)]
        reference = ((x @ y.T) * (-near.double() + 0.7 * far.double())).sum()
    reference_gradient, = torch.autograd.grad(reference, U)
    torch.testing.assert_close(loss.double(), reference, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(gradient.double(), reference_gradient, atol=1e-6, rtol=1e-5)
    assert not loss.requires_grad and not gradient.requires_grad
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("name", ["trace_ratio", "spherical"])
def test_alternating_update_and_encoder_backward(name):
    torch.manual_seed(11)
    q, p = [torch.randn(5, 8, requires_grad=True) for _ in range(2)]
    projector = FlagProjector(8, [2, 4, 8])
    criterion = FlagMRLLoss(projector, flag_objective=name, update_every=2, lambda_cross=0.5)
    initial = projector.U.clone()
    loss, metrics = criterion(q, p)
    assert metrics["flag_skipped"] == 0
    assert not torch.equal(initial, projector.U)
    assert orthogonality_error(projector.U) < 1e-5
    assert q.grad is None and p.grad is None  # Flag update must not touch encoder gradients.
    loss.backward()
    assert torch.isfinite(q.grad).all() and q.grad.norm() > 0
    assert torch.isfinite(p.grad).all() and p.grad.norm() > 0
    updated = projector.U.clone()
    _, metrics = criterion(q, p)
    assert metrics["flag_skipped"] == 1
    assert torch.equal(updated, projector.U)
    criterion.eval()
    criterion(q, p)
    assert torch.equal(updated, projector.U)


def test_new_objective_needs_only_registry_entry(monkeypatch):
    calls = []

    def custom(U, a_col, near, far, *, eps, lambda_neg):
        assert not near[0].requires_grad and not far[1].requires_grad
        torch.testing.assert_close(near[0].norm(dim=-1), torch.ones(3))
        torch.testing.assert_close(near[2].sum(), torch.tensor(1.0))
        torch.testing.assert_close(far[2].sum(), torch.tensor(1.0))
        calls.append(True)
        return U.new_zeros(()), torch.zeros_like(U), {"custom_metric": U.new_tensor(2.)}

    monkeypatch.setitem(FLAG_OBJECTIVES, "custom", custom)
    args = parse_args(["--model_key", "bert", "--data_path", "unused.csv",
                       "--flag_objective", "custom"])
    criterion = FlagMRLLoss(FlagProjector(8, [2, 4, 8]), flag_objective=args.flag_objective)
    loss, metrics = criterion(torch.randn(3, 8, requires_grad=True), torch.randn(3, 8))
    loss.backward()
    assert calls == [True]
    assert metrics["custom_metric"] == 2.


@pytest.mark.parametrize("name", ["trace_ratio", "spherical"])
def test_all_duplicate_pool_skips_flag_update(name):
    projector = FlagProjector(8, [2, 4, 8])
    criterion = FlagMRLLoss(projector, flag_objective=name)
    before = projector.U.clone()
    loss, metrics = criterion(torch.randn(3, 8), torch.randn(3, 8),
                              pos_ids=torch.zeros(3, dtype=torch.long))
    assert metrics["flag_skipped"] == 1 and torch.isfinite(loss)
    assert torch.equal(before, projector.U)


@pytest.mark.parametrize("name", ["trace_ratio", "spherical"])
def test_training_warmup_and_checkpoint_roundtrip(name, tmp_path, monkeypatch):
    from transformers import BertConfig, BertModel
    from torch.utils.data import DataLoader
    import FlagMRL.train as training
    from FlagMRL.model import FlagTextModel

    encoder_path = tmp_path / "encoder"
    BertModel(BertConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16,
        hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2,
    )).save_pretrained(encoder_path)
    csv = tmp_path / "train.csv"
    csv.write_text("text\none\ntwo\nthree\nfour\n")
    samples = []
    for index in range(4):
        tokens = torch.tensor([2, index + 4, 3])
        samples.append({f"{key}{side}_stu": value.clone()
                        for side in (1, 2)
                        for key, value in [("input_ids", tokens),
                                           ("attention_mask", torch.ones_like(tokens))]})
    loader = DataLoader(samples, batch_size=2)
    monkeypatch.setattr(training, "build_train_loader", lambda *args: (loader, None, None))

    calls = []
    objective = FLAG_OBJECTIVES[name]

    def observed(U, a_col, near, far, **options):
        calls.append((near[0].clone(), near[1].clone()))
        return objective(U, a_col, near, far, **options)

    monkeypatch.setitem(FLAG_OBJECTIVES, name, observed)
    output = tmp_path / "checkpoint"
    args = parse_args([
        "--model_name", str(encoder_path), "--data_path", str(csv),
        "--output_dir", str(output), "--device", "cpu", "--epochs", "1",
        "--batch_size", "2", "--matryoshka_dims", "2", "4", "8",
        "--flag_objective", name, "--flag_init", "warmup", "--flag_init_batches", "1",
    ])
    trained = training.train(args).eval()
    assert len(calls) == 3  # One frozen-encoder warmup and two training batches.
    assert not torch.equal(*calls[0])  # Dropout is retained during warmup.
    config = json.loads((output / "training_config.json").read_text())
    assert config["flag_objective"] == name
    records = [json.loads(line) for line in (output / "training_metrics.jsonl").read_text().splitlines()]
    assert records[-1]["optimizer_step"] == 2
    metric = "flag_trace_ratio" if name == "trace_ratio" else "flag_gap"
    assert all(metric in record for record in records)
    loaded = FlagTextModel.from_pretrained(output)
    tokens = samples[0]["input_ids1_stu"][None]
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.sentence_embeddings(tokens, torch.ones_like(tokens)),
            trained.sentence_embeddings(tokens, torch.ones_like(tokens)),
        )
