"""§4.5: prove the explicit TabM equals the reference implementation.

Compares parameter layout, initialization, eval/train forward passes, gradients
and one AdamW step between the explicit wrapper (`s4_tabm_wrapper_src`) and the
vendored Yandex reference (`s4_tabm.TabMWrapper` -> `vendor/tabm/tabm.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import s4_tabm_wrapper_src as explicit

SEED = 20261005
D_IN = 141
TABM = dict(k=8, width=128, blocks=2, dropout=0.1)


def _reference():
    vendor = Path("vendor/tabm").resolve()
    import sys
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from . import s4_tabm
    return s4_tabm.TabMWrapper


def _build(cls, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return cls(D_IN, **TABM)


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/stage4_verified_ic/v1_tabm_equiv.json")
    args = parser.parse_args()
    Reference = _reference()
    report: dict = {"k": TABM["k"], "width": TABM["width"], "blocks": TABM["blocks"],
                    "dropout": TABM["dropout"], "d_in": D_IN, "seed": SEED}

    ref, new = _build(Reference, SEED), _build(explicit.TabMWrapper, SEED)
    ref_sd, new_sd = ref.state_dict(), new.state_dict()
    report["state_dict_keys_equal"] = sorted(ref_sd) == sorted(new_sd)
    report["init_max_abs_diff"] = max(_max_diff(ref_sd[key], new_sd[key]) for key in ref_sd)
    report["init_bit_identical"] = report["init_max_abs_diff"] == 0.0

    x = torch.randn(300, D_IN)
    ref.eval(), new.eval()
    torch.manual_seed(101)
    eval_ref = ref(x)
    torch.manual_seed(101)
    eval_new = new(x)
    report["eval_forward_max_abs_diff"] = _max_diff(eval_ref, eval_new)

    ref.train(), new.train()
    torch.manual_seed(202)
    train_ref = ref(x)
    torch.manual_seed(202)
    train_new = new(x)
    report["train_forward_max_abs_diff"] = _max_diff(train_ref, train_new)

    target = torch.randn(300, TABM["k"])
    torch.manual_seed(303)
    ref.zero_grad()
    loss_ref = F.mse_loss(ref(x), target)
    loss_ref.backward()
    torch.manual_seed(303)
    new.zero_grad()
    loss_new = F.mse_loss(new(x), target)
    loss_new.backward()
    report["loss_abs_diff"] = abs(float(loss_ref.item()) - float(loss_new.item()))
    grad_diff = 0.0
    for (name, p_ref), (_, p_new) in zip(ref.named_parameters(), new.named_parameters()):
        grad_diff = max(grad_diff, _max_diff(p_ref.grad, p_new.grad))
    report["grad_max_abs_diff"] = grad_diff

    ref_opt, new_opt = _build(Reference, SEED), _build(explicit.TabMWrapper, SEED)
    for label, model, cls in (("ref", ref_opt, Reference), ("new", new_opt, explicit.TabMWrapper)):
        assert label in ("ref", "new")
    optimizers = {
        "ref": torch.optim.AdamW(ref_opt.parameters(), lr=1e-3, weight_decay=1e-3),
        "new": torch.optim.AdamW(new_opt.parameters(), lr=1e-3, weight_decay=1e-3),
    }
    models = {"ref": ref_opt, "new": new_opt}
    for step in range(3):
        for label in ("ref", "new"):
            torch.manual_seed(900 + step)
            models[label].train()
            optimizers[label].zero_grad()
            F.mse_loss(models[label](x), target).backward()
            optimizers[label].step()
    report["adamw_step_max_abs_diff"] = max(
        _max_diff(ref_opt.state_dict()[key], new_opt.state_dict()[key])
        for key in ref_opt.state_dict())

    report["forward_ops"] = "h = ((x * r) @ W.T) * s + b; ReLU; LinearEnsemble head; member mean"
    report["passed"] = bool(
        report["state_dict_keys_equal"] and report["init_bit_identical"]
        and report["eval_forward_max_abs_diff"] == 0.0
        and report["train_forward_max_abs_diff"] == 0.0
        and report["loss_abs_diff"] == 0.0 and report["grad_max_abs_diff"] == 0.0
        and report["adamw_step_max_abs_diff"] == 0.0)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
