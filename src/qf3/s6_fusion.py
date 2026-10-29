"""Canonical flat fusion with an explicit stage contract (task §6 frozen graph).

    per-seed atom scores  ->  q_g = mean_seed r_t(s_g)   (family stage)
    u = sum_g w_g * q_g                                  (weighted sum)
    output = r_t(u)                                      (final rank)

Every entry point checks its input:

* ``check_atom``  : unique ``(date, code)`` keys, finite scores, expected pool.
* ``family_q``    : per-seed checks, then returns a frame tagged ``stage=family_q``.
* ``as_family``   : validates a frame loaded from disk as a family-stage frame.
* ``fuse``        : requires tagged family inputs, finite non-negative weights
                    summing to 1, identical key sets, finite family values.

Renaming a final ``score`` column to ``q`` is therefore not enough: untagged
frames are rejected by ``fuse``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRAPH = "flat: u = sum_g w_g q_g ; output = rank_t(u); q_g = mean_seed rank_t(s_g)"
STAGE_FAMILY = "family_q"
STAGE_FINAL = "final_rank"
WEIGHT_TOL = 1e-9


class FusionContractError(ValueError):
    """Raised when the flat-fusion contract is violated."""


def daily_centred_rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


def _check_finite(values, label: str) -> None:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise FusionContractError(f"{label}: contains NaN/Inf")


def check_atom(frame: pd.DataFrame, value_column: str = "score",
               expected_keys: pd.DataFrame | None = None,
               label: str = "atom") -> pd.DataFrame:
    """Validate one seed's raw scores: keys, uniqueness, finiteness, exact pool."""
    for column in ("date", "code", value_column):
        if column not in frame.columns:
            raise FusionContractError(f"{label}: missing column {column}")
    if frame.duplicated(["date", "code"]).any():
        raise FusionContractError(f"{label}: duplicated (date, code) keys")
    if frame.empty:
        raise FusionContractError(f"{label}: empty frame")
    _check_finite(frame[value_column], f"{label}.{value_column}")
    if expected_keys is not None:
        wanted = set(map(tuple, expected_keys.loc[:, ["date", "code"]].to_numpy()))
        have = set(map(tuple, frame.loc[:, ["date", "code"]].to_numpy()))
        if wanted != have:
            missing, extra = len(wanted - have), len(have - wanted)
            raise FusionContractError(
                f"{label}: pool mismatch (missing {missing}, extra {extra})")
    return frame


def family_q(per_seed: list[pd.DataFrame], value_column: str = "score",
             expected_keys: pd.DataFrame | None = None,
             label: str = "family") -> pd.DataFrame:
    """q_g = mean over seeds of each seed's daily centred rank (tagged family stage)."""
    if not per_seed:
        raise FusionContractError(f"{label}: at least one seed is required")
    merged = None
    reference_keys = None
    for index, frame in enumerate(per_seed):
        checked = check_atom(frame, value_column, expected_keys,
                             label=f"{label} seed {index}")
        keys = set(map(tuple, checked.loc[:, ["date", "code"]].to_numpy()))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise FusionContractError(
                f"{label}: seed {index} key set differs from seed 0 "
                f"(missing {len(reference_keys - keys)}, extra {len(keys - reference_keys)})")
        part = checked.loc[:, ["date", "code", value_column]].rename(
            columns={value_column: f"s{index}"})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=["date", "code"], how="inner",
                                  validate="one_to_one")
            if len(merged) != len(part) or len(merged) != len(reference_keys):
                raise FusionContractError(f"{label}: seed pools differ")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = np.zeros(len(block))
        for index in range(len(per_seed)):
            value += daily_centred_rank(block[f"s{index}"].reset_index(drop=True)) / len(per_seed)
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "q": value}))
    out = pd.concat(pieces, ignore_index=True)
    _check_finite(out["q"], f"{label}.q")
    out.attrs["stage"] = STAGE_FAMILY
    return out


def as_family(frame: pd.DataFrame, column: str = "q",
              expected_keys: pd.DataFrame | None = None,
              label: str = "family") -> pd.DataFrame:
    """Validate and tag a family-stage frame loaded from disk (no reranking)."""
    if column not in frame.columns:
        raise FusionContractError(f"{label}: column {column} is not a family-stage value")
    out = frame.loc[:, ["date", "code", column]].rename(columns={column: "q"}).copy()
    if out.duplicated(["date", "code"]).any():
        raise FusionContractError(f"{label}: duplicated (date, code) keys")
    if out.empty:
        raise FusionContractError(f"{label}: empty frame")
    _check_finite(out["q"], f"{label}.q")
    if expected_keys is not None:
        wanted = set(map(tuple, expected_keys.loc[:, ["date", "code"]].to_numpy()))
        have = set(map(tuple, out.loc[:, ["date", "code"]].to_numpy()))
        if wanted != have:
            raise FusionContractError(
                f"{label}: pool mismatch (missing {len(wanted - have)}, "
                f"extra {len(have - wanted)})")
    out.attrs["stage"] = STAGE_FAMILY
    return out


def _check_weights(weights: dict[str, float], names) -> None:
    if set(weights) != set(names):
        raise FusionContractError(f"weight keys {sorted(weights)} != components {sorted(names)}")
    for name, value in weights.items():
        if not np.isfinite(value):
            raise FusionContractError(f"weight {name} is not finite: {value}")
        if value < 0:
            raise FusionContractError(f"weight {name} is negative: {value}")
    total = float(sum(weights.values()))
    if abs(total - 1.0) > WEIGHT_TOL:
        raise FusionContractError(f"weights must sum to 1, got {total}")


def fuse(components: dict[str, pd.DataFrame], weights: dict[str, float],
         column: str = "q") -> pd.DataFrame:
    """r_t(sum_g w_g * q_g) on the complete pool; no intermediate family rank."""
    if not components:
        raise FusionContractError("no components supplied")
    _check_weights(weights, components)
    merged = None
    for name, frame in components.items():
        if frame.attrs.get("stage") != STAGE_FAMILY:
            raise FusionContractError(
                f"component {name} is not tagged as the family stage; pass it through "
                f"family_q/as_family (a renamed final score is not accepted)")
        if column not in frame.columns:
            raise FusionContractError(f"component {name}: missing {column}")
        _check_finite(frame[column], f"component {name}.{column}")
        part = frame.loc[:, ["date", "code", column]].rename(columns={column: name})
        if merged is None:
            merged = part
        else:
            before = len(merged)
            merged = merged.merge(part, on=["date", "code"], how="outer",
                                  validate="one_to_one")
            if len(merged) != before or merged[name].isna().any():
                raise FusionContractError(f"component {name}: key set differs from the others")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = np.zeros(len(block))
        for name, weight in weights.items():
            value += weight * block[name].to_numpy()
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "score": daily_centred_rank(value)}))
    out = pd.concat(pieces, ignore_index=True)
    out.attrs["stage"] = STAGE_FINAL
    return out


def push_down_alpha(parent: dict[str, float], new_member: str, alpha: float) -> dict[str, float]:
    """Weight vector for adding a new member to a frozen parent combination."""
    if new_member in parent:
        raise FusionContractError(f"{new_member} already in the parent combination")
    if not np.isfinite(alpha) or alpha < 0 or alpha > 1:
        raise FusionContractError(f"alpha must be in [0, 1], got {alpha}")
    weights = {name: (1.0 - alpha) * value for name, value in parent.items()}
    weights[new_member] = alpha
    return weights
