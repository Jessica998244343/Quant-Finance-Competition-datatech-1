"""Independent CPU reproduction with durable epoch checkpoints.

This is an audit runner, not submission code. Original ZIP sources stay intact.
Uses the packaged architecture, preprocessing, losses, exports and CatBoost
trainer. NN epoch scheduling is made resumable without changing any updates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "repro_20261012"
SUB = RUN / "original/submission"
sys.path.insert(0, str(SUB))
import train as original
import alpha_train_tabm as tab
import alpha_tcn_train as tcn


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    writer(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def write_json(path, obj):
    atomic(path, lambda p: p.write_text(json.dumps(obj, indent=2)))


def fingerprint():
    ref = json.loads((ROOT / "repro_20261010/reference.json").read_text())
    expected = ref["original_zip"]["sha256"]
    assert sha(ROOT / "releases/stage4_verified_ic/V3_full_pkg.zip") == expected
    for name, entry in ref["package_files"].items():
        assert sha(SUB / name) == entry["sha256"], name
    for name, h in ref["data_inputs"]["sha256_by_relpath"].items():
        assert sha(ROOT / "Source/Data" / name) == h, name
    import catboost, pyarrow, safetensors
    versions = {"numpy": np.__version__, "pandas": pd.__version__,
                "torch": torch.__version__, "catboost": catboost.__version__,
                "pyarrow": pyarrow.__version__, "safetensors": safetensors.__version__}
    assert versions == ref["environment"]["versions"], versions
    assert sys.version_info[:2] == (3, 11)
    content = {"zip_sha256": expected, "data": ref["data_inputs"]["sha256_by_relpath"],
               "versions": versions, "cat_threads": 6, "nn_threads": 2,
               "runner_sha256": sha(Path(__file__))}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    content["fingerprint"] = digest
    path = RUN / "identity.json"
    if path.exists():
        assert json.loads(path.read_text()) == content, "run identity changed"
    else:
        write_json(path, content)
    return digest


def verified_file(path, binding):
    marker = path.with_name(path.name + ".complete.json")
    if not path.exists() or not marker.exists():
        return False
    record = json.loads(marker.read_text())
    return record.get("binding") == binding and record.get("sha256") == sha(path)


def complete(path, binding):
    write_json(path.with_name(path.name + ".complete.json"),
               {"binding": binding, "sha256": sha(path), "bytes": path.stat().st_size})


def prepare(binding):
    cache = RUN / "cache"
    cache.mkdir(exist_ok=True)
    panel_path = cache / "raw_rebuilt_panel.parquet"
    if not verified_file(panel_path, binding):
        started = time.time()
        features, labels = original.alpha_panel.build_panel(ROOT / "Source/Data", membership_filter=True)
        panel = original.alpha_panel.merge_labels(features, labels)
        panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
        atomic(panel_path, lambda p: panel.to_parquet(p, index=False))
        complete(panel_path, binding)
        print("panel completed", panel.shape, "seconds", round(time.time()-started, 1), flush=True)
    else:
        panel = pd.read_parquet(panel_path)
        print("using verified raw-data rebuild", panel.shape, flush=True)
    seq_path = cache / "raw_rebuilt_sequences.npz"
    if not verified_file(seq_path, binding):
        started = time.time()
        sequences, keys = tcn.build_sequences(ROOT / "Source/Data")
        days = sorted(sequences)
        arrays = {"days": np.array(days), "codes": np.array([keys[d] for d in days]),
                  "sequences": np.stack([sequences[d] for d in days])}
        def save(p):
            with open(p, "wb") as f:
                np.savez(f, **arrays)
        atomic(seq_path, save)
        complete(seq_path, binding)
        print("sequences completed", "seconds", round(time.time()-started, 1), flush=True)
    else:
        with np.load(seq_path, allow_pickle=False) as a:
            days, codes, values = a["days"], a["codes"], a["sequences"]
        sequences = {int(d): values[i] for i, d in enumerate(days)}
        keys = {int(d): codes[i].tolist() for i, d in enumerate(days)}
        print("using verified sequences", len(sequences), flush=True)
    return panel, sequences, keys


class InterruptedForTest(Exception):
    pass


class Checkpoint:
    def __init__(self, root, binding, interrupt=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = binding
        self.interrupt = interrupt

    def load(self, model, optimizer, rng):
        paths = sorted(self.root.glob("epoch*.pt"), reverse=True)
        for path in paths:
            expected = path.stem.split("_")[-1]
            if sha(path) != expected:
                print("ignoring corrupt checkpoint", path, flush=True)
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["binding"] != self.binding:
                raise RuntimeError("checkpoint identity mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            rng.bit_generator.state = state["generator"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            print("restored", str(self.root.relative_to(RUN)) if RUN in self.root.parents else self.root,
                  "epoch", state["epoch"], flush=True)
            return state
        if paths:
            raise RuntimeError("all checkpoints corrupt; refusing silent restart")
        return {"epoch": 0, "history": [], "best": -np.inf, "best_epoch": 0}

    def save(self, model, optimizer, rng, epoch, history=None, best=-np.inf, best_epoch=0):
        state = {"binding": self.binding, "epoch": epoch, "history": history or [],
                 "best": best, "best_epoch": best_epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "generator": rng.bit_generator.state,
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate()}
        tmp = self.root / "checkpoint.partial"
        torch.save(state, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        path = self.root / f"epoch{epoch:04d}_{sha(tmp)}.pt"
        os.replace(tmp, path)
        sync_dir(self.root)
        old = sorted(self.root.glob("epoch*.pt"), reverse=True)
        for p in old[2:]:
            p.unlink()
        if self.interrupt == epoch:
            raise InterruptedForTest("simulated interruption after durable epoch")


def new_nn(kind, d_in, seed):
    if kind == "tabm":
        return tab._new(d_in, seed)
    torch.manual_seed(seed)
    model = tcn.TCNModel(d_in)
    return model, torch.optim.AdamW(model.parameters(), lr=tcn.LR, weight_decay=tcn.WD), np.random.default_rng(seed)


def epoch_nn(kind, model, optimizer, rng, tx, tz, groups, sequence):
    if kind == "tabm":
        tab._run(model, optimizer, rng, tx, tz, groups, 1)
        return
    model.train()
    order = rng.permutation(len(groups))
    for step in range(0, len(order), tcn.BATCH_DAYS):
        batch = [groups[i] for i in order[step:step+tcn.BATCH_DAYS]]
        optimizer.zero_grad()
        losses = [tcn.F.mse_loss(model(tx[g], sequence[g]), tz[g]) for g in batch]
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcn.CLIP)
        optimizer.step()


def predictions(kind, model, x, groups, sequence):
    if kind == "tabm":
        return tab._predict(model, x, groups)
    model.eval()
    out = np.zeros(len(x))
    with torch.no_grad():
        for group in groups:
            out[group] = model(torch.from_numpy(x[group]), sequence[group]).numpy()
    return out


def train_nn(kind, data, split, seed, sequences, keys, directory, binding, interrupt=None):
    torch.set_num_threads(2)
    dates = pd.to_numeric(data["date"]).astype("float64").to_numpy()
    label = pd.to_numeric(data["future_return"], errors="coerce")
    finite = label.notna().to_numpy()
    ranks = label.groupby(data["date"]).rank(method="average")
    counts = label.where(finite).groupby(data["date"]).transform("count")
    z = ((ranks-(counts+1.0)/2.0)/counts).to_numpy(dtype=float)
    tz = torch.from_numpy(np.nan_to_num(z))
    raw = label.to_numpy(dtype=float)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    design = tab.Design().fit(data.loc[internal])
    x = design.transform(data)
    sequence = None
    if kind == "tcn":
        stats = tcn._channel_stats(data, sequences, internal)
        sequence = tcn._sequence_tensor(data, sequences, keys, stats)
    model, optimizer, rng = new_nn(kind, x.shape[1], seed)
    ck = Checkpoint(directory / "inner", binding, interrupt)
    state = ck.load(model, optimizer, rng)
    history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
    max_epochs, patience = (tab.MAX_EPOCHS, tab.PATIENCE) if kind == "tabm" else (tcn.MAX_EPOCHS, tcn.PATIENCE)
    groups, valid = tab._groups(dates, internal), tab._groups(dates, early)
    tx = torch.from_numpy(x)
    if state["epoch"] == 0 or state["epoch"]-best_epoch < patience:
        for epoch in range(state["epoch"]+1, max_epochs+1):
            epoch_nn(kind, model, optimizer, rng, tx, tz, groups, sequence)
            ic = tab._daily_ic(raw, valid, predictions(kind, model, x, valid, sequence))
            history.append({"epoch": epoch, "valid_daily_ic": ic})
            if np.isfinite(ic) and ic > best:
                best, best_epoch = ic, epoch
            ck.save(model, optimizer, rng, epoch, history, best, best_epoch)
            print(kind, seed, "inner", epoch, "IC", round(ic, 9), "best epoch", best_epoch, flush=True)
            if epoch-best_epoch >= patience:
                break
    assert best_epoch > 0
    outer_design = tab.Design().fit(data.loc[outer])
    x_out = outer_design.transform(data)
    outer_stats = None
    if kind == "tcn":
        outer_stats = tcn._channel_stats(data, sequences, outer)
        sequence = tcn._sequence_tensor(data, sequences, keys, outer_stats)
    model2, optimizer2, rng2 = new_nn(kind, x_out.shape[1], seed)
    ck2 = Checkpoint(directory / "outer", binding)
    state2 = ck2.load(model2, optimizer2, rng2)
    assert state2["epoch"] <= best_epoch
    tx2 = torch.from_numpy(x_out)
    outer_groups = tab._groups(dates, outer)
    for epoch in range(state2["epoch"]+1, best_epoch+1):
        epoch_nn(kind, model2, optimizer2, rng2, tx2, tz, outer_groups, sequence)
        ck2.save(model2, optimizer2, rng2, epoch)
        print(kind, seed, "outer", epoch, "/", best_epoch, flush=True)
    return {"model": model2, "history": pd.DataFrame(history), "best_epoch": best_epoch,
            "best_inner_ic": best, "design": outer_design.to_json(), "channel_stats": outer_stats}


def run(binding):
    import fcntl
    lock = open(RUN / "run.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    panel, sequences, keys = prepare(binding)
    split = original.full_split(panel)
    out = RUN / "model"
    out.mkdir(exist_ok=True)
    stage = RUN / "stages"
    stage.mkdir(exist_ok=True)
    records = {}
    for seed in original.SEEDS:
        for kind in ("c1", "tabm", "tcn"):
            started = time.time()
            tag = f"{kind}_{seed}"
            path = out / (f"catboost_c1_seed{seed}.cbm" if kind == "c1" else f"{kind}_seed{seed}.npz")
            record_path = stage / f"{tag}.json"
            if verified_file(path, binding) and record_path.exists():
                records[tag] = json.loads(record_path.read_text())
                print("verified complete member", tag, flush=True)
                continue
            print("START", tag, flush=True)
            if kind == "c1":
                # Original trainer, fresh model. A short member (~3 minutes)
                # is retried as a whole if interrupted; no parameter changes.
                record = original.train_catboost(panel, split, seed, 6, out)
            else:
                result = train_nn(kind, panel, split, seed, sequences, keys,
                                  RUN / "checkpoints" / tag, binding)
                arrays = {k: v.detach().cpu().numpy() for k,v in result["model"].state_dict().items()}
                def save(p):
                    with open(p, "wb") as f:
                        (original.export_tabm_npz if kind == "tabm" else original.export_tcn_npz)(arrays, f)
                atomic(path, save)
                atomic(stage / f"{tag}_inner_curve.parquet", lambda p: result["history"].to_parquet(p, index=False))
                record = {k:v for k,v in result.items() if k not in ("model", "history")}
            record["seconds_this_attempt"] = time.time()-started
            write_json(record_path, record)
            complete(path, binding)
            records[tag] = record
            write_json(RUN / "progress.json", {"binding":binding,"completed_members":list(records),"latest":tag,"complete":False})
            print("COMPLETE", tag, record.get("best_epoch", record.get("selected_trees")), "seconds", round(time.time()-started,1), flush=True)
    tab_designs = [records[f"tabm_{s}"]["design"] for s in original.SEEDS]
    tcn_designs = [records[f"tcn_{s}"]["design"] for s in original.SEEDS]
    channel_stats = [records[f"tcn_{s}"]["channel_stats"] for s in original.SEEDS]
    assert all(d == tab_designs[0] for d in tab_designs)
    assert all(d == tcn_designs[0] for d in tcn_designs)
    assert all(d == channel_stats[0] for d in channel_stats)
    write_json(out / "tabm_design.json", tab_designs[0])
    write_json(out / "tcn_design.json", tcn_designs[0])
    write_json(out / "tcn_channel_stats.json", channel_stats[0])
    members = {"c1":[f"catboost_c1_seed{s}.cbm" for s in original.SEEDS],
               "tabm":[f"tabm_seed{s}.npz" for s in original.SEEDS],
               "tcn":[f"tcn_seed{s}.npz" for s in original.SEEDS]}
    hashes={n:sha(out/n) for ns in members.values() for n in ns}
    meta={"version":"V3_full_pkg","train_label_cutoff":original.TRAIN_END,"features":original.FEATURES,
          "weights":original.WEIGHTS,"members":members,"fusion_graph":"atom raw -> per-seed daily centred rank -> seed mean -> family weight -> final rank",
          "train_cutoff_note":"outer fit uses label_end_date <= 2022-12-30",
          "runtime":"NumPy inference for TabM/TCN; no torch required at run time",
          "member_hashes":hashes,"catboost_trees":{str(s):records[f"c1_{s}"]["actual_tree_count"] for s in original.SEEDS}}
    write_json(out/"meta.json", meta)
    write_json(RUN/"progress.json",{"binding":binding,"completed_members":list(records),"complete":True})
    print("ALL NINE MEMBERS COMPLETE",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    binding=fingerprint()
    if args.prepare_only:
        prepare(binding)
    else:
        run(binding)


if __name__ == "__main__":
    main()
