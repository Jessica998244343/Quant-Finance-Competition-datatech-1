"""V3 final-compliance P0-1 (resumable variant): finish the ``train.py`` run.

The delivered ``train.py`` has no checkpointing: one crash (this machine keeps
rebooting and wiping ``/tmp``) restarts all nine members.  This driver imports
the package's **own** ``train.py`` as a module and calls its exact functions with
the same arguments, seeds and threads - it only changes the scheduling: a member
already written to ``--out`` is skipped, so the full reproduction can be
completed across restarts.  The output file names, layout and hyper-parameters
are identical to a single ``train.py`` run; the report records that the run was
resumed rather than done in one uninterrupted process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_package(submission: Path, name: str = "_qf3_repro_pkg"):
    # train.py imports its helpers as top-level modules, exactly as it does when
    # run as a script; mirror that search path.
    if str(submission) not in sys.path:
        sys.path.insert(0, str(submission))
    spec = importlib.util.spec_from_file_location(
        name, submission / "train.py", submodule_search_locations=[str(submission)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--state", default=None)
    args = parser.parse_args()

    submission = Path(args.submission)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state) if args.state else out.parent / "repro_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"members": {}}
    started = time.time()

    module = load_package(submission)
    name = module.__name__
    alpha_panel = importlib.import_module(name + ".alpha_panel")
    alpha_train_tabm = importlib.import_module(name + ".alpha_train_tabm")
    alpha_tcn_train = importlib.import_module(name + ".alpha_tcn_train")

    def save_state() -> None:
        state_path.write_text(json.dumps(state, indent=2))

    print("building panel + split + sequences (identical to train.py) ...", flush=True)
    features, labels = alpha_panel.build_panel(args.data_dir, membership_filter=True)
    panel = alpha_panel.merge_labels(features, labels)
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    split = module.full_split(panel)
    sequences, keys = alpha_tcn_train.build_sequences(args.data_dir)
    print(f"panel {panel.shape}, signal days {split['outer_train'].sum()} rows", flush=True)

    for seed in module.SEEDS:
        c1_path = out / f"catboost_c1_seed{seed}.cbm"
        if not c1_path.exists():
            begin = time.time()
            record = module.train_catboost(panel, split, seed, args.threads, out)
            state["members"][f"c1_{seed}"] = {"trees": int(record["actual_tree_count"]),
                                              "seconds": round(time.time() - begin, 1)}
            print(f"trained c1 seed {seed}: {record['actual_tree_count']} trees "
                  f"({state['members'][f'c1_{seed}']['seconds']}s)", flush=True)
            save_state()
        else:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(str(c1_path))
            state["members"].setdefault(f"c1_{seed}", {"trees": int(model.tree_count_),
                                                       "resumed": True})
            print(f"c1 seed {seed}: already present", flush=True)

        tabm_path = out / f"tabm_seed{seed}.npz"
        if not tabm_path.exists():
            begin = time.time()
            module.train_tabm(panel, split, seed, out)
            state["members"][f"tabm_{seed}"] = {"seconds": round(time.time() - begin, 1)}
            print(f"trained tabm seed {seed} ({state['members'][f'tabm_{seed}']['seconds']}s)",
                  flush=True)
            save_state()
        else:
            state["members"].setdefault(f"tabm_{seed}", {"resumed": True})
            print(f"tabm seed {seed}: already present", flush=True)

        tcn_path = out / f"tcn_seed{seed}.npz"
        if not tcn_path.exists():
            begin = time.time()
            module.train_tcn(panel, sequences, keys, split, seed, out)
            state["members"][f"tcn_{seed}"] = {"seconds": round(time.time() - begin, 1)}
            print(f"trained tcn seed {seed} ({state['members'][f'tcn_{seed}']['seconds']}s)",
                  flush=True)
            save_state()
        else:
            state["members"].setdefault(f"tcn_{seed}", {"resumed": True})
            print(f"tcn seed {seed}: already present", flush=True)

    finite = pd.to_numeric(panel["future_return"], errors="coerce").notna().to_numpy()
    outer = split["outer_train"] & finite
    designs = [json.dumps(alpha_train_tabm.Design().fit(panel.loc[outer]).to_json(),
                          sort_keys=True) for _ in module.SEEDS]
    if len(set(designs)) != 1:
        raise RuntimeError("TabM designs differ across seeds")
    (out / "tabm_design.json").write_text(designs[0])
    (out / "tcn_design.json").write_text(json.dumps(
        alpha_train_tabm.Design().fit(panel.loc[outer]).to_json(), indent=2))
    (out / "tcn_channel_stats.json").write_text(json.dumps(
        {str(k): v for k, v in alpha_tcn_train._channel_stats(panel, sequences, outer).items()},
        indent=2))

    members = {"c1": [f"catboost_c1_seed{s}.cbm" for s in module.SEEDS],
               "tabm": [f"tabm_seed{s}.npz" for s in module.SEEDS],
               "tcn": [f"tcn_seed{s}.npz" for s in module.SEEDS]}
    hashes = {member: _sha(out / member) for group in members.values() for member in group}
    from catboost import CatBoostRegressor
    trees = {}
    for seed in module.SEEDS:
        model = CatBoostRegressor()
        model.load_model(str(out / f"catboost_c1_seed{seed}.cbm"))
        trees[str(seed)] = int(model.tree_count_)
    meta = {"version": "V3_full_pkg", "train_label_cutoff": module.TRAIN_END,
            "features": module.FEATURES, "weights": module.WEIGHTS, "members": members,
            "fusion_graph": "atom raw -> per-seed daily centred rank -> seed mean -> "
                            "family weight -> final rank",
            "train_cutoff_note": "outer fit uses label_end_date <= 2022-12-30",
            "runtime": "NumPy inference for TabM/TCN; no torch required at run time",
            "member_hashes": hashes, "catboost_trees": trees}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    state["total_seconds"] = round(time.time() - started, 1)
    state["completed"] = True
    save_state()
    print(json.dumps({"done": True, "members": members, "catboost_trees": trees,
                      "seconds_this_launch": state["total_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
