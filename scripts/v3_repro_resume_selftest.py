"""Small deterministic interruption test against original packaged trainers."""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_repro_resume as runner


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(812)
    days = np.arange(20170101, 20170113)
    codes = [f"TEST{i}" for i in range(6)]
    frame = pd.DataFrame({"date": np.repeat(days, len(codes)), "code": codes * len(days)})
    for feature in runner.original.FEATURES:
        frame[feature] = rng.normal(size=len(frame))
    frame["future_return"] = rng.normal(size=len(frame))
    split = {"internal": frame.date.to_numpy() < days[7],
             "early_stop": frame.date.to_numpy() >= days[7],
             "outer_train": np.ones(len(frame), dtype=bool),
             "predict": np.zeros(len(frame), dtype=bool)}
    seq = {int(day): rng.normal(size=(len(codes),20,8)).astype(np.float32) for day in days}
    keys = {int(day): codes for day in days}
    report = {}
    for kind in ("tabm", "tcn"):
        if kind == "tabm":
            ref = runner.tab.train_fold(frame, split, 20261005)
        else:
            ref = runner.tcn.train_fold(frame, seq, keys, split, 20261005)
        folder = runner.RUN / "selftest" / kind
        binding = "synthetic-resume-test-v1"
        try:
            runner.train_nn(kind, frame, split, 20261005, seq, keys, folder, binding, interrupt=2)
            raise AssertionError("interruption did not happen")
        except runner.InterruptedForTest:
            pass
        resumed = runner.train_nn(kind, frame, split, 20261005, seq, keys, folder, binding)
        def equal(a,b):
            return all(torch.equal(v, b["model"].state_dict()[k]) for k,v in a["model"].state_dict().items())
        assert equal(ref,resumed), kind
        assert ref["best_epoch"] == resumed["best_epoch"]
        assert ref["history"].equals(resumed["history"])
        assert ref["design"] == resumed["design"]
        if kind == "tcn":
            assert ref["channel_stats"] == resumed["channel_stats"]
        # Restore completed outer model, then corrupt the latest outer checkpoint
        # and verify fallback to previous epoch gives exactly the same result.
        again = runner.train_nn(kind, frame, split, 20261005, seq, keys, folder, binding)
        assert equal(ref,again)
        files = sorted((folder / "outer").glob("epoch*.pt"), reverse=True)
        if len(files) == 2:
            files[0].write_bytes(b"deliberate corruption for test")
            fallback=runner.train_nn(kind,frame,split,20261005,seq,keys,folder,binding)
            assert equal(ref,fallback)
        report[kind]={"parameters_bit_identical":True,"history_bit_identical":True,
                      "best_epoch":ref["best_epoch"],"completed_outer_restore":True,
                      "corrupt_checkpoint_fallback_tested":len(files)==2}
    runner.write_json(runner.RUN / "resume_equivalence_selftest.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
