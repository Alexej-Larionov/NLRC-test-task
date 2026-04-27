import json
import inspect
from pathlib import Path
from datetime import datetime

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


tasks = ["piqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag"]

runs_path = Path("runs/runs.jsonl")


def resolve_path(p, run_name=None):
    p = Path(p)
    qs = [p]

    s = str(p)
    if s.startswith("/workspace/"):
        qs.append(Path(s.removeprefix("/workspace/")))

    if run_name:
        qs += [
            Path("runs") / run_name / p.name,
            Path("runs") / run_name / "final",
        ]

    for q in qs:
        if q.exists():
            return q

    return qs[-1]


def metric(d, name):
    return d.get(name) or d.get(f"{name},none")


def pack_results(res):
    out = {}

    for task, d in res["results"].items():
        row = {
            "sample_len": d.get("sample_len"),
            "acc": metric(d, "acc"),
            "acc_stderr": metric(d, "acc_stderr"),
            "acc_norm": metric(d, "acc_norm"),
            "acc_norm_stderr": metric(d, "acc_norm_stderr"),
        }
        out[task] = {k: v for k, v in row.items() if v is not None}

    return out


def mean_metric(results, prefer_norm=False):
    vals = []

    for d in results.values():
        v = d.get("acc_norm") if prefer_norm else d.get("acc")
        if v is None:
            v = d.get("acc")
        if v is not None:
            vals.append(float(v))

    return sum(vals) / len(vals) if vals else None


def save_runs(runs):
    with open(runs_path, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


sig = inspect.signature(evaluator.simple_evaluate).parameters
runs = [json.loads(line) for line in open(runs_path, encoding="utf-8")]
updated = []

for r in runs:
    if r.get("eval") and r.get("eval_mean") is not None:
        updated.append(r)
        continue

    ckpt_dir = resolve_path(r["ckpt"], r.get("run_name"))
    ckpt = ckpt_dir / "model"

    if not ckpt.exists():
        print("missing:", ckpt)
        updated.append(r)
        continue

    print("load:", ckpt)

    lm = HFLM(
        pretrained=str(ckpt),
        trust_remote_code=True,
        dtype="bfloat16",
        device="cuda",
        batch_size="auto",
    )

    print("eval:", ckpt)

    kwargs = dict(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
    )

    if "log_samples" in sig:
        kwargs["log_samples"] = False

    if "cache_requests" in sig:
        kwargs["cache_requests"] = True

    if "bootstrap_iters" in sig:
        kwargs["bootstrap_iters"] = 0

    res = evaluator.simple_evaluate(**kwargs)
    results = pack_results(res)

    compact = {
        "run_name": r.get("run_name"),
        "mode": r.get("mode"),
        "step": r.get("step"),
        "ckpt": str(ckpt),
        "tasks": tasks,
        "num_fewshot": 0,
        "results": results,
        "eval_mean": mean_metric(results),
        "eval_mean_primary": mean_metric(results, prefer_norm=True),
    }

    out = ckpt / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    r["eval"] = out.as_posix()
    r["eval_mean"] = compact["eval_mean"]
    r["eval_mean_primary"] = compact["eval_mean_primary"]

    updated.append(r)
    save_runs(updated + runs[len(updated):])

    del lm, res
    torch.cuda.empty_cache()

    print("done:", ckpt)

save_runs(updated)