from time import time
from datetime import datetime
import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_cosine_schedule_with_warmup
from common import load_model, load_data, tokenize, RUNS, device, set_seed
from split import MUON, ADAM_BASE

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["adam", "muon", "mixed"], required=True)
p.add_argument("--config", default=None)
a = p.parse_args()

cfg = {
    "seed": 42,
    "seq_len": 1024,
    "batch_size": 2,
    "grad_acc": 40,
    "steps": 1318,
    "warmup": 66,
    "lr_adam": 5e-5,
    "lr_muon": 1e-5,
    "grad_clip": 1.0,
    "ckpt_every": 200,
    "wd_adam": 0.01,
    "wd_muon": 0.0,
    "ns_steps": 5,
}

if a.config:
    with open(a.config, encoding="utf-8") as f:
        cfg.update(json.load(f))

set_seed(cfg["seed"])

seq_len = cfg["seq_len"]
batch_size = cfg["batch_size"]
grad_acc = cfg["grad_acc"]
steps = cfg["steps"]
warmup = cfg["warmup"]
lr_adam = cfg["lr_adam"]
lr_muon = cfg["lr_muon"]
grad_clip = cfg["grad_clip"]
ckpt_every = cfg["ckpt_every"]
wd_adam = cfg["wd_adam"]
wd_muon = cfg["wd_muon"]
ns_steps = cfg["ns_steps"]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{a.mode}_{ts}"
run_dir = RUNS / run_name
run_dir.mkdir(parents=True, exist_ok=True)
log_path = run_dir / "train.jsonl"

tok, model = load_model()
model = model.to(device()).train()

ds = tokenize(load_data(), tok, seq_len)
dl = DataLoader(
    ds, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator
)

name_to_param = dict(model.named_parameters())

muon_p, adam_p = [], []

if a.mode == "adam":
    adam_p = list(name_to_param.values())

elif a.mode == "muon":
    muon_p = [name_to_param[n] for n in MUON]
    adam_p = [name_to_param[n] for n in ADAM_BASE if n in name_to_param]

    # lm_head если не tied
    if (
        "lm_head.weight" in name_to_param
        and name_to_param["lm_head.weight"]
        is not name_to_param["model.embed_tokens.weight"]
    ):
        adam_p.append(name_to_param["lm_head.weight"])

else:
    half = 24 // 2

    muon_names = [
        n for n in MUON if "model.layers." in n and int(n.split(".")[2]) < half
    ]
    adam_names = list(set(MUON) - set(muon_names)) + ADAM_BASE

    muon_p = [name_to_param[n] for n in muon_names]
    adam_p = [name_to_param[n] for n in adam_names if n in name_to_param]

    if (
        "lm_head.weight" in name_to_param
        and name_to_param["lm_head.weight"]
        is not name_to_param["model.embed_tokens.weight"]
    ):
        adam_p.append(name_to_param["lm_head.weight"])

all_p = set(name_to_param.values())
used = set(muon_p + adam_p)

assert all_p == used

opt_muon = torch.optim.Muon(muon_p, lr=lr_muon, weight_decay=wd_muon, ns_steps= ns_steps) if muon_p else None
opt_adam = (
    torch.optim.AdamW(adam_p, lr=lr_adam, betas=(0.9, 0.95), weight_decay=wd_adam)
    if adam_p
    else None
)

sch_muon = (
    get_cosine_schedule_with_warmup(opt_muon, warmup, steps) if opt_muon else None
)
sch_adam = (
    get_cosine_schedule_with_warmup(opt_adam, warmup, steps) if opt_adam else None
)


def append_run(d):
    with open(RUNS / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")


def save_ckpt(tag, step, tokens):
    d = run_dir / tag
    d.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(d / "model")
    tok.save_pretrained(d / "model")

    torch.save(
        {
            "step": step,
            "tokens_seen": tokens,
            "opt": {
                "muon": opt_muon.state_dict() if opt_muon else None,
                "adam": opt_adam.state_dict() if opt_adam else None,
            },
        },
        d / "trainer.pt",
    )

    append_run(
        {
            "run_name": run_name,
            "mode": a.mode,
            "step": step,
            "ckpt": str(d),
            "train_log": str(log_path),
            "eval": None,
        }
    )


def size_opt_mb(opt, param_ids=None):
    if not opt:
        return None

    total = 0
    for p, s in opt.state.items():
        if param_ids is not None and id(p) not in param_ids:
            continue

        for v in s.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()

    return total / 1024**2


hidden_ids = {id(name_to_param[n]) for n in MUON if n in name_to_param}
muon_ids = {id(p) for p in muon_p}
adam_ids = {id(p) for p in adam_p}


def get_mem(mode):
    res = {}

    if mode == "adam":
        res["adam_total"] = size_opt_mb(opt_adam)
        res["adam_hidden"] = size_opt_mb(opt_adam, adam_ids & hidden_ids)

    elif mode == "muon":
        res["muon_total"] = size_opt_mb(opt_muon)
        res["muon_hidden"] = size_opt_mb(opt_muon, muon_ids & hidden_ids)

        res["adam_total"] = size_opt_mb(opt_adam)
        res["adam_hidden"] = size_opt_mb(opt_adam, adam_ids & hidden_ids)

    else:  # mixed
        res["muon_total"] = size_opt_mb(opt_muon)
        res["muon_hidden"] = size_opt_mb(opt_muon, muon_ids & hidden_ids)

        res["adam_total"] = size_opt_mb(opt_adam)
        res["adam_hidden"] = size_opt_mb(opt_adam, adam_ids & hidden_ids)

    return res



def step_fn(mode, opt_muon, opt_adam, sch_muon, sch_adam, measure_time=True):
    t_muon = None
    t_adam = None

    if mode == "adam":
        if measure_time:
            t0 = time()
        opt_adam.step()
        if measure_time:
            torch.cuda.synchronize()
            t_adam = time() - t0

        sch_adam.step()
        opt_adam.zero_grad(set_to_none=True)

    else:
        if opt_muon:
            if measure_time:
                t0 = time()
            opt_muon.step()
            if measure_time:
                torch.cuda.synchronize()
                t_muon = time() - t0

        if opt_adam:
            if measure_time:
                t0 = time()
            opt_adam.step()
            if measure_time:
                torch.cuda.synchronize()
                t_adam = time() - t0

        if sch_muon:
            sch_muon.step()
        if sch_adam:
            sch_adam.step()

        if opt_muon:
            opt_muon.zero_grad(set_to_none=True)
        if opt_adam:
            opt_adam.zero_grad(set_to_none=True)

    return t_muon, t_adam

print("params total:", len(name_to_param))
print("muon params:", len(muon_p))
print("adam params:", len(adam_p))
print("hidden params:", len(hidden_ids))
print("muon hidden:", len(muon_ids & hidden_ids))
print("adam hidden:", len(adam_ids & hidden_ids))

it = iter(dl)
tokens_seen = 0

with open(log_path, "a", encoding="utf-8") as log_f:
    for step in range(1, steps + 1):
        loss_sum = 0.0
        t_fwd = 0.0
        t_bwd = 0.0

        for _ in range(grad_acc):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)

            b = {k: v.to(device()) for k, v in b.items()}

            t0 = time()
            out = model(**b)
            t_fwd += time() - t0

            t0 = time()
            loss = out.loss / grad_acc
            loss.backward()
            t_bwd += time() - t0

            loss_sum += loss.item()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        t_muon, t_adam = step_fn(a.mode, opt_muon, opt_adam, sch_muon, sch_adam)

        toks = batch_size * grad_acc * seq_len
        tokens_seen += toks

 
        if step % 2 == 0:
            mem = get_mem(a.mode)
        else:
            mem = {}

        row = {
            "step": step,
            "loss": loss_sum,
            "tokens_seen": tokens_seen,
            "tokens_per_sec": toks / (t_fwd + t_bwd + (t_muon or 0) + (t_adam or 0)),
            "forward_time": t_fwd,
            "backward_time": t_bwd,
            "opt_step_muon": t_muon,
            "opt_step_adam": t_adam,
            "opt_mem_muon_total": mem.get("muon_total"),
            "opt_mem_muon_hidden": mem.get("muon_hidden"),
            "opt_mem_adam_total": mem.get("adam_total"),
            "opt_mem_adam_hidden": mem.get("adam_hidden"),
        }

        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

        print(f"{step} loss={loss_sum:.4f} fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s")

        if step % ckpt_every == 0:
            save_ckpt(f"ckpt_{step}", step, tokens_seen)

save_ckpt("final", step, tokens_seen)
