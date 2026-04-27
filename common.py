from pathlib import Path
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
from itertools import chain
import os

MODEL = "Qwen/Qwen2.5-0.5B"
DATA = "Elriggs/openwebtext-100k"

#ROOT = Path("/workspace")
ROOT = Path(".")
ASSETS = ROOT / "assets"
RUNS = ROOT / "runs"

ASSETS.mkdir(exist_ok=True)
RUNS.mkdir(exist_ok=True)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_model(path=ASSETS / "model", dtype=torch.bfloat16):
    try:
        src = path
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            src,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    except Exception:
        src = MODEL
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            src,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    return tok, model



def load_data(path=ASSETS / "openwebtext_100k", n=100000):
    try:
        ds = load_from_disk(path)
    except Exception:
        ds = load_dataset(DATA)

    if isinstance(ds, DatasetDict):
        ds = ds["train"]

    return ds.select(range(min(n, len(ds))))

def tokenize(ds, tok, seq_len=256, add_eos=True, left_trim=0, right_trim=0, batch_size=1000, num_proc=10):
    eos = tok.eos_token_id

    def f(x):
        y = tok(x["text"], add_special_tokens=False)
        if add_eos:
            y["input_ids"] = [ids + [eos] for ids in y["input_ids"]]
        return {"input_ids": y["input_ids"]}

    def group_texts(x):
        ids = list(chain.from_iterable(x["input_ids"]))
        n = len(ids) // seq_len * seq_len
        if n == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        ids = ids[:n]
        chunks = [ids[i:i + seq_len] for i in range(0, n, seq_len)]

        if left_trim == 0 and right_trim == 0:
            return {
                "input_ids": chunks,
                "attention_mask": [[1] * seq_len for _ in range(len(chunks))],
                "labels": [c[:] for c in chunks],
            }

        masks = [[1] * seq_len for _ in range(len(chunks))]
        labels = [c[:] for c in chunks]

        for i, c in enumerate(chunks):
            first = next((j for j, t in enumerate(c) if t == eos), None)
            last = next((j for j, t in enumerate(reversed(c)) if t == eos), None)

            if first is not None and left_trim > 0 and first < left_trim:
                for j in range(first):
                    masks[i][j] = 0
                    labels[i][j] = -100

            if last is not None:
                last = seq_len - 1 - last
                if right_trim > 0 and seq_len - 1 - last < right_trim:
                    for j in range(last + 1, seq_len):
                        masks[i][j] = 0
                        labels[i][j] = -100

        return {
            "input_ids": chunks,
            "attention_mask": masks,
            "labels": labels,
        }

    ds = ds.map(
        f,
        batched=True,
        batch_size=batch_size,
        remove_columns=ds.column_names,
        num_proc=num_proc,
    )

    ds = ds.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    ds.set_format(type="torch")
    return ds