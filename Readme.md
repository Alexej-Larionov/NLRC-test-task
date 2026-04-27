# NLRC Test Task

This repository contains an experimental pipeline for comparing optimizer behavior during continued pretraining / fine-tuning of a small causal language model. The main comparison is between three modes: full AdamW, full Muon for transformer matrix weights with AdamW for auxiliary parameters, and a mixed setup where part of the transformer is optimized with Muon and the rest with AdamW.

The experiment logs training loss, processed tokens, throughput, forward/backward time, optimizer step time, optimizer state memory, checkpoints, and zero-shot evaluation results. The main output directory is `runs/`; the central run index is `runs/runs.jsonl`.

## Requirements

The recommended setup is Linux or WSL2 with an NVIDIA GPU, Docker, and NVIDIA Container Toolkit. The project is intended to run inside the provided Docker environment, which installs PyTorch, Transformers, Datasets, Accelerate, and `lm-evaluation-harness`.

Clone the repository:

```bash
git clone https://github.com/Alexej-Larionov/NLRC-test-task.git
cd NLRC-test-task
```

## Build and enter the Docker container:
```bash
chmod +x docker-run.sh
./docker-run.sh bash
```
Inside the container, download the cached model and dataset assets:

`python download_assets.py`

The assets are stored in assets/. Training outputs are stored in runs/.

## Training

Training is launched with train.py. The optimizer mode is selected with --mode, and training hyperparameters can be passed through a JSON config.
```bash
python train.py --mode adam --config Experiment2.json
python train.py --mode muon --config Experiment2.json
python train.py --mode mixed --config Experiment2.json
```
The config defines sequence length, batch size, gradient accumulation, total steps, warmup, AdamW learning rate, Muon learning rate, gradient clipping, checkpoint frequency, weight decay values, and Muon Newton-Schulz steps.

Example config:
```json
{
  "seed": 42,
  "seq_len": 1024,
  "batch_size": 2,
  "grad_acc": 40,
  "steps": 1318,
  "warmup": 66,
  "lr_adam": 0.00005,
  "lr_muon": 0.00001,
  "grad_clip": 1.0,
  "ckpt_every": 200,
  "wd_adam": 0.01,
  "wd_muon": 0.0,
  "ns_steps": 5
}
```
The effective number of tokens per optimizer update is:

tokens/update = batch_size × grad_acc × seq_len

For the config above, this is:

2 × 40 × 1024 = 81,920 tokens/update

Each run creates a directory like:

runs/<mode>_<timestamp>/
├── train.jsonl
├── ckpt_200/
├── ckpt_400/
└── final/

Each checkpoint contains the saved model and optimizer state. The training log contains per-step loss, tokens seen, throughput, forward/backward time, optimizer step time, and optimizer memory statistics.

## Evaluation

After training, run:

`python my_eval_lm.py`

The evaluation script reads runs/runs.jsonl, loads unevaluated checkpoints, runs lm-evaluation-harness, saves compact eval JSON files, and updates runs/runs.jsonl.

The default evaluation tasks are:

["piqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag"]

The script stores both raw average accuracy and a primary score. The primary score uses acc_norm when available and falls back to acc otherwise.

## Full pipeline

To run all optimizer modes and then evaluate the produced checkpoints:
```bash
chmod +x run_all.sh
./run_all.sh
```
If you want `run_all.sh` to use a specific config, make sure the train commands inside it include:
``` bash
python -u train.py --mode adam --config Experiment2.json
python -u train.py --mode muon --config Experiment2.json
python -u train.py --mode mixed --config Experiment2.json
```
## Notes

For analysis, compare training curves by tokens_seen, not by raw optimizer step. Different effective batch sizes produce different numbers of optimizer updates over the same token budget. For example, around 82k tokens/update gives roughly 1318 updates over 108M tokens, while around 492k tokens/update gives roughly 220 updates.

This matters because smaller effective batches apply learning-rate updates and weight decay more frequently. Therefore, changing effective batch size also changes the effective update pressure per token. Evaluation should be interpreted relative to the baseline pretrained model when possible, because lower training loss does not necessarily imply better zero-shot benchmark performance.