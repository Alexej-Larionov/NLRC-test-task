FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "transformers==4.46.0" \
    datasets \
    accelerate \
    evaluate \
    sentencepiece \
    einops



RUN git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness /opt/lm-evaluation-harness \
 && python -m pip install --no-cache-dir -e /opt/lm-evaluation-harness


WORKDIR /workspace
COPY . /workspace

