#!/bin/bash

IMAGE_NAME="nlrc-finetune"
CONTAINER_NAME="nlrc-experiment"

docker build -t $IMAGE_NAME .

echo ""
docker run --gpus all \
    --ipc=host \
    --rm \
    -it \
    --name $CONTAINER_NAME \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/lm-evaluation-harness:/workspace/lm-evaluation-harness \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    $IMAGE_NAME \
    "$@"