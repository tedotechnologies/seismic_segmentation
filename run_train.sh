#!/bin/bash

SCRIPT_PATH="/home/dmatveev/workdir/rosneft_segmentation/experiments/train.py"

PROJECT_NAME="Rosneft segmentation"
TASK_NAME="Experiment 4"
EPOCHS=1
BATCH_SIZE=4
LR=1e-4
FREEZE_BASE="--freeze_base"

python3 "$SCRIPT_PATH" \
    --project_name "$PROJECT_NAME" \
    --task_name "$TASK_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    $FREEZE_BASE
