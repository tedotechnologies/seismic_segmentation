#!/bin/bash

SCRIPT_PATH="/home/dmatveev/workdir/rosneft_segmentation/experiments/train.py"

PROJECT_NAME="Rosneft segmentation"
TASK_NAME='Experiment 10| select best mask (channel), with centre point | train on salt, test paleokarst'
EPOCHS=1
BATCH_SIZE=4
LR=1e-4
FREEZE_BASE="model.freeze_base=True"

python3 "$SCRIPT_PATH" \
    clearml.project_name="$PROJECT_NAME" \
    clearml.task_name="\"$TASK_NAME\"" \
    training.epochs="$EPOCHS" \
    training.batch_size="$BATCH_SIZE" \
    training.lr="$LR" \
    $FREEZE_BASE
