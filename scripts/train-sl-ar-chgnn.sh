#!/usr/bin/env bash

RUN_NAME="sl-ar-var-50pnn-chgnn-max"

PROBLEM="tspsl"

DEVICES="0"
NUM_WORKERS=0

MIN_SIZE=20
MAX_SIZE=50

TRAIN_DATASET="data/tsp/tsp50_train_concorde.txt"
VAL_DATASET1="data/tsp/tsp20_test_concorde.txt"
VAL_DATASET2="data/tsp/tsp50_test_concorde.txt"
# VAL_DATASET3="data/tsp/tsp100_test_concorde.txt"

N_EPOCHS=10
EPOCH_SIZE=51200
BATCH_SIZE=16
ACCUMULATION_STEPS=1

VAL_SIZE=1280
ROLLOUT_SIZE=1280

MODEL="attention"
ENCODER="ch_gnn"
AGGREGATION="sum"
AGGREGATION_GRAPH="mean"
NORMALIZATION="layer"
EMBEDDING_DIM=64
N_ENCODE_LAYERS=3

LR_MODEL=0.0001
MAX_NORM=1
CHECKPOINT_EPOCHS=0

CUDA_VISIBLE_DEVICES="$DEVICES" python run.py --problem "$PROBLEM" \
    --model "$MODEL" \
    --min_size "$MIN_SIZE" --max_size "$MAX_SIZE" \
    --train_dataset "$TRAIN_DATASET" \
    --val_datasets "$VAL_DATASET1" "$VAL_DATASET2" \
    --epoch_size "$EPOCH_SIZE" \
    --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS" \
    --n_epochs "$N_EPOCHS" \
    --val_size "$VAL_SIZE" --rollout_size "$ROLLOUT_SIZE" \
    --encoder "$ENCODER" --aggregation "$AGGREGATION" \
    --n_encode_layers "$N_ENCODE_LAYERS" --gated \
    --normalization "$NORMALIZATION" --learn_norm \
    --embedding_dim "$EMBEDDING_DIM" --hidden_dim "$EMBEDDING_DIM" \
    --lr_model "$LR_MODEL" --max_grad_norm "$MAX_NORM" \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_epochs "$CHECKPOINT_EPOCHS" \
    --run_name "$RUN_NAME"