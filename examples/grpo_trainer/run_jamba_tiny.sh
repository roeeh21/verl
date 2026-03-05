#!/bin/bash

set -e

export HYDRA_FULL_ERROR=1
export NCCL_NET_PLUGIN=none
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
# export VERL_DEBUG_PORT=5695

# Model configuration
MODEL_NAME="jamba-tiny-dev"
MODEL_GS_PATH="gs://ai21-mammoth-storage/models/jamba_hf/tiny_dev"
IS_JAMBA=true

# Training configuration
NODES=1
GPUS_PER_NODE=2
VLLM_TP_SIZE=$GPUS_PER_NODE
MICRO_BATCH_SIZE=1
TOTAL_EPOCHS=15
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=1024
SAVE_FREQ=100
N=8
LR=5e-6
KL_LOSS_COEF=1e-4
ENTROPY_COEFF=0.05

# Experiment configuration
PROJECT_NAME="verl"
EXPERIMENT_NAME="dev-test"
REMOTE_LOAD_PATH=${REMOTE_LOAD_PATH:-null}
REMOTE_LOAD_STEP=${REMOTE_LOAD_STEP:-null}
REMOTE_SAVE_PATH=${REMOTE_SAVE_PATH:-null}
LOCAL_SAVE_PATH="/dev/shm/training_outputs"

# Data paths
REMOTE_TRAIN_DATA_PATH=""
REMOTE_VAL_DATA_PATH=""

# Download model if needed
if [ -n "$MODEL_GS_PATH" ]; then
    MODEL_PATH=/dev/shm/jamba_tiny
    mkdir -p $MODEL_PATH
    gcloud storage cp "${MODEL_GS_PATH}"/* $MODEL_PATH
else
    MODEL_PATH=$MODEL_NAME
fi

export HF_HOME=/dev/shm
export PYTHONPATH=.

# Download data
if [ -n "$REMOTE_TRAIN_DATA_PATH" ]; then
    echo "Downloading train data"
    gcloud storage cp "$REMOTE_TRAIN_DATA_PATH" $HOME/data/train.parquet
    echo "Downloading val data"
    gcloud storage cp "$REMOTE_VAL_DATA_PATH" $HOME/data/test.parquet
else
    python examples/data_preprocess/gsm8k.py --local_dir ~/data
fi


# Build the training command
TRAIN_CMD="python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/test.parquet \
    data.train_batch_size=$((NODES*GPUS_PER_NODE*MICRO_BATCH_SIZE)) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.shuffle=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.move_left_padding_right=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((NODES*GPUS_PER_NODE*MICRO_BATCH_SIZE)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( $MAX_PROMPT_LENGTH+$MAX_RESPONSE_LENGTH )) \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$MODEL_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$SAVE_FREQ \
    trainer.val_before_train=False \
    trainer.log_freq=1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.nnodes=$NODES \
    trainer.remote_load_path=$REMOTE_LOAD_PATH \
    trainer.remote_load_step=$REMOTE_LOAD_STEP \
    trainer.remote_save_path=$REMOTE_SAVE_PATH \
    trainer.default_local_dir=$LOCAL_SAVE_PATH"

if [ "$IS_JAMBA" = true ]; then
    TRAIN_CMD="$TRAIN_CMD actor_rollout_ref.rollout.quantization=experts_int8"
    TRAIN_CMD="$TRAIN_CMD actor_rollout_ref.rollout.enable_prefix_caching=False"
fi

# Execute the training command
eval $TRAIN_CMD
