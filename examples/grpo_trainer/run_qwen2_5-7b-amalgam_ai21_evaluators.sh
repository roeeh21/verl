set -x

# Disable NCCL network plugins
export NCCL_NET_PLUGIN=none
# Disable CUDA memory pool
export NCCL_CUMEM_ENABLE=0
# Set NCCL debug level to warnings only
export NCCL_DEBUG=WARN
# Use all network interfaces except loopback
export NCCL_SOCKET_IFNAME=^lo
# Set NCCL timeout to 30 minutes
export NCCL_TIMEOUT=1800
# Set HuggingFace cache directory to RAM disk
export HF_HOME=/dev/shm
# Only score the assistant's response, not the full conversation
export VERL_SCORE_ONLY_ASSISTANT_RESPONSE=1
# Allow 10 minutes for Ray workers to register
export RAY_worker_register_timeout_seconds=600
# Set Ray RPC timeout to 100 seconds
export RAY_timeout_ms=100000
# Allow 60 seconds for Ray GCS server reconnection
export RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_S=60
# Set Ray heartbeat timeout to 30 seconds
export RAY_heartbeat_timeout_milliseconds=30000
# Enable Ray worker stdout to appear in logs
export RAY_DEDUP_LOGS=0

NAME="qwen2_5_7b_amalgam_ai21_evaluators_base"
LOCAL_SAVE_PATH="/dev/shm/$NAME"
SAVE_PATH=TODO # TODO: replace with your save path

if [ ! -d "$LOCAL_SAVE_PATH" ]; then
    mkdir -p $LOCAL_SAVE_PATH
fi


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=gs://ai21-algo-studio-research/verl_amalgam_experiments/amalgam_function_retrieval.json \
    data.val_files=gs://ai21-algo-studio-research/verl_amalgam_experiments/amalgam_function_retrieval.json \
    data.val_datasources_finite=True \
    data.train_datasources_finite=False \
    data.train_batch_size=16 \
    data.val_batch_size=1 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=False \
    data.custom_cls.path=verl/utils/dataset/amalgam_dataset.py \
    data.custom_cls.name=AmalgamDataset \
    data.cache_dir=/dev/shm/.gs_cache \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    algorithm.use_kl_in_reward=False \
    reward_model.train_num_examine=0 \
    reward_model.val_num_examine=1 \
    reward_model.reward_manager=ai21 \
    reward_model.ai21_evaluators_timeout=3.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_example_amalgam' \
    trainer.experiment_name=$NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.log_freq=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$LOCAL_SAVE_PATH \
    trainer.total_training_steps=100 \
    trainer.generations_save_freq=-1 \
    trainer.remote_save_path=$SAVE_PATH \
    trainer.val_examples_limit=10