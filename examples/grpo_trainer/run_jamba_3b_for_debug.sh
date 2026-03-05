#!/bin/sh
export PYTHONNOUSERSITE=1 # use global python
export AI21_EVALUATORS_SEND_CPU_BOUND_TO_PROCESS_POOL="False" # don't send to process pool for easier debugging
MODEL_PATH=/dev/shm/jamba_3b
MODEL_REMOTE_PATH="gs://ai21-algo-studio-checkpoints/conversions/hf/model_name=sft-jamba-3b-nv-cp175-j17-a2-hotels-lr1e-05-seq32768-bs64/checkpoint=5000/"
# fix downloading without asterisk
if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading model..."
    mkdir -p "$MODEL_PATH"
    # gcloud storage cp with asterisk can't fail due to gs limitations, we copy with sub dir and move it one dir up
    gcloud storage cp -r "$MODEL_REMOTE_PATH" "$MODEL_PATH"
    # Get the actual directory name by splitting on '/' and taking the last part
    MODEL_ACTUAL_DIR=$(basename "$MODEL_REMOTE_PATH")
    echo "MODEL_ACTUAL_DIR: $MODEL_ACTUAL_DIR"
    mv $MODEL_PATH/$MODEL_ACTUAL_DIR/* $MODEL_PATH
    rm -rf $MODEL_PATH/$MODEL_ACTUAL_DIR
fi

GPUS=`nvidia-smi -L | wc -l`

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=3 \
    algorithm.filter_groups.num_times_to_snooze_easy_examples=5 \
    algorithm.filter_groups.snooze_mean_score_threshold=0.99 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=1.0 \
    algorithm.rollout_correction.rollout_rs=null \
    data.train_files=gs://ai21-mammoth-storage/users/michaelg/verl-data-configs/simple-data-synth-mix.json \
    data.val_files=gs://ai21-algo-studio-research/raza/verl_data_configs/20250910/collection.json \
    data.val_datasources_finite=True \
    data.train_datasources_finite=False \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.prompt_key=messages \
    data.return_raw_chat=True \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.custom_cls.path=verl/utils/dataset/amalgam_dataset.py \
    data.custom_cls.name=AmalgamDataset \
    data.cache_dir=/dev/shm/.gs_cache \
    actor_rollout_ref.actor.optim.lr=5e-06 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.filter_disagreement_logprobs.enable=True \
    actor_rollout_ref.actor.filter_disagreement_logprobs.threshold=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.cast_params_to_inference_dtype_before_update=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.move_left_padding_right=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.mamba_ssm_cache_dtype_fp32=True \
    +actor_rollout_ref.rollout.val_kwargs.seed=42 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.train_num_examine=0 \
    reward_model.val_num_examine=1 \
    reward_model.length_reward.enable=False \
    reward_model.length_reward.coeff=0.2 \
    reward_model.reward_manager=ai21 \
    reward_model.ai21_evaluators_timeout=120.0 \
    reward_model.ai21_clean_thinking_trace=True \
    reward_model.ai21_evaluators_batch_size=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=debug \
    trainer.experiment_name=debug_jamba_3b \
    trainer.n_gpus_per_node=${GPUS} \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.log_freq=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    trainer.val_examples_limit=200 \
    trainer.generations_save_freq=1 \
    trainer.max_rollouts_to_keep=1 \
    trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.default_local_dir=/dev/shm/checkpoints \
    trainer.val_before_train=False
