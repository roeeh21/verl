#!/bin/sh

MODEL_PATH=/dev/shm/qwen2_5_3b

if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    gcloud storage cp -r "gs://ai21-publishing-huggingface-models/huggingface_models/qwen-qwen25-3b-instruct/*" "$MODEL_PATH"
fi

GPUS=4
IS_ASYNC=True

export VLLM_USE_V1=1

# set rollout mode to async if is_async is True, otherwise set to sync
if [ "$IS_ASYNC" = True ]; then
    ROLLOUT_MODE=async
else
    ROLLOUT_MODE=sync
fi

ENFORCE_EAGER=True

# export VLLM_ATTENTION_BACKEND=FLASHINFER

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=8 \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_checkpointing=True \
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
    actor_rollout_ref.actor.fsdp_config.aggressive_empty_cache=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.move_left_padding_right=False \
    actor_rollout_ref.model.minimize_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.mamba_ssm_cache_dtype_fp32=True \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
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
    trainer.experiment_name=debug_qwen3_4b \
    trainer.n_gpus_per_node=${GPUS} \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.log_freq=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    trainer.val_examples_limit=200 \
    trainer.generations_save_freq=1 \
    trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.default_local_dir=/dev/shm/checkpoints \
    trainer.val_before_train=False

