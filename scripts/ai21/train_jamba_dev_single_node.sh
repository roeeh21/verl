# ray start --disable-usage-stats --head --port=6379 --num-gpus=8
# rm -rf /dev/shm/jamba_dev && mkdir -p /dev/shm/jamba_dev && gcloud storage cp gs://ai21-mammoth-storage/models/jamba_hf/tiny_dev/* /dev/shm/jamba_dev


python3 -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                data.train_files=${HOME}/data/gsm8k/train.parquet \
                data.val_files=${HOME}/data/gsm8k/test.parquet \
                data.train_batch_size=32 \
                data.max_prompt_length=512 \
                data.max_response_length=1024 \
                data.shuffle=True \
                actor_rollout_ref.actor.optim.lr=0.00001 \
                actor_rollout_ref.actor.router_zloss_alpha=0.000006 \
                actor_rollout_ref.actor.router_lbloss_coef=0.15 \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.actor.ppo_mini_batch_size=32 \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_coef=0.001 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.actor.grad_clip=1.0 \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.actor.fsdp_config.param_offload=False \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
                actor_rollout_ref.rollout.n=8 \
                actor_rollout_ref.rollout.enable_prefix_caching=False \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
                actor_rollout_ref.model.path=/dev/shm/jamba_dev \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                actor_rollout_ref.rollout.enforce_eager=False \
                actor_rollout_ref.rollout.free_cache_engine=False \
                trainer.critic_warmup=0 \
                trainer.logger=['console'] \
                trainer.n_gpus_per_node=8 \
                trainer.save_freq=200000 \
                trainer.test_freq=20 \
                trainer.log_freq=1 \
                trainer.total_epochs=15 \
                trainer.nnodes=1 \
                trainer.remote_save_path=gs://ai21-mammoth-storage/temp/verl_debug_save \
                trainer.default_local_dir=/dev/shm/checkpoints \
                2>&1|tee log.txt