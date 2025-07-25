#!/usr/bin/env bash
set -xeuo pipefail

project_name='DAPO'
exp_name='DAPO-Qwen3-30B-A3B-MATH-megatron-one-step-off-2-6'
RUNTIME_ENV=recipe/one_step_off_policy/all2allv_runtime_env.yaml

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 12))
# max_prompt_length=$((512 * 1))
# max_response_length=$((512 * 2))


enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=0.1

loss_agg_mode="token-mean"

train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=32


NNODES=${NNODES:-2}

MODEL_PATH=${MODEL_PATH:-"/sfs_turbo/pretrained_models/Qwen3-30B-A3B"}
MCORE_MODEL_PATH="/sfs_turbo/mcore/Qwen3-30B-A3B"
MODEL_PATH="/sfs_turbo/pretrained_models/Qwen3-30B-A3B-No-think"
CKPTS_DIR="./ckpt"
TRAIN_FILE="/sfs_turbo/baymax/dapo-math-17k-shuffled-myc.parquet"
TEST_FILE="/sfs_turbo/baymax/dapo-math-17k-shuffled-myc.parquet"
# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

ref_offload=True
actor_offload=False
gen_tp=2
gen_dp=8
gen_world_size=16
train_tp=2
train_ep=4
train_pp=2
train_cp=4


# python3 -m recipe.one_step_off_policy.async_main_ppo \
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.one_step_off_policy.async_main_ppo \
    --config-path=config \
    --config-name='async_ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.strategy=megatron \
    critic.strategy=megatron \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    +actor_rollout_ref.rollout.rollout_world_size=${gen_world_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    +actor_rollout_ref.rollout.dp_model_parallel_size=${gen_dp} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.n_gpus=16 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.param_offload=${ref_offload} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.device=npu $@ \
    actor_rollout_ref.nccl_timeout=7200 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True $@ 2>&1 | tee /tmp/ray.output

ray_name=$(cat /tmp/ray.output | grep "submitted successfully" | awk '{print $2}')
ray_name=${ray_name//\'}
path_log_dir=./logs_724_longrun/
mkdir -p $path_log_dir
echo "日志存放文件夹: $path_log_dir"
cp $0 $path_log_dir/
nohup ray job logs -f $ray_name 2>&1 | tee $path_log_dir/baymax_724_12k.log &