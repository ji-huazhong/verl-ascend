import psutil
from codetiming import Timer
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.profiler import (
    DistProfiler,
    GPUMemoryLogger,
    log_gpu_memory_usage,
)
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.megatron_workers import ActorRolloutRefWorker, logger

from ..patch_util import NPUPatchHelper


class ActorRolloutRefWorkerPatch(NPUPatchHelper[ActorRolloutRefWorker]):

    _orig_build_rollout = ActorRolloutRefWorker._build_rollout

    def _build_rollout(self, trust_remote_code=False):
        rollout, sharding_manager = self._orig_build_rollout(trust_remote_code)

        if self.config.rollout.name == "vllm":
            del sharding_manager
            from torch.distributed.device_mesh import init_device_mesh

            from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager

            layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
            )
            # perform weight resharding between actor and rollout
            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            sharding_manager = MegatronVLLMShardingManager(
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                rollout_config=self.config.rollout,
                layer_name_mapping=layer_name_mapping,
                actor_module=self.actor.actor_module,
                weight_converter=weight_converter,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                rollout=rollout,
                bridge=self.bridge,
            )
            log_gpu_memory_usage("After rebuilding sharding manager", logger=logger)
        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        data.batch = data.batch.to(get_device_name())

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module, offload_param=False)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        aggressive_empty_cache(force_sync=True)
        return output
