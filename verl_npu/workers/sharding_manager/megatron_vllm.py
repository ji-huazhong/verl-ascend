from omegaconf import DictConfig
from torch import nn
from vllm import LLM
from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
from verl.utils.device import get_torch_device
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.profiler.performance import simple_timer
from verl.utils.megatron_utils import (
    offload_megatron_model_to_cpu,
    per_tensor_generator,
)
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager, logger

from verl_npu.patch_util import NPUPatchHelper


class MegatronVLLMSshardingManagerPatch(NPUPatchHelper[MegatronVLLMShardingManager]):

    _orig_init_ = MegatronVLLMShardingManager.__init__

    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: LLM,
        model_config: DictConfig,
        transformer_config,
        rollout_config: DictConfig,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
        device_mesh,
        offload_param: bool = True,
        rollout=None,
        bridge=None,
    ):
        self._orig_init_(
            actor_module,
            inference_engine,
            model_config,
            transformer_config,
            rollout_config,
            layer_name_mapping,
            weight_converter,
            device_mesh,
            offload_param,
            bridge,
        )
        self.rollout = rollout
    
    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            aggressive_empty_cache(force_sync=True)

            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            # NOTE(hz): There is no need to reload the param since it will not be unloaded 
            # after updating the actor.
            # This is a small optimization: one offload+onload operation is omitted.
            # if self.offload_param:
            #     load_megatron_model_to_gpu(self.actor_module, load_grad=False)

            if self.rollout_config.free_cache_engine:    
                self.rollout.onload_model_weights()

            if self.bridge is not None:
                per_tensor_param = self.bridge.export_weights(self.actor_module)
            else:
                per_tensor_param = per_tensor_generator(
                    self.actor_module,
                    self.model_config,
                    self.weight_converter,
                    self.transformer_config,
                    self.layer_name_mapping,
                )
            model = self.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            loaded_params = model.load_weights(per_tensor_param)
            info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
            logger.info(info)

            if self.offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            aggressive_empty_cache(force_sync=True)

            self.rollout.init_cache_engine() # TODO: should we move this func call into sharding_manager?

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)
    

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if self.rollout_config.free_cache_engine:
        #     self.inference_engine.sleep(level=1)
            self.rollout.free_cache_engine()
            log_gpu_memory_usage("After free vllm cache", logger=logger)
            self.rollout.offload_model_weights()
            log_gpu_memory_usage("After offload vllm model", logger=logger)
        for model in self.actor_module:
            model.train()

        aggressive_empty_cache(force_sync=True)

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
