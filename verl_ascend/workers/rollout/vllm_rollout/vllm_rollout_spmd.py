import os
import torch
from copy import deepcopy

from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from verl.workers.rollout.base import BaseRollout


def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
    """A vLLM rollout. It requires the module is supported by the vllm.

    Args:
        module: module here follows huggingface APIs
        config: DictConfig
        tokenizer: the task/model tokenizer
        model_hf_config: the huggingface config to initiallize the generating model in vllm
        **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
    """
    super(BaseRollout, self).__init__()
    self.config = config
    assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

    tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
    assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
    max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

    if kwargs.get("train_tp") is not None:
        # deployed with megatron
        import os

        os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
        os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            train_tp = kwargs.get("train_tp")
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
        else:
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

    rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
    if not rope_scaling_config:
        max_position_embeddings = None
        if hasattr(model_hf_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.max_position_embeddings
        elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
        elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.text_config.max_position_embeddings
        if max_position_embeddings is None:
            raise ValueError("max_position_embeddings not found in model_hf_config")

        assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

    max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

    if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
        raise ValueError(
            "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                            please increase max_num_batched_tokens or disable chunked prefill"
        )

    trust_remote_code = kwargs.get("trust_remote_code", False)
    load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

    lora_kwargs = kwargs.pop("lora_kwargs", {})
    self.lora_kwargs = lora_kwargs
    # copy it to avoid secretly modifying the engine config
    engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
    # For each vLLM engine parameter,
    # - `None` means not setting it, so we pop it, and leave it to vLLM default value
    #    (which can vary across different vLLM versions);
    # - Otherwise it's the desired value we want to explicitly set.
    engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
    if config.get("limit_images", None):  # support for multi-image data
        engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

    # TODO: add dp envs
    # _init_dp_envs(config)
    self.inference_engine = LLM(
        model=model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype=config.dtype,
        enforce_eager=config.enforce_eager,
        gpu_memory_utilization=config.gpu_memory_utilization,
        disable_custom_all_reduce=False,
        # disable_mm_preprocessor_cache=False,
        skip_tokenizer_init=False,
        max_model_len=max_model_len,
        load_format='safetensors',
        disable_log_stats=config.disable_log_stats,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        enable_prefix_caching=True,
        trust_remote_code=trust_remote_code,
        seed=config.get("seed", 0),
        enable_expert_parallel=True,
        **lora_kwargs,
        **engine_kwargs,
    )

    # Offload vllm model to reduce peak memory usage
    if config.free_cache_engine:
        self.inference_engine.sleep(level=1)

    kwargs = dict(
        n=1,
        logprobs=0,  # can be set to 0 and let actor to recompute
        max_tokens=config.response_length,
    )

    # # we may detokenize the result all together later
    if vllm_version != "0.3.1":
        kwargs["detokenize"] = False

    # supporting adding any sampling params from the config file
    for k in config.keys():
        if hasattr(SamplingParams(), str(k)):
            kwargs[k] = config.get(k)

    self.sampling_params = SamplingParams(**kwargs)

    self.pad_token_id = tokenizer.pad_token_id


def patch_vllm_rollout_spmd():
    from verl.workers.rollout.vllm_rollout import vLLMRollout
    from verl_ascend.patch_utils import apply_patches

    patch_list = [
        ("__init__", __init__),
    ]

    apply_patches(patch_list, vLLMRollout)
