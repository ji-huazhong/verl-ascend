import os
import torch
import socket
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf

import torch.distributed as dist
from vllm import LLM, SamplingParams
import vllm.envs as envs
from vllm.distributed import parallel_state as vllm_ps
from verl.workers.rollout.base import BaseRollout


def get_cluster_info():
    # 确保分布式环境已初始化
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized")

    world_size = dist.get_world_size()

    # 获取当前节点的IP地址
    ip_address = _get_current_node_ip()

    # 收集所有rank的IP地址
    ip_list = [None] * world_size
    dist.all_gather_object(ip_list, ip_address)

    return ip_list


def _get_current_node_ip() -> str:
    try:
        # 创建一个 UDP 套接字（仅用于获取接口信息）
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 连接到一个外部地址（无需真实通信）
            s.connect(("8.8.8.8", 80))  # Google DNS 服务器
            local_ip = s.getsockname()[0]
    except Exception:
        local_ip = _get_ip_by_ifname()
        if not local_ip:
            # 如果失败，回退到遍历接口
            local_ip = "127.0.0.1"
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None):
                ip = addr[4][0]
                if not ip.startswith("::"):
                    local_ip = ip
                    break
    return local_ip

def _init_dp_envs(config):
    rank = torch.distributed.get_rank()
    world_size = int(config.get("rollout_world_size", 1))
    # world_size = int(os.getenv("WORLD_SIZE", "-1"))
    tp_size = int(config.get("tensor_model_parallel_size", 1))
    dp_size = int(config.get("dp_model_parallel_size", 1))

    all_ranks = torch.arange(world_size).reshape(-1, dp_size, 1, tp_size)  # noqa
    group_ranks = all_ranks.transpose(1, 3).reshape(-1, dp_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    ip_list = get_cluster_info()
    for index, group_rank in enumerate(group_ranks):
        if torch.distributed.get_rank() in group_rank:
            os.environ["VLLM_DP_MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT")) + 1 + index)
            os.environ["VLLM_DP_MASTER_IP"] = ip_list[group_rank[0]]
    local_dp_rank = rank // tp_size % dp_size
    os.environ["VLLM_DP_RANK"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_PORT"] = os.environ["VLLM_DP_MASTER_PORT"]
    envs.VLLM_DP_RANK = int(os.environ["VLLM_DP_RANK"])
    envs.VLLM_DP_MASTER_IP = os.environ["VLLM_DP_MASTER_IP"]
    envs.VLLM_DP_MASTER_PORT = int(os.environ["VLLM_DP_MASTER_PORT"])

    print(f"[VLLM] using TP={tp_size}, DP={dp_size}", flush=True)


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

    tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
    assert tensor_parallel_size <= torch.distributed.get_world_size(), (
        "tensor parallel size should be less than or equal to the world size"
    )
    max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

    if kwargs.get("train_tp") is not None:
        # deployed with megatron
        import os

        os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
        os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
        vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

    rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
    if not rope_scaling_config:
        max_position_embeddings = None
        if hasattr(model_hf_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.max_position_embeddings
        elif hasattr(model_hf_config, "llm_config") and hasattr(
            model_hf_config.llm_config, "max_position_embeddings"
        ):
            max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
        elif hasattr(model_hf_config, "text_config") and hasattr(
            model_hf_config.text_config, "max_position_embeddings"
        ):
            max_position_embeddings = model_hf_config.text_config.max_position_embeddings
        if max_position_embeddings is None:
            raise ValueError("max_position_embeddings not found in model_hf_config")
        assert max_position_embeddings >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
        )
    else:
        # handle type where there's a length extend factor
        # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
        # for using yarn as an example
        rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

        assert (
            model_hf_config.max_position_embeddings * rope_scaling_factor
            >= config.prompt_length + config.response_length
        ), (
            "model context length should be greater than total sequence length, "
            + f"got rope_scaling_factor={rope_scaling_factor} and "
            + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
        )

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
    engine_kwargs = (
        {}
        if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
        else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
    )
    # For each vLLM engine parameter,
    # - `None` means not setting it, so we pop it, and leave it to vLLM default value
    #    (which can vary across different vLLM versions);
    # - Otherwise it's the desired value we want to explicitly set.
    engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
    if config.get("limit_images", None):  # support for multi-image data
        engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

    # patch this for npu
    enable_infer_ep = False
    if hasattr(config, "dp_model_parallel_size") and config.dp_model_parallel_size > 1:
        _init_dp_envs(config)
        enable_infer_ep = True

    self.inference_engine = LLM(
        model=model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype=config.dtype,
        enforce_eager=config.enforce_eager,
        gpu_memory_utilization=config.gpu_memory_utilization,
        disable_custom_all_reduce=False,
        skip_tokenizer_init=False,
        max_model_len=max_model_len,
        load_format=load_format,
        disable_log_stats=config.disable_log_stats,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        enable_prefix_caching=True,
        trust_remote_code=trust_remote_code,
        enable_expert_parallel=enable_infer_ep,
        seed=config.get("seed", 0),
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

    kwargs["detokenize"] = False

    # supporting adding any sampling params from the config file
    for k in config.keys():
        if hasattr(SamplingParams(), str(k)):
            kwargs[k] = config.get(k)
    kwargs["n"] = 1  # already repeat in ray_trainer
    print(f"kwargs: {kwargs}")
    self.sampling_params = SamplingParams(**kwargs)

    self.pad_token_id = tokenizer.pad_token_id


def patch_vllm_rollout_spmd():
    from verl.workers.rollout.vllm_rollout import vLLMRollout
    from mindspeed_rl.boost.patch_utils import apply_patches

    patch_list = [
        ("__init__", __init__),
    ]

    apply_patches(patch_list, vLLMRollout)