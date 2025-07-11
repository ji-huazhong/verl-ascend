from verl_ascend.workers.rollout.vllm_rollout.vllm_rollout_spmd import patch_vllm_rollout_spmd
from verl_ascend.models.mcore.model_forward import patch_mcore_model_forwrd


def adpat_verl_to_ascend():
    from mindspeed import megatron_adaptor  # noqa: F401

    patch_mcore_model_forwrd()
    patch_vllm_rollout_spmd()
