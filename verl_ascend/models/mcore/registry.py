from typing import Callable, Dict, Type

from .model_forward import gptmodel_forward
from verl.models.mcore.registry import SupportedModel, gptmodel_forward_qwen2_5_vl


# Registry for model forward functions
MODEL_FORWARD_REGISTRY: Dict[SupportedModel, Callable] = {
    SupportedModel.LLAMA: gptmodel_forward,
    SupportedModel.QWEN2: gptmodel_forward,
    SupportedModel.QWEN2_MOE: gptmodel_forward,
    SupportedModel.MIXTRAL: gptmodel_forward,
    SupportedModel.DEEPSEEK_V3: gptmodel_forward,
    SupportedModel.QWEN2_5_VL: gptmodel_forward,
    SupportedModel.LLAMA4: gptmodel_forward,
    SupportedModel.QWEN3: gptmodel_forward,
    SupportedModel.QWEN3_MOE: gptmodel_forward,
}


def patch_mcore_registry():
    from verl.models import mcore
    from verl_ascend.patch_utils import apply_patches

    patch_list = [
        ("registry.MODEL_FORWARD_REGISTRY", MODEL_FORWARD_REGISTRY),
    ]

    apply_patches(patch_list, mcore)