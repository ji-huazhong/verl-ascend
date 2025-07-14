import sys
import types


def apply_patches(monkey_patches, root_module):
    def _getattr(module_list, root_module):
        if len(module_list) <= 1:
            return root_module

        if hasattr(root_module, module_list[0]):
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))
        else:
            empty_module_name = f"{root_module.__name__}.{module_list[0]}"
            sys.modules[empty_module_name] = types.ModuleType(empty_module_name)
            setattr(root_module, module_list[0], sys.modules.get(empty_module_name))
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))

    for patch_pair in monkey_patches:
        dest, patch = patch_pair
        dest_module = _getattr(dest.split("."), root_module)
        last_module_level = dest.split(".")[-1]
        if not isinstance(patch, types.ModuleType):
            setattr(dest_module, last_module_level, patch)
            continue

        if not hasattr(dest_module, last_module_level) or not hasattr(patch, "__all__"):
            setattr(dest_module, last_module_level, patch)
            sys.modules[f"{dest_module.__name__}.{last_module_level}"] = patch
            continue

        if not hasattr(patch, "__all__"):
            raise NotImplementedError("Patch module must have __all__ definition.")
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))
