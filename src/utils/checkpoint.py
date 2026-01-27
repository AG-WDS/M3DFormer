import torch


def _strip_prefix(key, prefixes):
    if prefixes:
        for p in prefixes:
            if key.startswith(p):
                return key[len(p):]
    return key


def load_partial_checkpoint(model : torch.nn.Module,
                            ckpt_path,
                            ckpt_state_dict_key='state_dict',
                            ckpt_prefixes_to_strip:tuple="model.",
                            skip_prefixes_in_model:list=None):
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and ckpt_state_dict_key in ckpt:
        old_sd = ckpt[ckpt_state_dict_key]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        old_sd = ckpt
    else:
        raise ValueError(f"ckpt must be a dict with key '{ckpt_state_dict_key}'")

    stripped_old = {}
    for k, v in old_sd.items():
        new_k = _strip_prefix(k, ckpt_prefixes_to_strip)
        stripped_old[new_k] = v
        print(new_k)

    model_sd = model.state_dict()
    
    skip_prefixes_in_model = skip_prefixes_in_model or []
    def should_skip_model_key(k):
        return any(k.startswith(pref) for pref in skip_prefixes_in_model)

    to_load = {}
    skipped_shape_mismatch = []
    skipped_missing_in_model = []
    skipped_by_prefix = []
    for k_old, v_old in stripped_old.items():
        if should_skip_model_key(k_old):
            skipped_by_prefix.append(k_old)
            continue
        if k_old not in model_sd:
            skipped_missing_in_model.append(k_old)
            continue
        if v_old.shape != model_sd[k_old].shape:
            skipped_shape_mismatch.append((k_old, v_old.shape, model_sd[k_old].shape))
            continue
        to_load[k_old] = v_old

    new_model_sd = model_sd.copy()
    new_model_sd.update(to_load)
    model.load_state_dict(new_model_sd)


    # print(f"== ckpt: {ckpt_path} ==")
    # print(f"Total ckpt params: {len(stripped_old)}")
    # print(f"Loaded params: {len(to_load)}, example: {to_load.keys()}")
    # print(f"Skipped (by prefix): {len(skipped_by_prefix)} -> example: {skipped_by_prefix[:]}")
    # print(f"Skipped (missing in model): {len(skipped_missing_in_model)} -> example: {skipped_missing_in_model[:]}")
    # print(f"Skipped (shape mismatch): {len(skipped_shape_mismatch)} -> example (key, ckpt_shape, model_shape): {skipped_shape_mismatch[:]}")


    return {
        'loaded': to_load,
        'skipped_by_prefix': skipped_by_prefix,
        'skipped_missing': skipped_missing_in_model,
        'skipped_shape_mismatch': skipped_shape_mismatch,
        # 'not_loaded_model_keys': not_loaded_model_keys
    }