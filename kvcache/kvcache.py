from config.model_config import ModelConfig


def get_mha_kvcache_size(config: ModelConfig, use_fp8):
    kvcache_size = (
        2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim
    )
    if not use_fp8:
        kvcache_size *= 2
    return kvcache_size


def get_mla_kvcache_size(config: ModelConfig, use_fp8):
    kvcache_size = config.num_hidden_layers * (
        config.kv_lora_rank + config.qk_rope_head_dim
    )
    if not use_fp8:
        kvcache_size *= 2
    return kvcache_size


def get_kvcache_size(config: ModelConfig, use_fp8):
    if config.attn_type == "MHA/GQA":
        return get_mha_kvcache_size(config, use_fp8)
    elif config.attn_type == "MLA":
        return get_mla_kvcache_size(config, use_fp8)
