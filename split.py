MUON = [
    *(f"model.layers.{i}.self_attn.q_proj.weight" for i in range(24)),
    *(f"model.layers.{i}.self_attn.k_proj.weight" for i in range(24)),
    *(f"model.layers.{i}.self_attn.v_proj.weight" for i in range(24)),
    *(f"model.layers.{i}.self_attn.o_proj.weight" for i in range(24)),
    *(f"model.layers.{i}.mlp.gate_proj.weight" for i in range(24)),
    *(f"model.layers.{i}.mlp.up_proj.weight" for i in range(24)),
    *(f"model.layers.{i}.mlp.down_proj.weight" for i in range(24)),
]

ADAM_BASE = [
    "model.embed_tokens.weight",
    "model.norm.weight",

    *(f"model.layers.{i}.self_attn.q_proj.bias" for i in range(24)),
    *(f"model.layers.{i}.self_attn.k_proj.bias" for i in range(24)),
    *(f"model.layers.{i}.self_attn.v_proj.bias" for i in range(24)),

    *(f"model.layers.{i}.input_layernorm.weight" for i in range(24)),
    *(f"model.layers.{i}.post_attention_layernorm.weight" for i in range(24)),
]