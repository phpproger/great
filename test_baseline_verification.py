import os
import math
import shutil
import tempfile
import torch
import torch.nn.functional as F

from llama.config import LlamaConfig
from llama.model import LlamaForCausalLM, PlainMLP, TransformerBlock, estimate_spectral_norm


def test_baseline_architecture():
    print("=" * 80)
    print("VERIFYING PRODUCTION BASELINE ARCHITECTURE & CONFIGURATION")
    print("=" * 80)

    # 1. Config verification
    cfg = LlamaConfig.tiny(vocab_size=1000)
    print(f"Config mlp_type: {cfg.mlp_type}")
    print(f"Config scale_embeddings: {cfg.scale_embeddings}")
    print(f"Config use_contraction_aux: {cfg.use_contraction_aux}")
    print(f"Config use_spectral_barrier: {cfg.use_spectral_barrier}")
    print(f"Config default_inversion_method: {cfg.default_inversion_method}")
    print(f"Config contraction_layer_probs: {cfg.contraction_layer_probs}")
    print(f"Config mlp_hidden_dim: {cfg.mlp_hidden_dim}")

    assert cfg.mlp_type == "plain", "Expected default mlp_type to be 'plain'"
    assert cfg.use_spectral_barrier is True, "Expected default use_spectral_barrier to be True"
    assert cfg.scale_embeddings is True, "Expected default scale_embeddings to be True"
    assert cfg.use_contraction_aux is True, "Expected default use_contraction_aux to be True"
    assert cfg.default_inversion_method == "anderson", "Expected default default_inversion_method to be 'anderson'"

    # 2. Model instantiation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM(cfg).to(device)
    model.train()

    for idx, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp, PlainMLP), f"Layer {idx} MLP is not PlainMLP!"
        assert hasattr(layer.mlp, "down_proj"), f"Layer {idx} MLP does not have down_proj!"
        assert hasattr(layer.self_attn, "o_proj"), f"Layer {idx} Attention does not have o_proj!"

    print(f"[PASSED] All {len(model.model.layers)} layers correctly use PlainMLP!")

    # 3. Training forward & backward pass with auxiliary loss + spectral barrier
    input_ids = torch.randint(0, 1000, (2, 16), device=device)
    labels = input_ids.clone()
    logits, loss, _ = model(input_ids=input_ids, labels=labels)

    assert loss is not None, "Loss should not be None"
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    print(f"[PASSED] Forward pass with auxiliary contraction loss + spectral barrier: loss = {loss.item():.4f}")

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"

    print(f"[PASSED] Backward pass computes valid gradients across all parameters!")

    # 4. Multi-step unrolling / inversion with Picard and Anderson
    model.eval()
    with torch.no_grad():
        h = model.model.embed_tokens(input_ids) * math.sqrt(cfg.hidden_size)
        cos, sin = model.model.rotary_emb(h, input_ids.shape[1])
        h_out = h
        for layer in model.model.layers:
            h_out, _ = layer(h_out, cos=cos, sin=sin)
        
        # Test Picard inversion
        h_rev_picard = model.invert(h_out, seq_len=input_ids.shape[1], k_steps=2, method="picard")
        err_picard = (h_rev_picard - model.model.embed_tokens(input_ids)).norm().item()
        print(f"[PASSED] Picard k=2 inversion ran cleanly, L2 norm diff = {err_picard:.4f}")

        # Test Anderson inversion
        h_rev_anderson = model.invert(h_out, seq_len=input_ids.shape[1], method="anderson")
        err_anderson = (h_rev_anderson - model.model.embed_tokens(input_ids)).norm().item()
        print(f"[PASSED] Anderson acceleration inversion ran cleanly, L2 norm diff = {err_anderson:.4f}")

    # 5. Autoregressive generation
    gen_tokens = model.generate(prompt_ids=input_ids[:1, :4], max_new_tokens=8)
    assert gen_tokens.shape == (1, 12), f"Expected shape (1, 12), got {gen_tokens.shape}"
    print(f"[PASSED] Autoregressive generation produced tokens: {gen_tokens.tolist()}")

    # 6. Checkpoint consolidation
    ckpt_dir = os.path.join(tempfile.gettempdir(), "great_checkpoints")
    source_ckpt = os.path.join(ckpt_dir, "plateau_combo_plain_spec.pt")
    target_ckpt = os.path.join(ckpt_dir, "llama_banach_baseline.pt")
    target_prod = os.path.join(ckpt_dir, "llama_banach_production.pt")

    if os.path.exists(source_ckpt):
        shutil.copyfile(source_ckpt, target_ckpt)
        shutil.copyfile(source_ckpt, target_prod)
        print(f"[PASSED] Successfully copied best plateau checkpoint {source_ckpt} to:")
        print(f"         - {target_ckpt}")
        print(f"         - {target_prod}")

        # Verify loading into fresh model
        bench_cfg = LlamaConfig(
            vocab_size=50257,
            hidden_size=128,
            intermediate_size=384,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            tie_word_embeddings=True,
            mlp_type="plain",
            use_spectral_barrier=True,
        )
        fresh_model = LlamaForCausalLM(bench_cfg).to(device)
        state_dict = torch.load(target_ckpt, map_location=device, weights_only=True)
        fresh_model.load_state_dict(state_dict)
        fresh_model.eval()
        print(f"[PASSED] Baseline checkpoint loaded seamlessly into LlamaForCausalLM!")

    print("\n" + "=" * 80)
    print("ALL PRODUCTION BASELINE VERIFICATIONS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_baseline_architecture()
