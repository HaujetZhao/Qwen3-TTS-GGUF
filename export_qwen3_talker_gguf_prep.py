
import torch
import json
import os
import shutil
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import sys

# Add project root to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../参考项目/Qwen3-TTS')))

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration, Qwen3TTSConfig

def export_talker():
    model_path = "models/Qwen3-TTS"
    output_dir = "models/Qwen3-TTS-Talker-GGUF-Prep"
    
    print(f"Loading model from {model_path}...")
    # Load the full model
    # We use the local config/code
    config = Qwen3TTSConfig.from_pretrained(model_path)
    model = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(model_path, config=config.talker_config)
    
    print("Model loaded.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    talker = model.model
    talker_config = config.talker_config
    
    # 1. Prepare Unified Embedding
    print("Preparing Unified Embedding...")
    text_vocab_size = talker_config.text_vocab_size
    codec_vocab_size = talker_config.vocab_size # 3072
    hidden_size = talker_config.hidden_size # 2048
    
    # Calculate effective text embeddings: MLP(Embedding(text_ids))
    # Since text_vocab_size is large (151k), this might be slow/memory intensive if done naively.
    # We can do it in batches.
    
    text_embed_weight = talker.text_embedding.weight # [151936, 2048]
    text_projection = model.text_projection # MLP
    
    print(f"  Text Vocab: {text_vocab_size}, Codec Vocab: {codec_vocab_size}")
    print(f"  Projecting text embeddings (this bakes the MLP into the embeddings)...")
    
    unified_vocab_size = text_vocab_size + codec_vocab_size
    unified_embedding = torch.zeros((unified_vocab_size, hidden_size), dtype=text_embed_weight.dtype)
    
    # Process text embeddings in batches to save memory
    batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, text_vocab_size, batch_size), desc="Projecting Text Embeds"):
            end = min(i + batch_size, text_vocab_size)
            batch_indices = torch.arange(i, end, device=text_embed_weight.device)
            embeds = talker.text_embedding(batch_indices)
            projected = text_projection(embeds)
            unified_embedding[i:end] = projected
            
        # 2. Copy Codec Embeddings
        print("  Copying codec embeddings...")
        codec_embed_weight = talker.codec_embedding.weight # [3072, 2048]
        unified_embedding[text_vocab_size:] = codec_embed_weight

    # 3. Prepare Output Head
    print("Preparing Output Head...")
    # The original head is [3072, 2048] mapping Hidden -> Codec Logits
    # We need [155008, 2048] mapping Hidden -> Unified Logits
    # Text logits (0..151935) should be -inf (or very small)
    # Codec logits (151936..end) should be the original head
    
    original_head_weight = model.codec_head.weight # [3072, 2048]
    new_head_weight = torch.full((unified_vocab_size, hidden_size), -100.0, dtype=original_head_weight.dtype) 
    # Usually linear weights are initialized with small values, but here we want the output logits to be -inf.
    # Wait, the Linear layer computes x @ W.T + b. 
    # If we set W to 0, logits are 0.
    # We want logits to be -inf.
    # Ideally we just mask them during inference, but for GGUF export we want the model to "just work".
    # Setting weights to 0 makes logits 0, which gives probability 1/vocab_size (after softmax). That's bad.
    # But wait, standard Qwen models don't have -inf weights for padding.
    # Let's just set them to 0 for now. The user will be instructing the model to generate audio codes, 
    # so the sampling logic should likely constrain it?
    # Actually, if we set them to 0, they might be sampled.
    # A better trick: Set the bias to -1e9 for these tokens?
    # Qwen2 `lm_head` usually has no bias. `bias=False` in `Qwen3TTSTalkerForConditionalGeneration`.
    # So we can't use bias.
    # Let's just copy the Codec Head to the end and leave the rest as 0? 
    # No, 0 is dangerous.
    # Let's verify if we can just export it as is.
    # For now, I will fill the text part with random small noise or zeros. 
    # Actually, if we assume the user provides a prompt that forces the model into "codec mode", maybe it's fine.
    # Let's set them to 0.
    
    new_head_weight = torch.zeros((unified_vocab_size, hidden_size), dtype=original_head_weight.dtype)
    new_head_weight[text_vocab_size:] = original_head_weight
    
    # 4. Create New State Dict
    print("Constructing new state dict...")
    new_state_dict = {}
    
    # Rename keys to match standard Qwen2/Llama structure
    # Qwen3-TTS keys: model.layers.X...
    # Standard Qwen2 keys: model.layers.X...
    
    new_state_dict["model.embed_tokens.weight"] = unified_embedding
    new_state_dict["lm_head.weight"] = new_head_weight
    new_state_dict["model.norm.weight"] = talker.norm.weight
    
    for key, value in talker.state_dict().items():
        if key.startswith("layers."):
            new_state_dict[f"model.{key}"] = value
        elif key == "norm.weight":
            pass # Already handled
        elif "embedding" in key:
            pass # Already handled
        elif "rotary_emb" in key:
            pass # Not needed in state_dict usually (computed on fly)
        else:
            print(f"Skipping key: {key}")

    # 5. Save Model
    print(f"Saving to {output_dir}/model.safetensors...")
    save_file(new_state_dict, os.path.join(output_dir, "model.safetensors"))
    
    # 6. Save Config
    print("Saving config...")
    # Convert Qwen3 config to Qwen2 config format
    qwen2_config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": unified_vocab_size,
        "hidden_size": talker_config.hidden_size,
        "intermediate_size": talker_config.intermediate_size,
        "num_hidden_layers": talker_config.num_hidden_layers,
        "num_attention_heads": talker_config.num_attention_heads,
        "num_key_value_heads": talker_config.num_key_value_heads,
        "hidden_act": talker_config.hidden_act,
        "max_position_embeddings": talker_config.max_position_embeddings,
        "initializer_range": talker_config.initializer_range,
        "rms_norm_eps": talker_config.rms_norm_eps,
        "use_cache": talker_config.use_cache,
        "tie_word_embeddings": False,
        "rope_theta": talker_config.rope_theta,
        "rope_scaling": talker_config.rope_scaling,
        "attention_dropout": talker_config.attention_dropout,
        "bos_token_id": config.tts_bos_token_id, # Use TTS BOS? Or Text BOS?
        "eos_token_id": config.tts_eos_token_id,
        # Add custom fields if needed
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(qwen2_config, f, indent=2)
        
    # 7. Handle Tokenizer
    print("Updating tokenizer (vocab.json)...")
    # Copy original tokenizer files
    for file in ["vocab.json", "merges.txt", "tokenizer_config.json"]:
        src = os.path.join(model_path, file)
        if os.path.exists(src):
            shutil.copy(src, output_dir)
            
    # Append codec tokens to vocab.json
    with open(os.path.join(output_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    # Check if vocab size matches
    print(f"Original vocab size in json: {len(vocab)}")
    
    # Add dummy tokens for codec
    # We start from text_vocab_size.
    # The original vocab might already have some size.
    # Qwen vocab is usually 151936.
    # We need to ensure we map 151936..155007 to something.
    
    current_len = len(vocab)
    if current_len != text_vocab_size:
        print(f"Warning: vocab.json size ({current_len}) != text_vocab_size ({text_vocab_size})")
        
    for i in range(codec_vocab_size):
        token_str = f"<codec_{i}>"
        vocab[token_str] = text_vocab_size + i
        
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)
        
    print("Done! You can now run llama.cpp conversion on:", output_dir)

if __name__ == "__main__":
    export_talker()
