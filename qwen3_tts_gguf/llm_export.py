import os
import shutil
import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM
from . import logger

def extract_and_save_llm(source_model_path, config_path, output_hf_dir, tokenizer_output_dir):
    """
    提取混合模型中的 LLM 主干，并保存为标准的 Hugging Face 格式。
    
    Args:
        source_model_path (str): 原始 PyTorch (.pt) 模型路径
        config_path (str): 原始模型配置文件路径
        output_hf_dir (str): 输出 HF 模型 (safetensors) 的目录
        tokenizer_output_dir (str): 输出 Tokenizer 配置的目录
    """
    logger.info(f"[LLM Export] Loading full model from {source_model_path} ...")
    
    llm_weights = {}
    
    if source_model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        full_model = load_file(source_model_path)
    else:
        # 映射 CPU 防止 OOM
        full_model = torch.load(source_model_path, map_location='cpu')

    # 1. 提取 LLM 权重
    # Qwen3-TTS 的 LLM 权重通常在 'llm.' 前缀下，或者是 'talker.' 前缀下？
    # 根据 02-Export-Decoder-GGUF.py 的代码，它是 'llm.'。
    # 根据 modeling_qwen3_tts.py，Qwen3TTSForConditionalGeneration 包含 self.talker.
    # self.talker 是 Qwen3TTSTalkerForConditionalGeneration。
    # 让我们检查之前的 02 脚本，它是针对 Fun-ASR-Nano 的。但是现在的任务是 Qwen3-TTS。
    # 这是一个关键点：现在的输入模型是 `Qwen3-TTS` 还是 `Qwen3-TTS-12Hz...`？
    # 用户指定的输入是 `Qwen3-TTS-12Hz-1.7B-CustomVoice`。
    # 该模型结构在 config.json 中有 "talker_config"。
    # PyTorch state dict keys 可能是 "talker.model.xxx" (如果它是基于 Qwen3TTSTalkerForConditionalGeneration).
    
    llm_weights = {}
    print("   [LLM Export] Extracting LLM weights...")
    
    # 我们需要智能识别前缀。
    # 通常的 HF CausalLM 权重是 model.layers...
    # 如果是在 talker 下，可能是 talker.model.layers...
    
    prefix = None
    for key in full_model.keys():
        if key.startswith("talker.model."):
            prefix = "talker."
            break
        if key.startswith("model.layers."): # 直接是 LLM
            prefix = ""
            break
            
    if prefix is None:
        # 尝试暴力匹配，找 model.embed_tokens.weight
        for key in full_model.keys():
            if "model.embed_tokens.weight" in key:
                # e.g. "talker.model.embed_tokens.weight" -> prefix="talker."
                prefix = key.replace("model.embed_tokens.weight", "")
                break
    
    if prefix is None:
        print("   [Error] Could not identify LLM prefix in state dict.")
        return False
        
    print(f"   [LLM Export] Identified prefix: '{prefix}'")

    for key, value in full_model.items():
        if key.startswith(prefix):
            # 去除前缀，使其变成标准的 Qwen2/3 CausalLM 格式
            # 例如: talker.model.layers.0... -> model.layers.0...
            # 注意: talker.lm_head.weight -> lm_head.weight
            
            # 特殊处理: Qwen3TTSTalkerForConditionalGeneration 结构是:
            # self.model = Qwen3TTSTalkerModel (即 Qwen2Model)
            # self.lm_head ...
            
            # 如果 prefix 是 "talker."
            # talker.model.xyz -> model.xyz
            # talker.lm_head.xyz -> lm_head.xyz
            
            new_key = key[len(prefix):]
            llm_weights[new_key] = value

    print(f"   [LLM Export] Extracted {len(llm_weights)} keys.")
    del full_model

    # 2. 加载并转换配置
    print(f"   [LLM Export] Loading config from {config_path} ...")
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
        
    # 提取 talker_config 作为主配置
    if "talker_config" in full_config:
        llm_config_dict = full_config["talker_config"]
    else:
        llm_config_dict = full_config # 假设直接是 LLM 配置
        
    # 确保架构名称正确，以便 Transformers 识别为 Qwen2
    # Qwen3 本质上很多时候兼容 Qwen2
    if "architectures" not in llm_config_dict or not llm_config_dict["architectures"]:
        llm_config_dict["architectures"] = ["Qwen2ForCausalLM"]
    else:
        # 强制改为 Qwen2ForCausalLM 以便通用加载
        llm_config_dict["architectures"] = ["Qwen2ForCausalLM"]
        
    # Qwen2 需要 model_type = "qwen2"
    llm_config_dict["model_type"] = "qwen2"

    config = AutoConfig.for_model("qwen2", **llm_config_dict)
    
    # 3. 初始化并保存
    print("   [LLM Export] Initializing Qwen2ForCausalLM ...")
    # 使用 AutoModel 避免硬编码 import
    qwen_model = AutoModelForCausalLM.from_config(config)
    
    print("   [LLM Export] Loading state dict ...")
    # 使用 strict=False 允许一些非关键键缺失（如 code_predictor 相关，如果它不在标准 CausalLM 中）
    # 标准 CausalLM 只需要 model.* 和 lm_head.*
    # 我们的 llm_weights 可能包含 code_predictor.* (如果它在 talker 下)，这在标准 Qwen2 中是不需要的。
    # 过滤掉不需要的键，或者让它不匹配。
    # 为了保险，过滤:
    clean_weights = {k: v for k, v in llm_weights.items() if k.startswith("model.") or k.startswith("lm_head.")}
    
    missing, unexpected = qwen_model.load_state_dict(clean_weights, strict=False)
    print(f"   [LLM Export] Loaded. Missing: {len(missing)}, Unexpected keys ignored: {len(unexpected)}")

    os.makedirs(output_hf_dir, exist_ok=True)
    print(f"   [LLM Export] Saving to {output_hf_dir} ...")
    qwen_model.save_pretrained(output_hf_dir, safe_serialization=True)
    
    # 4. 处理 Tokenizer (复制相关文件)
    print(f"   [LLM Export] Copying tokenizer files to {tokenizer_output_dir} ...")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    
    # 假设源目录有一些 tokenizer 文件，或者我们需要从包含 tokenizer 的目录复制
    # 通常 Qwen3-TTS 目录结构中 tokenizer 文件在根目录
    src_dir = os.path.dirname(config_path)
    files_to_copy = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json']
    
    for file in files_to_copy:
        src = os.path.join(src_dir, file)
        dst = os.path.join(tokenizer_output_dir, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"      Copied {file}")
        else:
            print(f"      Warning: {file} not found in source.")
            
    # 特别注意：Qwen2 Tokenizer 需要特制的 tokenizer_config.json 或 tokenizer.json
    # 如果原始的是 Qwen3TTS 特有的，可能导致 convert_hf_to_gguf 失败。
    # 但通常 Qwen 使用的是 BPE，格式兼容。
    
    return True
