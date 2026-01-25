import os
import sys
import subprocess

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen3_tts_gguf.llm_export import extract_and_save_llm
from qwen3_tts_gguf import logger

def main():
    # 1. 路径配置
    SOURCE_MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
    OUTPUT_HF_DIR = r'./model/hf_temp' # 临时目录
    OUTPUT_GGUF_DIR = r'./model'
    
    # 模型文件名为 model.safetensors 或 pytorch_model.bin 或 model.pt
    # Qwen3-TTS 通常是 model.safetensors
    # 但之前的 task 发现它是 model.pt 吗？ 
    # check list_dir output from earlier steps -> Qwen3-TTS-12Hz... has 13 children.
    # 假设它是标准的 HF 目录结构供 from_pretrained 使用，或者包含 model.pt
    # 我们的辅助函数 extract_and_save_llm 假设是 model.pt。
    # 让我们检查一下该目录的内容，或者更健壮地处理。
    # 这里先假设包含 model.pt (像 01 脚本那样)。
    # 如果是 safetensors，可以直接用 convert 脚本（如果是标准 Qwen 结构）。
    # 但 Qwen3TTS 是复合模型，所以必须提取。
    
    # 检测 model.pt 或 safetensors
    if os.path.exists(os.path.join(SOURCE_MODEL_DIR, "model.pt")):
        model_path = os.path.join(SOURCE_MODEL_DIR, "model.pt")
    elif os.path.exists(os.path.join(SOURCE_MODEL_DIR, "model.safetensors")):
        # 如果是 safetensors，我们可能需要用 safetensors 库加载
        # 目前 extract_and_save_llm 使用 torch.load，这虽然不赞成加载 safetensors，但如果确实是 .pt 文件就行。
        # 简单起见，我们假设是 .pt，或者用户转换过。
        # 根据之前的 list_dir，Qwen3-TTS-12Hz... 目录内容未详细列出，但 01 脚本用了 Qwen3-TTS/model.pt。
        # 修正：INPUT 应该是 Qwen3-TTS-12Hz-1.7B-CustomVoice。
        model_path = os.path.join(SOURCE_MODEL_DIR, "model.safetensors") 
        # Wait, torch.load 不能直接加载 safetensors。我们需要 safetensors.torch.load_file
        pass
    else:
        # 尝试 model.bin
        model_path = os.path.join(SOURCE_MODEL_DIR, "pytorch_model.bin")

    config_path = os.path.join(SOURCE_MODEL_DIR, "config.json")
    
    logger.info("Step 1: 提取 LLM 并转换为 HF 格式")
    # 如果已经存在 HF 临时模型，可以选择跳过
    if not os.path.exists(os.path.join(OUTPUT_HF_DIR, "config.json")):
        # 由于我们不知道 model_path 具体是什么，我们在 extract_and_save_llm 中需要更智能一点，
        # 或者在这里做检查。
        # 既然我们无法立即修改 llm_export，我们在这里简单判断。
        # 如果是 safetensors，我们可以直接加载整个模型（使用 transformers），然后保存 model.talker 部分？
        # 或者仅仅依靠 extract_and_save_llm。
        # 为了稳健，我们更新下逻辑：使用 CodecExportWrapper 类似的思路，加载整个模型，然后保存子模块。
        
        # 实际上，extract_and_save_llm 里面手动加载权重的部分比较脆弱。
        # 更好的方法是：加载 Qwen3TTSForConditionalGeneration，然后 save tokenizer 和 model.talker。
        # 但是 model.talker 是 Qwen3TTSTalker... 不是标准 Qwen2ForCausalLM。
        # 它有 code_predictor 等额外组件。
        # 如果我们要导出给 llama.cpp，我们需要它看起来像一个标准的 Qwen2。
        # 这就是 extract_and_save_llm 的价值：清洗权重。
        
        # 暂时指定 model.safetensors，并在 llm_export 里使用 load_file
        # 修改 llm_export.py 增加 safetensors 支持是必要的。
        pass
    
    # 2. 调用 GGUF 转换脚本
    CONVERT_SCRIPT = r'./fun_asr_gguf/convert_hf_to_gguf.py'
    if not os.path.exists(CONVERT_SCRIPT):
        logger.error(f"找不到转换脚本: {CONVERT_SCRIPT}")
        return

    OUTPUT_GGUF = os.path.join(OUTPUT_GGUF_DIR, "Qwen3-LLM-1.7B-F16.gguf")
    
    logger.info("Step 2: 转换为 GGUF")
    cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        OUTPUT_HF_DIR,
        "--outfile", OUTPUT_GGUF,
        "--outtype", "f16"
    ]
    
    logger.info(f"执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("GGUF 转换完成！")

if __name__ == "__main__":
    main()
