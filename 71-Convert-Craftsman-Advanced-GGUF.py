import sys
import os
import json
from pathlib import Path

# 1. 确保能导入 llama.cpp 的转换脚本
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT_LIB_DIR = os.path.join(PROJECT_ROOT, "ref", "llama.cpp")

if CONVERT_LIB_DIR not in sys.path:
    sys.path.insert(0, CONVERT_LIB_DIR)

# 2. 导入目标模块
try:
    import convert_hf_to_gguf
    from convert_hf_to_gguf import TextModel, ModelBase
except ImportError as e:
    print(f"❌ Error importing convert_hf_to_gguf: {e}")
    sys.exit(1)

# 3. 定义补丁函数
def patched_get_vocab_base_pre(self, tokenizer) -> str:
    print(f"💉 [补丁] 拦截到 get_vocab_base_pre 调用。强制返回 'qwen2'。")
    return "qwen2"

def patched_load_hparams(dir_model: Path, is_mistral_format: bool):
    print(f"💉 [补丁] 强制从 {dir_model / 'config.json'} 加载配置。")
    with open(dir_model / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

# 4. 转换主逻辑
def main():
    CRAFTSMAN_DIR = os.path.join(PROJECT_ROOT, "model", "craftsman_advanced_hf")
    GGUF_OUT = os.path.join(PROJECT_ROOT, "model", "qwen3_tts_craftsman_advanced.gguf")

    print(f"--- 正在将高级工匠组件 (30720词表) 转换为 GGUF ---")
    print(f"源目录: {CRAFTSMAN_DIR}")
    print(f"输出文件: {GGUF_OUT}")

    # 应用补丁
    TextModel.get_vocab_base_pre = patched_get_vocab_base_pre
    ModelBase.load_hparams = staticmethod(patched_load_hparams)
    
    # 动态映射 Tensor 名
    original_index_tensors = ModelBase.index_tensors
    def patched_index_tensors(self, *args, **kwargs):
        tensors = original_index_tensors(self, *args, **kwargs)
        new_tensors = {}
        for name, data_gen in tensors.items():
            if name.startswith("lm_head") or name.startswith("output.") or name.startswith("token_embd."):
                new_tensors[name] = data_gen
            elif name.startswith("model."):
                new_tensors[name] = data_gen 
            else:
                new_tensors[f"model.{name}"] = data_gen
        return new_tensors
    ModelBase.index_tensors = patched_index_tensors

    # 模拟命令行参数
    sys.argv = [
        "convert_hf_to_gguf.py",
        CRAFTSMAN_DIR,
        "--outfile", GGUF_OUT,
        "--outtype", "f16"
    ]
    
    try:
        convert_hf_to_gguf.main()
        print(f"\n✅ 高级工匠 GGUF 转换成功!")
        print(f"输出路径: {GGUF_OUT}")
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
