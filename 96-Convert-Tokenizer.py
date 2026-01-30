import os
from transformers import AutoTokenizer

def convert_tokenizer():
    model_dir = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    output_file = os.path.join(model_dir, "tokenizer.json")
    
    print(f"⌛ 正在从 {model_dir} 加载原始分词器 (使用 transformers)...")
    # 使用与引擎一致的参数
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, 
        trust_remote_code=True, 
        fix_mistral_regex=True
    )
    
    print(f"💾 正在导出为轻量级格式: {output_file}")
    # 这里我们直接保存，AutoTokenizer 如果是快分词器会生成 tokenizer.json
    tokenizer.backend_tokenizer.save(output_file)
    
    if os.path.exists(output_file):
        print(f"✅ 转换成功！现在可以使用 tokenizers 库直接加载该文件。")
    else:
        print(f"❌ 转换失败，未生成文件。")

if __name__ == "__main__":
    convert_tokenizer()
