"""
检查提取出来的大师模型的 embedding 表大小
"""
from safetensors import safe_open

MODEL_PATH = 'Standalone-Bare-Master/model.safetensors'

print('=== Embedding 表信息 ===')
with safe_open(MODEL_PATH, framework='pt', device='cpu') as f:
    keys = list(f.keys())

    # 查找 embedding 层
    embed_keys = [k for k in keys if 'embed' in k.lower()]
    print(f'\n找到 {len(embed_keys)} 个 embedding 相关层:')

    total_params = 0
    for key in sorted(embed_keys):
        tensor = f.get_tensor(key)
        params = tensor.numel()
        total_params += params
        print(f'  {key}:')
        print(f'    形状: {tuple(tensor.shape)}')
        print(f'    参数量: {params:,} ({params * 4 / 1024**2:.2f} MB)')

    print(f'\n总 embedding 参数量: {total_params:,} ({total_params * 4 / 1024**2:.2f} MB)')

print('\n=== Tokenizer 信息 (来自 config.json) ===')
print('  vocab_size (codec tokens): 3,072')
print('  text_vocab_size (文本 tokens): 151,936')
print('  hidden_size: 2,048')
print()
print('  最大 codec token ID 范围:')
print('    - codec_pad_id: 2148')
print('    - codec_bos_id: 2149')
print('    - codec_eos_token_id: 2150')
print('    - codec_think_id: 2154')
print('    - codec_nothink_id: 2155')
print('    - codec_think_bos_id: 2156')
print('    - codec_think_eos_id: 2157')
print()
print('  说话人 ID (spk_id) 最大值: ~3066 (serena, vivian)')
print()
print('  语言 ID (codec_language_id) 范围: 2050-2074')
