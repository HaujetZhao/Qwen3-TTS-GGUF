import os
import sys
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 捕获配置
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
os.makedirs(SAVE_DIR, exist_ok=True)

class CaptureState:
    def __init__(self):
        self.master_step = -1
        self.target_master_step = 0
        self.captured = False

state = CaptureState()

def talker_forward_pre_hook(module, args, kwargs):
    """
    挂载在 talker.forward 之前，捕获进入 Master Backbone 之前的状态。
    """
    # 获取 state
    # input_ids: 正在生成的 token
    # past_hidden: 上一步 Master 的输出
    input_ids = kwargs.get('input_ids')
    past_hidden = kwargs.get('past_hidden')
    
    if state.master_step == -1:
        # 捕获 Prefill 阶段的 Embedding 输入 (用于喂给 GGUF)
        # 尝试获取 inputs_embeds，如果没有则手动计算
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None and input_ids is not None:
             inputs_embeds = module.get_input_embeddings()(input_ids)
        
        if inputs_embeds is not None:
             print(f"[CAPTURE] Prefill Input Embeds Hooked.")
             np.save(os.path.join(SAVE_DIR, "prefill_input_embeds.npy"), inputs_embeds.detach().cpu().to(torch.float32).numpy())

    if state.master_step == state.target_master_step:
        print(f"[CAPTURE] Master Step {state.master_step} Entry Hooked.")
        if input_ids is not None:
            np.save(os.path.join(SAVE_DIR, "master_step_0_input_ids.npy"), input_ids.cpu().numpy())
        if past_hidden is not None:
            np.save(os.path.join(SAVE_DIR, "master_step_0_past_hidden.npy"), past_hidden.detach().cpu().to(torch.float32).numpy())
    return None

def predictor_generate_pre_hook(module, args, kwargs):
    """
    挂载在 predictor.generate 之前，这是真正的工匠输入。
    """
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is None and len(args) > 0:
        inputs_embeds = args[0]
        
    if state.master_step == state.target_master_step:
        print(f"[CAPTURE] Craftsman Gen Step 0 Input Hooked.")
        if inputs_embeds is not None:
            np.save(os.path.join(SAVE_DIR, "craftsman_step_0_input_2048.npy"), inputs_embeds.detach().cpu().to(torch.float32).numpy())
    return None

def talker_forward_post_hook(module, input, output):
    """
    挂载在 talker.forward 之后，记录步数并保存该步 Master 的输出。
    """
    step = state.master_step
    
    # post_hook 之后增加步数
    state.master_step += 1
    
    # 捕获第 -1 步 (Prefill) 的输出，它将作为第 0 步 (Gen Start) 的 past_hidden
    if step == -1:
        current_hidden = output.past_hidden
        np.save(os.path.join(SAVE_DIR, "prefill_output_hidden.npy"), current_hidden.detach().cpu().to(torch.float32).numpy())
        print(f"[CAPTURE] Prefill Output Saved (Dim: {current_hidden.shape})")

    # 捕获第 0 步 (Gen) 的结果
    if step == 0:
        # codec_ids 包含 (input_ids, predictor_result.sequences)
        codec_ids = output.hidden_states[1] if output.hidden_states is not None else None
        if codec_ids is not None:
            np.save(os.path.join(SAVE_DIR, "master_step_0_result_codes.npy"), codec_ids.detach().cpu().numpy())
            print(f"[CAPTURE] Master Step 0 Result Codes Saved (Dim: {codec_ids.shape})")
            state.captured = True

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"载入官方模型进行组装联合数据捕获 (设备: {device})...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
        talker = tts.model.talker
        
        # 1. 挂载 Talker Hooks
        talker.register_forward_pre_hook(talker_forward_pre_hook, with_kwargs=True)
        talker.register_forward_hook(talker_forward_post_hook)
        
        # 2. 挂载 Predictor Hooks
        talker.code_predictor.register_forward_pre_hook(predictor_generate_pre_hook, with_kwargs=True)
        
        # 固定随机性
        deterministic_kwargs = {
            "do_sample": False,
            "subtalker_dosample": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        
        print("开始推理「今天天气不错」...")
        with torch.no_grad():
            tts.generate_custom_voice(
                text="今天天气不错",
                speaker="Vivian",
                language="Chinese",
                **deterministic_kwargs
            )
            
        if state.captured:
            print(f"\n✅ 组装联合数据捕获完成！保存在: {SAVE_DIR}")
        else:
            print(f"\n❌ 捕获失败。")
            
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
