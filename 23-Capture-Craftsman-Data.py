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
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_craftsman")
os.makedirs(SAVE_DIR, exist_ok=True)

class CaptureState:
    def __init__(self):
        self.master_step = -1  # -1 表示 Prefill，以 0 开始表示第一个生成步
        self.pred_step = 0
        self.target_master_step = 0 # 我们只捕获 Master 的第一个生成步

state = CaptureState()

def master_backbone_post_hook(module, input, output):
    """
    监控 Master (Talker Backbone) 的步数。
    每执行一次 forward，步数加 1。
    """
    state.master_step += 1
    return None

def predictor_model_pre_hook(module, args, kwargs):
    """
    拦截工匠 (Code Predictor) 的 Transformer 主干输入。
    """
    if state.master_step != state.target_master_step:
        return None
    
    # 1. 拦截输入隐藏状态 (进入 Transformer 前的，已投影)
    inputs_embeds = kwargs.get('inputs_embeds')
    # 2. 拦截初始 KV Cache
    past_key_values = kwargs.get('past_key_values')
    
    if inputs_embeds is not None:
        step = state.pred_step
        # 保存已投影的输入 (用于验证 Transformer 主干)
        np.save(os.path.join(SAVE_DIR, f"step_{step}_projected_input.npy"), 
                inputs_embeds.detach().cpu().to(torch.float32).numpy())
        
        # 保存进入时的 KV Cache (如果有)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                k, v = past_key_values[layer_idx]
                if k is not None:
                    np.save(os.path.join(SAVE_DIR, f"step_{step}_layer_{layer_idx}_k_input.npy"), 
                            k.detach().cpu().to(torch.float32).numpy())
                if v is not None:
                    np.save(os.path.join(SAVE_DIR, f"step_{step}_layer_{layer_idx}_v_input.npy"), 
                            v.detach().cpu().to(torch.float32).numpy())
            
        print(f"[CAPTURE] Master Step {state.master_step}, Pred Step {step}: Inputs Saved.")
    return None

def predictor_model_post_hook(module, input, output):
    """
    拦截工匠 (Code Predictor) 的 Transformer 主干输出。
    """
    if state.master_step != state.target_master_step:
        return None
        
    hidden_states = output[0] if isinstance(output, (list, tuple)) else output.last_hidden_state
    
    step = state.pred_step
    # 保存输出隐藏状态
    np.save(os.path.join(SAVE_DIR, f"step_{step}_output_hidden.npy"), 
            hidden_states.detach().cpu().to(torch.float32).numpy())
    
    # 捕获更新后的 KV Cache (Present)
    # output[1] 在 Qwen3 中通常是 Cache 对象
    if hasattr(output, 'past_key_values'):
        pkv = output.past_key_values
        for layer_idx in range(len(pkv)):
            k, v = pkv[layer_idx]
            if k is not None:
                np.save(os.path.join(SAVE_DIR, f"step_{step}_layer_{layer_idx}_k_present.npy"), 
                        k.detach().cpu().to(torch.float32).numpy())
            if v is not None:
                np.save(os.path.join(SAVE_DIR, f"step_{step}_layer_{layer_idx}_v_present.npy"), 
                        v.detach().cpu().to(torch.float32).numpy())
    
    print(f"[CAPTURE] Pred Step {step}: Internal Outputs and Present KV Saved.")
    state.pred_step += 1
    return None

def predictor_wrapper_pre_hook(module, args, kwargs):
    """
    拦截工匠整体 Forward 的输入 (未投影的小维 Embedding)。
    这是为了验证 ONNX Wrapper 整体。
    """
    if state.master_step != state.target_master_step:
        return None
    
    # 在 generate 模式下，Predictor 的 forward 可能被多次调用，
    # 或者由 predictor.generate 内部循环调用。
    # 我们关注的是 predictor 整体接收到的 inputs_embeds。
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is not None:
        step = state.pred_step
        np.save(os.path.join(SAVE_DIR, f"step_{step}_raw_small_input.npy"), 
                inputs_embeds.detach().cpu().to(torch.float32).numpy())
        print(f"[CAPTURE] Pred Step {step}: Raw Small Input Saved.")
    return None

def projection_pre_hook(module, args, kwargs):
    """
    拦截投影层的输入 (2048 维)。
    这是 ONNX 模型真正的输入。
    """
    if state.master_step != state.target_master_step:
        return None
    
    input_tensor = args[0] if len(args) > 0 else kwargs.get('input')
    if input_tensor is not None:
        step = state.pred_step
        np.save(os.path.join(SAVE_DIR, f"step_{step}_input_2048.npy"), 
                input_tensor.detach().cpu().to(torch.float32).numpy())
        print(f"[CAPTURE] Pred Step {step}: 2048-dim Input Saved.")
    return None

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"载入官方模型进行工匠数据捕获 (设备: {device})...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
        talker = tts.model.talker
        predictor = talker.code_predictor
        
        # 1. 插桩 Master Backbone 以记录 Master 步数
        talker.model.register_forward_hook(master_backbone_post_hook)
        
        # 2. 插桩投影层以捕获 2048 维输入
        predictor.small_to_mtp_projection.register_forward_pre_hook(projection_pre_hook, with_kwargs=True)
        
        # 3. 插桩 Predictor Transformer 主干以捕获中间状态
        predictor.model.register_forward_pre_hook(predictor_model_pre_hook, with_kwargs=True)
        predictor.model.register_forward_hook(predictor_model_post_hook)
        
        # 4. 插桩 Predictor 整体以捕获最终被选中的 Token ID
        # 注意: 由于 model_post_hook 会让 step + 1，所以这里的 step 要 -1
        def predictor_wrapper_post_hook(module, input, output):
            if state.master_step != state.target_master_step:
                return None
            
            # output 可能是 CausalLMOutputWithPast 或 tuple(logits, past_key_values)
            if hasattr(output, 'logits'):
                logits = output.logits
            elif isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                return None
            
            # Logits shape: [batch, 1, vocab_size] (usually)
            # Take argmax
            token_id = torch.argmax(logits[:, -1, :], dim=-1) # [batch]
            
            # 因为 model_post_hook 已经把 step 加 1 了，所以这里对应的是上一步
            step = state.pred_step - 1
            if step >= 0:
                np.save(os.path.join(SAVE_DIR, f"step_{step}_output_ids.npy"), 
                        token_id.detach().cpu().to(torch.int32).numpy())
                print(f"[CAPTURE] Pred Step {step}: Output Token ID Saved.")
            return None

        predictor.register_forward_hook(predictor_wrapper_post_hook)
        
        # 固定随机性
        deterministic_kwargs = {
            "do_sample": False,
            "subtalker_dosample": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        
        print("开始推理「今天天气好」...")
        with torch.no_grad():
            tts.generate_custom_voice(
                text="今天天气好",
                speaker="Vivian",
                language="Chinese",
                **deterministic_kwargs
            )
        
        print(f"\n✅ 工匠数据捕获完成！")
        print(f"文件保存在: {SAVE_DIR}")
        print(f"总计捕获工匠推理步数: {state.pred_step} (应为 15 步)")
        
    except BaseException as e:
        sys.stdout.flush()
        print(f"❌ 捕获失败 (BaseException): {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    main()
