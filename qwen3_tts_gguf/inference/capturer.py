import functools
import types
import numpy as np
import torch
from .schema.result import TTSResult, Timing

class OfficialCapturer:
    """
    打桩捕获器：拦截官方模型的内部调用以自动提取关键中间变量。
    同时包装高层 API 使其直接返回 TTSResult。
    """
    def __init__(self, tts_wrapper):
        self.tts = tts_wrapper
        self.model = tts_wrapper.model
        self.talker = self.model.talker
        
        # 缓存区
        self.last_codes = None
        self.last_spk_emb = None
        self.last_text_ids = None

        self._patch_all()

    def _patch_all(self):
        # 1. 拦截 model.generate 获取 codes 和 text_ids
        orig_gen = self.model.generate
        @functools.wraps(orig_gen)
        def hooked_gen(inner_self, *args, **kwargs):
            # 自动捕获 input_ids 并提取核心文本 Token
            input_ids = kwargs.get("input_ids") or (args[0] if args else None)
            if input_ids is not None:
                raw_ids = input_ids[0][0].tolist()
                # 官方格式启发式：[4:-5] 对应中间的文本内容
                if len(raw_ids) > 9:
                    self.last_text_ids = raw_ids[4:-5]
                else:
                    self.last_text_ids = raw_ids

            res_codes, res_hiddens = orig_gen(*args, **kwargs)
            if res_codes and len(res_codes) > 0:
                self.last_codes = res_codes[0].detach().cpu().numpy()
            return res_codes, res_hiddens
        self.model.generate = types.MethodType(hooked_gen, self.model)

        # 2. 拦截嵌入层捕获 speaker_embed (针对 CustomVoice 模式)
        talker_emb_layer = self.talker.get_input_embeddings()
        orig_emb_forward = talker_emb_layer.forward
        @functools.wraps(orig_emb_forward)
        def hooked_emb_forward(inner_layer_self, indices):
            emb = orig_emb_forward(indices)
            if self.last_spk_emb is None:
                if indices.dim() == 0 or (indices.dim() == 1 and indices.size(0) == 1):
                    self.last_spk_emb = emb.detach().cpu().to(torch.float32).numpy().flatten()
            return emb
        talker_emb_layer.forward = types.MethodType(hooked_emb_forward, talker_emb_layer)

        # 3. 拦截 talker.forward 捕捉传入的 speaker_embed (针对 Clone/Design 模式)
        orig_talker_forward = self.talker.forward
        @functools.wraps(orig_talker_forward)
        def hooked_talker_forward(inner_self, **kwargs):
            if "speaker_embed" in kwargs:
                spk_emb = kwargs["speaker_embed"]
                if spk_emb is not None:
                    self.last_spk_emb = spk_emb[0].detach().cpu().to(torch.float32).numpy().flatten()
            return orig_talker_forward(**kwargs)
        self.talker.forward = types.MethodType(hooked_talker_forward, self.talker)

        # 4. 包装高层 API，使其直接返回 TTSResult
        methods = ["generate_custom_voice", "generate_voice_design", "generate_voice_clone"]
        for m_name in methods:
            if hasattr(self.tts, m_name):
                orig_m = getattr(self.tts, m_name)
                def make_hooked(original):
                    @functools.wraps(original)
                    def hooked(inner_self, *args, **kwargs):
                        # 重置缓冲区，确保每一轮生成都是独立的
                        self.last_codes = None
                        self.last_spk_emb = None
                        self.last_text_ids = None
                        
                        text = kwargs.get("text") or (args[0] if args else "")
                        
                        # 执行原始推理逻辑 (wavs, sr)
                        wavs, sr = original(*args, **kwargs)
                        
                        # 自动汇总并封装返回
                        return self.pull_result(text, wavs[0])
                    return hooked
                setattr(self.tts, m_name, types.MethodType(make_hooked(orig_m), self.tts))

    def pull_result(self, text, audio_wav):
        """汇总当前捕获的数据并封装为 TTSResult"""
        res = TTSResult(
            text=text,
            spk_emb=self.last_spk_emb,
            text_ids=self.last_text_ids if self.last_text_ids is not None else [],
            codes=self.last_codes,
            audio=audio_wav,
            stats=Timing(total_steps=len(self.last_codes) if self.last_codes is not None else 0)
        )
        return res
