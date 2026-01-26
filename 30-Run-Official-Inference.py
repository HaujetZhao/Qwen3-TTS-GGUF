import os
import sys
import time
import torch
import soundfile as sf
import subprocess

# Add current directory to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR) # Prioritize local source

import qwen_tts
print(f"Imported qwen_tts from: {qwen_tts.__file__}")

from qwen_tts import Qwen3TTSModel

def play_audio(file_path):
    print(f"Playing {file_path}...")
    # Use powershell to play audio
    try:
        subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();"], check=True)
    except Exception as e:
        print(f"Failed to play audio: {e}")

def main():
    # Use CPU or CUDA if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Local model path
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found: {MODEL_PATH}")
        return

    print(f"Loading model from: {MODEL_PATH}")
    
    try:
        # Measure Model Loading Time
        print("Starting model load...")
        
        # Define dtype
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        
        t_load_start = time.time()
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map=device,
            dtype=dtype,
            # attn_implementation="flash_attention_2" # Comment out if no flash attn support on Windows/CPU
        )
        t_load_end = time.time()
        load_time = t_load_end - t_load_start
        print(f"Model loaded in {load_time:.4f} seconds.")
        
        text = "今天天气好"
        speaker = "Vivian" # Using a preset speaker
        output_file = "30_qwen3_inference_output.wav"

        print(f"Generating audio for text: '{text}' with speaker: '{speaker}'")

        # Measure Inference Time
        t_infer_start = time.time()
        wavs, sr = tts.generate_custom_voice(
            text=text,
            language="Chinese",
            speaker=speaker,
            instruct="", # No specific instruction
        )
        t_infer_end = time.time()
        infer_time = t_infer_end - t_infer_start
        print(f"Inference completed in {infer_time:.4f} seconds.")


        # Save audio
        sf.write(output_file, wavs[0], sr)
        print(f"Audio saved to: {output_file}")
        
        # Play audio
        play_audio(output_file)

    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
