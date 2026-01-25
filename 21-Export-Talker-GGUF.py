
import os
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
        sys.exit(1)

def main():
    # 1. Prepare Talker HF (Spoofing Qwen3VL)
    print("--- Step 1: Prepare Talker HF ---")
    run_command("python 20-Prepare-Talker-HF.py")

    # 2. Convert to GGUF
    print("\n--- Step 2: Convert to GGUF ---")
    # Using internal python interpreter to run the script
    cmd = f"{sys.executable} qwen3_tts_gguf/convert_hf_to_gguf.py ./model/Talker-HF-Temp/ --outfile ./model/Qwen3-LLM-1.7B-F16.gguf --outtype f16"
    run_command(cmd)

    # 3. Simple Verification
    print("\n--- Step 3: Verify GGUF Metadata ---")
    try:
        from gguf import GGUFReader
        reader = GGUFReader("./model/Qwen3-LLM-1.7B-F16.gguf")
        
        # Helper to safely get string value
        def get_str(key):
            field = reader.get_field(key)
            if not field: return None
            val = field.parts[0]
            if isinstance(val, (bytes, bytearray)):
                try: 
                    return val.decode('utf-8').strip('\x00') 
                except: 
                    return str(val)
            # Handle numpy arrays (as lists of bytes sometimes)
            if hasattr(val, 'tolist'):
                val = val.tolist()
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], bytes):
                 return val[0].decode('utf-8').strip('\x00')
            return str(val).strip("[]'")

        arch = get_str("general.architecture")
        print(f"Architecture: {arch}")
        
        if arch == "qwen3vl":
            print("[PASS] Architecture matches")
        else:
            print(f"[FAIL] Architecture mismatch: expected qwen3vl, got {arch}")

        # Check QK-Norm existence by scanning tensor names
        tensors = reader.tensors
        has_q_norm = any("attn_q_norm" in t.name for t in tensors)
        has_k_norm = any("attn_k_norm" in t.name for t in tensors)
        
        if has_q_norm and has_k_norm:
             print("[PASS] Found attn_q_norm and attn_k_norm tensors")
        else:
             print(f"[FAIL] Missing QK-Norm tensors. Q: {has_q_norm}, K: {has_k_norm}")

    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    main()
