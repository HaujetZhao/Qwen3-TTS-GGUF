import os
import numpy as np

def inspect_array(name, arr):
    print(f"\n--- Inspecting: {name} ---")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Size in memory: {arr.nbytes} bytes")
    print(f"C_CONTIGUOUS: {arr.flags['C_CONTIGUOUS']}")
    print(f"F_CONTIGUOUS: {arr.flags['F_CONTIGUOUS']}")
    print(f"Data pointer address: {arr.ctypes.data}")
    
    # Check for anomalies
    print(f"Min:  {np.min(arr):.8f}")
    print(f"Max:  {np.max(arr):.8f}")
    print(f"Mean: {np.mean(arr):.8f}")
    print(f"Has NaN: {np.isnan(arr).any()}")
    print(f"Has Inf: {np.isinf(arr).any()}")
    
    # Sample values
    print(f"First 5 values of first token: {arr[0, :5] if arr.ndim > 1 else arr[:5]}")

def main():
    OFFICIAL_EMBEDS_PATH = os.path.abspath("40_first_step_embeds.npy")
    
    # 1. Generate Successful Random Data (Simulation of Script 41)
    random_data = np.random.randn(14, 2048).astype(np.float32)
    inspect_array("Random Data (Succeeded)", random_data)
    
    # 2. Load Failing Official Data (Simulation of Script 42)
    if os.path.exists(OFFICIAL_EMBEDS_PATH):
        official_data_raw = np.load(OFFICIAL_EMBEDS_PATH)
        inspect_array("Official Data Raw", official_data_raw)
        
        # Squeeze if needed
        if official_data_raw.ndim == 3:
            official_data = official_data_raw[0]
            inspect_array("Official Data Squeezed (used in 42)", official_data)
            
            # Simulated transformation in engine.py
            engine_data = official_data.astype(np.float32)
            if not engine_data.flags['C_CONTIGUOUS']:
                engine_data = np.ascontiguousarray(engine_data)
            inspect_array("Official Data after Engine.py Transforms", engine_data)
    else:
        print(f"❌ Error: {OFFICIAL_EMBEDS_PATH} not found.")

if __name__ == "__main__":
    main()
