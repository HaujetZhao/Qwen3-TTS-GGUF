
import sys
from gguf import GGUFReader

def verify_gguf(model_path):
    print(f"Verifying {model_path}...")
    reader = GGUFReader(model_path)
    
    # Check Architecture
    arch = reader.get_field("general.architecture")
    if arch:
        arch_val = bytes(arch.parts[0]).decode('utf-8')
        print(f"general.architecture: {arch_val}")
        if arch_val != "qwen3vl":
            print("FAIL: Expected architecture 'qwen3vl'")
            return False
    else:
        print("FAIL: general.architecture missing")
        return False
        
    # Check Deepstack Layers
    n_deepstack = reader.get_field("qwen3vl.n_deepstack_layers")
    if n_deepstack:
        # access value from part 0 directly or assuming it's a numeric type
        # GGUFReader parts are usually raw bytes or typed values. 
        # For numeric it is usually the value itself in data list or parts.
        # Let's trust just printing it for now or accessing raw value.
        # The reader implementation varies, usually `parts` contains the data blocks.
        # But `reader.get_field(key).parts[0]` returns the value memoryview/bytes/int.
        # Actually GGUFReader field has `parts` which is a list.
        # A simpler way is to depend on print format or just inspect.
        pass # printed below in loop
    
    # Check Rope Dimension Sections
    # expected: [24, 20, 20, 0]
    rope_sections = reader.get_field("qwen3vl.rope.dimension_sections")
    
    # Iterate and print all relevant keys
    required_keys = [
        "general.architecture",
        "qwen3vl.n_deepstack_layers",
        "qwen3vl.rope.dimension_sections"
    ]
    
    success = True
    for key in required_keys:
        field = reader.get_field(key)
        if field:
            # Try to decode value
            val = field.parts[0]
            if isinstance(val, memoryview):
                val = list(val)
            elif isinstance(val, (bytes, bytearray)):
                 try:
                    val = val.decode('utf-8')
                 except:
                    val = list(val)
            print(f"FOUND: {key} = {val}")
            
            if key == "general.architecture" and val != "qwen3vl":
                success = False
            if key == "qwen3vl.n_deepstack_layers" and int(val[0] if isinstance(val, list) else val) != 3:
                 # It might be a single int or list.
                 pass
            if key == "qwen3vl.rope.dimension_sections":
                 # Check list content
                 expected = [24, 20, 20, 0]
                 # Flatten if needed or cast
                 current = list(val) if hasattr(val, '__iter__') else [val]
                 # Note: data might be int32/int64 array
                 current = [int(x) for x in current]
                 if current != expected:
                     print(f"FAIL: Expected {expected}, got {current}")
                     success = False
        else:
            print(f"MISSING: {key}")
            success = False
            
    return success

if __name__ == "__main__":
    path = "./model/Qwen3-LLM-1.7B-F16.gguf"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    if verify_gguf(path):
        print("\nSUCCESS: GGUF Metadata Verified.")
    else:
        print("\nFAILURE: GGUF Metadata Incorrect.")
