import torch
from bare_master.configuration_bare_master import Qwen3TTSTalkerConfig
from bare_master.modeling_bare_master import Qwen3TTSTalkerModel
import json

def test_load():
    MODEL_PATH = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    CONFIG_PATH = f"{MODEL_PATH}/config.json"
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    
    # Try to initialize config
    print("Initializing TalkerConfig...")
    talker_cfg = Qwen3TTSTalkerConfig(**full_config['talker_config'])
    print(f"Success! Hidden size: {talker_cfg.hidden_size}")
    
    # Try to initialize model
    print("Initializing TalkerModel (Backbone)...")
    model = Qwen3TTSTalkerModel(talker_cfg)
    print("Success! Model initialized.")

if __name__ == "__main__":
    try:
        test_load()
        print("\n✅ Bare Master module loaded successfully!")
    except Exception as e:
        print(f"\n❌ Failed to load Bare Master module: {e}")
        import traceback
        traceback.print_exc()
