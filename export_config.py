from dataclasses import dataclass
from pathlib import Path

model_home = Path('~/.cache/modelscope/hub/models/Qwen').expanduser()
dest_home = Path(__file__).parent

@dataclass
class ModelConfig:
    source: Path
    dest: Path

class Models:
    base = ModelConfig(
        model_home / 'Qwen3-TTS-12Hz-1.7B-Base',
        dest_home / 'model-base'
    )
    custom = ModelConfig(
        model_home / 'Qwen3-TTS-12Hz-1.7B-CustomVoice', 
        dest_home / 'model-custom'
    )
    design       = ModelConfig(
        model_home / 'Qwen3-TTS-12Hz-1.7B-VoiceDesign', 
        dest_home / 'model-design'
    )
    base_small       = ModelConfig(
        model_home / 'Qwen3-TTS-12Hz-0.6B-Base', 
        dest_home / 'model-base-small'
    )
    custom_small       = ModelConfig(
        model_home / 'Qwen3-TTS-12Hz-0.6B-CustomVoice', 
        dest_home / 'model-custom-small'
    )


model = Models.base

MODEL_DIR  = model.source
EXPORT_DIR = model.dest
