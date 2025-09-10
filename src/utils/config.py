import yaml
from pathlib import Path

def load_config(name):
    config_path = Path(__file__).parent.parent / "configs" / f"{name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
