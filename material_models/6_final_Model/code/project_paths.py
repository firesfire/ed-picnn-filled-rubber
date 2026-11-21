"""Resolve paths in material-model configurations relative to the code package."""
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[2]


def load_config(content):
    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str) and value.startswith("@project/"):
            return str(PROJECT / value[len("@project/"):])
        return value

    return resolve(yaml.safe_load(content))
