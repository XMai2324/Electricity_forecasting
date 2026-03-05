import json
import joblib

def load_model(model_path: str):
    return joblib.load(model_path)

def load_feature_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)