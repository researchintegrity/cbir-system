import os
import yaml
from pydantic import BaseModel
from typing import Optional

class AppConfig(BaseModel):
    host: str
    port: int
    workers: int

class ModelConfig(BaseModel):
    name: str
    type: str
    path: str
    download_url: str
    device: str
    embedding_dim: int
    input_size: int

class MilvusConfig(BaseModel):
    host: str
    port: int
    collection_name: str
    metric_type: str
    index_type: str
    nlist: int

class StorageConfig(BaseModel):
    base_path: str

class Config(BaseModel):
    app: AppConfig
    model: ModelConfig
    milvus: MilvusConfig
    storage: StorageConfig

def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        # Fallback for running from src/
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Override with environment variables if needed (basic implementation)
    if os.getenv("MILVUS_HOST"):
        config_dict["milvus"]["host"] = os.getenv("MILVUS_HOST")
    if os.getenv("MODEL_DEVICE"):
        config_dict["model"]["device"] = os.getenv("MODEL_DEVICE")
    if os.getenv("CBIR_SERVICE_PORT"):
        config_dict["app"]["port"] = int(os.getenv("CBIR_SERVICE_PORT"))

    return Config(**config_dict)

# Global config instance
try:
    settings = load_config()
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    settings = None
