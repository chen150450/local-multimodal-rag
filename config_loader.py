#!/usr/bin/env python3
"""
Configuration loader for Local Multimodal RAG Pipeline.

Loads configuration from:
1. config.yaml - main configuration file
2. .env - environment variables (for sensitive data like API keys)

Usage:
    from config_loader import get_config
    
    config = get_config()
    db_path = config['database']['path']
    api_key = config.get_env('QWEN_API_KEY')
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional

# Default config file path (relative to this module)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Global config cache
_config_cache: Optional[dict] = None
_env_loaded: bool = False


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Load environment variables from .env file.
    
    Does NOT override existing environment variables.
    """
    global _env_loaded
    
    if _env_loaded:
        return
    
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        return
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = value
        _env_loaded = True
    except Exception:
        pass


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml (default: same directory as this module)
    
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Load .env first
    load_env_file()
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create config.yaml from config.yaml template."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand paths with ~ and environment variables
    config = _expand_paths(config)
    
    _config_cache = config
    return config


def _expand_paths(config: dict) -> dict:
    """Expand ~ in paths and substitute environment variables."""
    if isinstance(config, dict):
        return {k: _expand_paths(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_paths(item) for item in config]
    elif isinstance(config, str):
        # Expand ~ to home directory
        if config.startswith('~'):
            config = os.path.expanduser(config)
        # Substitute environment variables ${VAR} or $VAR
        # (simple substitution, not full shell syntax)
        if '$' in config:
            for key, value in os.environ.items():
                config = config.replace(f'${{{key}}}', value)
                config = config.replace(f'$:{key}', value)  # Avoid replacing $ in URLs
        return config
    else:
        return config


def get_config(config_path: Optional[Path] = None) -> dict:
    """Get configuration (cached).
    
    Args:
        config_path: Optional path to config.yaml
    
    Returns:
        Configuration dictionary
    """
    return load_config(config_path)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable (loads .env if not loaded).
    
    Args:
        key: Environment variable name
        default: Default value if not found
    
    Returns:
        Environment variable value or default
    """
    load_env_file()
    return os.environ.get(key, default)


def get_env_required(key: str) -> str:
    """Get required environment variable (raises if not found).
    
    Args:
        key: Environment variable name
    
    Returns:
        Environment variable value
    
    Raises:
        ValueError: If variable not found
    """
    value = get_env(key)
    if value is None:
        raise ValueError(
            f"Required environment variable '{key}' not found.\n"
            f"Please set it in .env file or environment."
        )
    return value


def get_db_path() -> str:
    """Get database path from config."""
    config = get_config()
    return config['database']['path']


def get_log_dir() -> str:
    """Get log directory from config."""
    config = get_config()
    log_dir = config['logging']['dir']
    # Create directory if not exists
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_embedding_dim() -> int:
    """Get embedding dimension from config."""
    config = get_config()
    return config['embedding']['dimension']


def get_ocr_api_url() -> str:
    """Get OCR API URL from config or environment."""
    # Environment override
    env_url = get_env('OCR_API_URL')
    if env_url:
        return env_url
    config = get_config()
    return config['ocr']['api_url']


def get_chunking_config() -> dict:
    """Get chunking configuration."""
    config = get_config()
    return config['chunking']


def get_limits_config() -> dict:
    """Get file processing limits."""
    config = get_config()
    return config['limits']


def get_vision_config() -> dict:
    """Get vision model configuration."""
    config = get_config()
    return config['vision']


def get_pipeline_config() -> dict:
    """Get pipeline configuration."""
    config = get_config()
    return config['pipeline']


# Convenience functions for backward compatibility
def get_jina_model_path() -> str:
    """Get Jina model path."""
    config = get_config()
    return config['embedding']['model_path']


def get_vllm_api_url() -> str:
    """Get vLLM API URL."""
    env_url = get_env('VLLM_API_URL')
    if env_url:
        return env_url
    config = get_config()
    return config['embedding']['vllm']['api_url']


def get_vllm_model_name() -> str:
    """Get vLLM model name."""
    env_name = get_env('VLLM_MODEL_NAME')
    if env_name:
        return env_name
    config = get_config()
    return config['embedding']['vllm']['model_name']


# Reset cache (for testing or reloading config)
def reset_config_cache():
    """Reset configuration cache."""
    global _config_cache, _env_loaded
    _config_cache = None
    _env_loaded = False


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loader...")
    config = get_config()
    print(f"Database path: {get_db_path()}")
    print(f"Log directory: {get_log_dir()}")
    print(f"Embedding dim: {get_embedding_dim()}")
    print(f"OCR API URL: {get_ocr_api_url()}")
    print(f"Chunking config: {get_chunking_config()}")
    print("Configuration loaded successfully!")