"""
Configuration settings for LLM integration using Ollama.
"""

import os
from typing import Dict, Any

# Model configuration for Ollama
LLM_CONFIG = {
    # Default model name - Using Qwen3-Coder through Ollama
    "model_name": os.getenv("LLM_MODEL_NAME", "qwen3-coder"), # "qwen3-coder:30b"
    
    # Ollama server configuration
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "timeout": int(os.getenv("OLLAMA_TIMEOUT", "300")),
    
    # Generation parameters
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("LLM_TOP_P", "0.8")),
    "top_k": int(os.getenv("LLM_TOP_K", "20")),
    "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.05")),
    "seed": int(os.getenv("LLM_SEED", "-1")),  # -1 for random
    
    # Context and response length settings
    "num_ctx": int(os.getenv("LLM_NUM_CTX", "4096")),  # Context window size
    "num_predict": int(os.getenv("LLM_NUM_PREDICT", "2048")),  # Max tokens to predict
    
    # Advanced parameters
    "mirostat": int(os.getenv("LLM_MIROSTAT", "0")),  # 0=disabled, 1=Mirostat, 2=Mirostat 2.0
    "mirostat_eta": float(os.getenv("LLM_MIROSTAT_ETA", "0.1")),
    "mirostat_tau": float(os.getenv("LLM_MIROSTAT_TAU", "5.0")),
    "num_thread": int(os.getenv("LLM_NUM_THREAD", "-1")),  # -1 for auto
    
    # Fallback configuration
    "fallback_model": os.getenv("LLM_FALLBACK_MODEL", "llama3.2:3b"),
    "enable_fallback": os.getenv("LLM_ENABLE_FALLBACK", "true").lower() == "true",
}

# Logging configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

def get_model_config() -> Dict[str, Any]:
    """
    Get model configuration for Ollama client.
    
    Returns:
        dict: Model configuration parameters for Ollama
    """
    config = LLM_CONFIG.copy()
    
    # Validate Ollama host URL
    host = config["host"]
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    config["host"] = host
    
    return config

def get_fallback_config() -> Dict[str, Any]:
    """
    Get fallback configuration for environments with limited resources.
    
    Returns:
        dict: Fallback model configuration for Ollama
    """
    config = get_model_config()
    config["model_name"] = config["fallback_model"]
    config["num_predict"] = 1024
    config["num_ctx"] = 2048
    config["temperature"] = 0.3  # Lower temperature for more consistent results
    
    return config

def get_ollama_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Ollama generation options from configuration.
    For multi-host setups, each instance uses all available GPU memory.
    
    Args:
        config (dict): Full model configuration
        
    Returns:
        dict: Options suitable for Ollama generate API
    """
    options = {
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "repeat_penalty": config["repeat_penalty"],
        "seed": config["seed"],
        "num_ctx": config["num_ctx"],
        "num_predict": config["num_predict"],
        "mirostat": config["mirostat"],
        "mirostat_eta": config["mirostat_eta"],
        "mirostat_tau": config["mirostat_tau"],
    }
    
    # Add num_thread if specified (> 0)
    if config.get("num_thread", -1) > 0:
        options["num_thread"] = config["num_thread"]
    
    # Remove options with default values to keep the request clean
    clean_options = {}
    for key, value in options.items():
        if key == "seed" and value == -1:
            continue  # Skip random seed
        if key == "mirostat" and value == 0:
            continue  # Skip disabled mirostat
        clean_options[key] = value
        
    return clean_options

def get_ollama_client_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Ollama client configuration parameters.
    
    Args:
        config (dict): Full model configuration
        
    Returns:
        dict: Configuration for Ollama client initialization
    """
    client_config = {
        "host": config["host"],
        "timeout": config["timeout"],
    }
        
    return client_config