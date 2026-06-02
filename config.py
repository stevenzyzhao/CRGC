import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

DELTA = 0.3
TAU = 0.8
N_SAMPLES = 11
N_REPEATS = 10
MAX_REFINEMENT_TURNS = 6
TEMPERATURE = 0.7
BRIDGE_TEMPERATURE = 0.7

MODELS = {
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o"},
    "claude-3.7-sonnet": {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514"},
    "gemini-2.5-pro": {"provider": "google", "model_id": "gemini-2.5-pro-preview-06-05"},
    "qwen2.5-72b": {"provider": "vllm", "model_id": "Qwen/Qwen2.5-72B-Instruct"},
    "llama-3.1-72b": {"provider": "vllm", "model_id": "meta-llama/Llama-3.1-70B-Instruct"},
    "mixtral-8x22b": {"provider": "vllm", "model_id": "mistralai/Mixtral-8x22B-Instruct-v0.1"},
}

EVAL_MODEL = "claude-3.7-sonnet"
EXTRACTION_MODEL = "gpt-4o"
BRIDGE_GENERATION_MODEL = "claude-3.7-sonnet"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./results")
CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")
