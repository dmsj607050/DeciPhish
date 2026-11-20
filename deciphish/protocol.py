"""Constants for the reported Qwen3-0.6B experiment."""

MAX_TOKENS = 4096
REGION_NAMES = ("URL", "Head", "Interactive", "Body")
REGION_HEADERS = (
    "URL evidence:\n",
    "HTML head evidence:\n",
    "HTML interactive evidence:\n",
    "HTML body evidence:\n",
)
# Keep training order fixed to preserve the order of dropout RNG draws.
ORDERS = {"url_html": (0, 1, 2, 3), "html_url": (1, 2, 3, 0)}
PRIMARY_VIEW = "html_url"
EVAL_VIEWS = ("html_url", "url_html")
THRESHOLD = 0.5

SEED = 20261027
BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.03
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DATASET_SHA256 = {
    "train": "3ca3975a6e6f0c768fb7b686f7a351721b0fce0a640ca49273b0dde300105ae7",
    "test": "8b341546a8115322d634f6f692ab9d0b8a75b9b25fbaa297974ac1cd5290d9c5",
}
