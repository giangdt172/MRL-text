"""Encoder aliases and the original MIPIC alignment schedules.

Layer indices refer directly to Hugging Face hidden_states (index 0 is embeddings).
"""

MODEL_CONFIGS = {
    "tinybert": {
        "model_name": "huawei-noah/TinyBERT_General_6L_768D",
        "matryoshka_dims": [768, 512, 256, 128, 64, 32, 16],
        "align_layers": [2, 4],
        "pipeline_pairs": [(1, 16, 3, 128, 5, 768)],
        "d_full": 768,
    },
    "bert": {
        "model_name": "google-bert/bert-base-uncased",
        "matryoshka_dims": [768, 512, 256, 128, 64, 32, 16],
        "align_layers": [8, 10],
        "pipeline_pairs": [(3, 16, 7, 128, 11, 768)],
        "d_full": 768,
    },
    "qwen": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "matryoshka_dims": [1024, 512, 256, 128, 64, 32, 16],
        "align_layers": [2, 8,14,20, 23],
        "pipeline_pairs": [(2, 16, 8, 64,14,128, 20, 512, 23, 1024)],
        "d_full": 1024,
    },
    "bgem3": {
        "model_name": "BAAI/bge-m3",
        "matryoshka_dims": [1024, 512, 256, 128, 64, 32, 16],
        "align_layers": [2, 8,14,20, 23],
        "pipeline_pairs": [(2, 16, 8, 64,14,128, 20, 512, 23, 1024)],
        "d_full": 1024,
    },
}

model_name_map_short_name = {
    MODEL_CONFIGS[short_name]["model_name"]: short_name for short_name in MODEL_CONFIGS
}

