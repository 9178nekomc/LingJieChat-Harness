# config.py — 8M MoE + LoRA
import os

DATA_DIR = "data"
CKPT_DIR = "checkpoints"
PRETRAIN_DATA = os.path.join(DATA_DIR, "pretrain.jsonl")
SFT_DATA = os.path.join(DATA_DIR, "sft.jsonl")
PRETRAIN_CKPT = os.path.join(CKPT_DIR, "pretrain.pt")
SFT_CKPT = os.path.join(CKPT_DIR, "sft.pt")
LORA_CKPT = os.path.join(CKPT_DIR, "lora.pt")
TOKENIZER_PATH = os.path.join(CKPT_DIR, "tokenizer.json")

# ===== 模型（约 8M 参数） =====
DIM = 256
N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 4
D_FF = 512
MAX_SEQ_LEN = 192
N_EXPERTS = 4
TOP_K = 2
DROPOUT = 0.0

# ===== LoRA =====
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]

# ===== 预训练 =====
PRETRAIN_BATCH = 32
PRETRAIN_LR = 3e-4
PRETRAIN_EPOCHS = 10
PRETRAIN_WARMUP = 300

# ===== SFT（LoRA） =====
SFT_BATCH = 32
SFT_LR = 1e-3
SFT_EPOCHS = 8
SFT_WARMUP = 100

GRAD_CLIP = 1.0
LOG_EVERY = 50
SAVE_EVERY = 500

# ===== 特殊 token =====
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
TOOL_START = "<tool>"
TOOL_END = "</tool>"
ARGS_START = "<args>"
ARGS_END = "</args>"
