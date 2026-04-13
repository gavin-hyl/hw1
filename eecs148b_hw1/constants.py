from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TINYSTORIES_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VAL = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
TINYSTORIES_TRAIN_ENC = DATA_DIR / "TinyStoriesV2-GPT4-train-encoded.npy"
TINYSTORIES_VAL_ENC = DATA_DIR / "TinyStoriesV2-GPT4-valid-encoded.npy"
MODEL_DIR = DATA_DIR / "models"

VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 4
USE_LAYERNORM = True
USE_SIN_PE = False
SPECIAL_TOKENS = ["<|endoftext|>"]

OUTPUT_VOCAB = DATA_DIR / "bpe_vocab.json"
OUTPUT_MERGES = DATA_DIR / "bpe_merges.txt"