import os

DATASET_PATH = "./data"

TRAIN_PATH = os.path.join(DATASET_PATH, "train.hf")
VALID_PATH = os.path.join(DATASET_PATH, "valid.hf")

TRAIN_CLEAN_PATH = os.path.join(DATASET_PATH, "train_clean.hf")
VALID_CLEAN_PATH = os.path.join(DATASET_PATH, "valid_clean.hf")

VOCAB_PATH = os.path.join(DATASET_PATH, "vocab.json")

TRAIN_ENCODED_PATH = os.path.join(DATASET_PATH, "train_encoded.hf")
VALID_ENCODED_PATH = os.path.join(DATASET_PATH, "valid_encoded.hf")

# vocab params

PAD = "<PAD>"
PAD_IDX = 0

EOS = "<EOS>"
EOS_IDX = 1

UNK = "<UNK>"
UNK_IDX = 2

N_ENG_TOKENS = 115736
N_ID_TOKENS = 136818

