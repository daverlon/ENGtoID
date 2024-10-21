import os

DATASET_PATH = "./data"

TRAIN_PATH = os.path.join(DATASET_PATH, "train.hf")
VALID_PATH = os.path.join(DATASET_PATH, "valid.hf")

TRAIN_CLEAN_PATH = os.path.join(DATASET_PATH, "train_clean.hf")
VALID_CLEAN_PATH = os.path.join(DATASET_PATH, "valid_clean.hf")

VOCAB_PATH = os.path.join(DATASET_PATH, "vocab.json")

TRAIN_ENCODED_PATH = os.path.join(DATASET_PATH, "train_encoded.hf")
VALID_ENCODED_PATH = os.path.join(DATASET_PATH, "valid_encoded.hf")

N_ENG_TOKENS = 115735
N_ID_TOKENS = 136817
