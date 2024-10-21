import os

from config import *

from preprocessing.preprocessing import tokenize, save_vocab_to_disk

from datasets import Dataset
from tqdm import tqdm

if __name__ == "__main__":

    if not os.path.exists(TRAIN_CLEAN_PATH) or not os.path.exists(VALID_CLEAN_PATH):
        print("[Vocab] Error: clean dataset does not exist on disk.")
        exit(1)

    vocab = {"id": set(), "eng": set()}

    # load clean train data
    ds = Dataset.load_from_disk(TRAIN_CLEAN_PATH)["text"]
    print(f"[Vocab] Loaded {len(ds)} clean train samples.")

    for text in tqdm(ds):
        # tokenize cleaned and filtered train data
        tokens = tokenize(text)
    
        for tk_id in tokens["id"]:
            vocab["id"].add(tk_id)

        
        # Add English tokens to the vocabulary
        for tk_eng in tokens["eng"]:
            vocab["eng"].add(tk_eng)

    # vocab["id"] = sorted(vocab["id"])
    # vocab["eng"] = sorted(vocab["eng"])

    vocab["id"] = ["<EOS>", "<UNK>"] + list(vocab["id"])
    vocab["eng"] = ["<EOS>", "<UNK>"] + list(vocab["eng"])

    print(f"[Vocab] Found {len(vocab["id"])} Indonesian tokens.")
    print(f"[Vocab] Found {len(vocab["eng"])} English tokens.")

    save_vocab_to_disk(vocab, VOCAB_PATH)
    print("[Vocab] Saved vocab to:", VOCAB_PATH)