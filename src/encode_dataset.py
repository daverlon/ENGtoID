import os

from config import *

from utils.preprocessing import load_vocab_from_disk, tokenize
from utils.coder import Coder

from datasets import Dataset
from tqdm import tqdm

END_TOKEN = "<EOS>" # end of sentence

if __name__ == "__main__":

    if not os.path.exists(TRAIN_CLEAN_PATH) or not os.path.exists(VALID_CLEAN_PATH):
        print("[Encode] Error: clean dataset not found.")
        exit(1)

    if not os.path.exists(VOCAB_PATH):
        print("[Encode] Error: vocab.json not found.")
        exit(1)

    # load vocab
    vocab = load_vocab_from_disk(VOCAB_PATH)

    # create encoder/decoder
    coder_id = Coder(vocab["id"])
    coder_eng = Coder(vocab["eng"])

    # =============== TRAIN =============== 

    # load train dataset
    ds = Dataset.load_from_disk(TRAIN_CLEAN_PATH)["text"]
    print(f"[Encode] Loaded {len(ds)} clean train samples.")

    encoded_ds = []

    # tokenize
    for text in tqdm(ds):
        tokens = tokenize(text)
        id_tokens, eng_tokens = tokens["id"], tokens["eng"]

        # check for unknown tokens
        # unknown_id_tokens = [token for token in id_tokens if token not in vocab["id"]]
        # unknown_eng_tokens = [token for token in eng_tokens if token not in vocab["eng"]]
        # if unknown_id_tokens:
        #     print(f"[Encode] Error: Unknown Indonesian tokens found: {unknown_id_tokens}")
        # if unknown_eng_tokens:
        #     print(f"[Encode] Error: Unknown English tokens found: {unknown_eng_tokens}")

        encoded_id_tokens = coder_id.encode(id_tokens)
        encoded_eng_tokens = coder_eng.encode(eng_tokens)
        # print(encoded_id_tokens, encoded_eng_tokens)

        # ds[i] = {"id": encoded_id_tokens, "eng": encoded_eng_tokens}
        encoded_ds.append({"id": encoded_id_tokens, "eng": encoded_eng_tokens})
    
    # save the encoded data
    ds = Dataset.from_dict({"text": encoded_ds})
    ds.save_to_disk(TRAIN_ENCODED_PATH)
    print("[Encode] Saved:", TRAIN_ENCODED_PATH)

    # =============== VALID =============== 

    # load valid dataset
    ds = Dataset.load_from_disk(VALID_CLEAN_PATH)["text"]
    print(f"[Encode] Loaded {len(ds)} raw valid samples.")

    encoded_ds = []

    # tokenize
    for text in tqdm(ds):
        tokens = tokenize(text)
        id, eng = tokens["id"], tokens["eng"]

        encoded_id = coder_id.encode(id)
        encoded_eng = coder_eng.encode(eng)

        encoded_ds.append({"id": encoded_id, "eng": encoded_eng})
    
    # save the encoded data
    ds = Dataset.from_dict({"text": encoded_ds})
    ds.save_to_disk(VALID_ENCODED_PATH)
    print("[Encode] Saved:", VALID_ENCODED_PATH)

    print("[Encode] Finished.")