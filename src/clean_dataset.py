import os

from config import *

from preprocessing.preprocessing import is_text_clean

from datasets import Dataset

if __name__ == "__main__":

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VALID_PATH):
        print("[Clean] Error: Dataset not found.")
        exit(1)
    
    # =============== TRAIN =============== 

    ds = Dataset.load_from_disk(TRAIN_PATH)["text"]
    print(f"[Clean] Loaded {len(ds)} raw train samples.")

    # clean and filter data
    ds_clean = []
    for text in ds:
        if is_text_clean(text):
            ds_clean.append(text)
    print(f"[Clean] Found {len(ds_clean)} clean train samples.") 

    # save the cleaned data
    ds_clean = Dataset.from_dict({"text": ds_clean})
    ds_clean.save_to_disk(TRAIN_CLEAN_PATH)
    print("[Clean] Saved:", TRAIN_CLEAN_PATH)

    # =============== VALID =============== 

    ds = Dataset.load_from_disk(VALID_PATH)["text"]
    print(f"[Clean] Loaded {len(ds)} raw valid samples.")

    # clean and filter data
    ds_clean = []
    for text in ds:
        if is_text_clean(text):
            ds_clean.append(text)
    print(f"[Clean] Found {len(ds_clean)} clean valid samples.") 

    # save the cleaned data
    ds_clean = Dataset.from_dict({"text": ds_clean})
    ds_clean.save_to_disk(VALID_CLEAN_PATH)
    print("[Clean] Saved:", VALID_CLEAN_PATH)

    print("[Clean] Finished.")