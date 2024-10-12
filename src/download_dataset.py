import os

from config import *

from datasets import load_dataset

def download_dataset(dir: str = "./"):
    ds = load_dataset("kaitchup/opus-Indonesian-to-English", split="train")
    ds.save_to_disk(os.path.join(dir, TRAIN_PATH))
    print("[Download] Saved:", TRAIN_PATH)

    ds = load_dataset("kaitchup/opus-Indonesian-to-English", split="validation")
    ds.save_to_disk(os.path.join(dir, VALID_PATH))
    print("[Download] Saved:", VALID_PATH)

if __name__ == "__main__":
    download_dataset()
    print("[Download] Finished.")