#!/bin/bash

python3 ./download_dataset.py
python3 ./clean_dataset.py
python3 ./generate_vocab.py
python3 ./encode_dataset.py