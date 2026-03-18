# ICT303 Assignment 2 — English to Indonesian Neural Machine Translation

A sequence-to-sequence LSTM encoder-decoder model for translating English text into Bahasa Indonesia, trained on the [opus-Indonesian-to-English](https://huggingface.co/datasets/kaitchup/opus-Indonesian-to-English) dataset (~1 million sentence pairs) using a cloud instance RTX 4090.

**[View the Jupyter Notebook](./src/ICT303%20-%20Assignment%202.ipynb)**

---

## Project Structure

```
src/
├── config.py                  # Paths, vocab sizes, and special token constants
├── setup.sh                   # Full data pipeline (download → clean → vocab → encode)
├── download_dataset.py        # Downloads train/validation splits from HuggingFace
├── clean_dataset.py           # Filters noisy/invalid sentence pairs via regex
├── generate_vocab.py          # Builds token vocabularies for both languages
├── encode_dataset.py          # Encodes cleaned data to token index sequences
├── train_model.py             # Training entry point (1-layer, bs=128 config)
├── train_seq2seq_lstm.py      # Alternative training script (2-layer variant)
├── test_preprocessing.py      # Unit tests for cleaning/tokenization logic
├── test_dataset.py            # Tests the PyTorch Dataset/DataLoader pipeline
├── test_trainer.py            # Short training loop sanity check
├── test_valid.py              # Evaluates model on validation set (BLEU score)
├── data/
│   └── vocab.json             # Token→index maps for both languages (order-sensitive!)
├── assets/
│   ├── lstm_cell_diagram.png  # LSTM cell diagram
│   └── model_diagram.png      # Encoder-decoder architecture diagram
├── dataset/
│   └── ENGtoID.py             # PyTorch Dataset with padding collate_fn
├── models/
│   └── seq2seq_lstm.py        # LSTMEncoder, LSTMDecoder, Seq2SeqLSTM
├── trainers/
│   └── trainer.py             # Training/validation loop with TensorBoard logging
└── utils/
    ├── preprocessing.py       # Tokenization, cleaning, vocab I/O
    └── coder.py               # Token ↔ index encoding/decoding
```

---

## Setup

> **Warning:** Running `setup.sh` regenerates `vocab.json`. Token order in this file is **not sorted** and varies each run. Any previously trained model weights are **permanently coupled** to their original vocab — regenerating it will break saved models.

```bash
cd src/
bash setup.sh
```

This runs four steps in sequence:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `download_dataset.py` | Downloads train & validation splits to `data/` |
| 2 | `clean_dataset.py` | Removes malformed sentence pairs using regex validation |
| 3 | `generate_vocab.py` | Tokenizes cleaned data; saves `vocab.json` |
| 4 | `encode_dataset.py` | Converts tokens to integer indices; saves encoded datasets |

---

## Data Pipeline

### Format

Raw sentence pairs use the delimiter format:
```
<Indonesian sentence> ###> <English sentence>
```

### Cleaning

`clean_dataset.py` applies regex-based filtering to remove entries with invalid formatting, URLs, special symbols, or broken grammar. The filter is intentionally conservative — it removes a small fraction of the ~1M samples but reliably eliminates the most problematic entries.

### Vocabulary

`generate_vocab.py` tokenizes every entry in the **training set only** and builds separate token sets per language. Three special tokens are manually prepended at fixed indices:

| Token | Index | Purpose |
|-------|-------|---------|
| `<PAD>` | 0 | Sequence padding (ignored by loss function) |
| `<EOS>` | 1 | End of sequence marker |
| `<UNK>` | 3 | Unknown/out-of-vocabulary token |

After processing: **~136,819 Indonesian tokens**, **~115,737 English tokens**.

> **Note:** The validation set contains some tokens absent from the training vocabulary. These map to `<UNK>` at inference time. This is a known limitation of the current train/validation split.

### Encoding

`encode_dataset.py` converts each sentence pair into two lists of integer indices (one per language), with `<EOS>` appended to every sequence. Example encoded entry:

```python
{'eng': [31852, 95738, 87251, 50861, 16639, 12296, 36826, 113736, 36153, 1],
 'id':  [52280, 98559, 81350, 132670, 132442, 43015, 1]}
```

---

## Model

A standard **Seq2Seq LSTM Encoder-Decoder** architecture (~200M parameters).

### Architecture

1. **Encoder** (`LSTMEncoder`): Embeds English tokens into a 512-dimensional learned vector space, then processes the full sequence through a single-layer LSTM. Outputs the final hidden state and cell state as a context vector.
2. **Decoder** (`LSTMDecoder`): Receives the context vector and auto-regressively generates Indonesian tokens one at a time. Each step embeds the previous token (or teacher-forced target) and runs it through the LSTM to produce logits over the full Indonesian vocabulary.

### Configuration

```python
hidden_state_size    = 512
layers               = 1
max_inference_tokens = 128
vocab_eng            = 115_737   # encoder embedding size
vocab_id             = 136_819   # decoder embedding size + output classes
```

### Design Choices

- **Adam optimizer** — chosen over SGD for its moving average of gradients, which provides stable learning over ~1M-sample epochs without constant manual learning rate adjustment.
- **CrossEntropyLoss** with `ignore_index=PAD_IDX (0)` — ensures padding tokens do not contribute to the loss.
- **Single layer at 512 hidden units** — preferred over 2 layers at 256 to maximise representational capacity given the dataset size.

---

## Training

```bash
python3 train_model.py
```

Training used the following hyperparameter schedule, applied sequentially to the same model checkpoint:

| Epochs | Batch Size | Learning Rate | Teacher Forcing |
|--------|-----------|---------------|-----------------|
| 5 | 128 | 0.001 | 1.0 |
| 5 | 128 | 0.0005 | 1.0 |
| 5 | 128 | 0.0001 | 1.0 |
| 5 | 128 | 0.00005 | 1.0 |
| 5 | 64 | 0.00005 | 0.5 |
| 5 | 64 | 0.0001 | 1.0 |

**Rationale:** The first 5 epochs at lr=0.001 establish basic token mappings (~65%+ training accuracy). Each subsequent lower-lr phase fine-tunes weights more precisely. The 50% teacher forcing phase (group 5) teaches the model to predict from its own output rather than always relying on ground-truth tokens — this caused a visible spike in loss as the model had to learn sentence termination without length guidance.

Training was performed on a rented **RTX 4090** via [vast.ai](https://vast.ai).

### Teacher Forcing

During training, the decoder receives the actual target token at each timestep rather than its own previous prediction. This stabilises early training by preventing error accumulation, and avoids wasting epochs teaching the model when to emit `<EOS>` across 130k+ possible tokens. Teacher forcing is disabled during inference.

### Accuracy Metric

`.argmax()` is applied across output logits, then compared against the target sequence excluding padding positions. This gives a token-match accuracy percentage. It has known limitations — it gives no credit to valid synonyms or alternative phrasings. A BLEU score would be more appropriate but requires richer reference data.

### Monitoring

Metrics are logged to TensorBoard and saved as plots in `./checkpoints/{model_name}/`. To view in a notebook:

```python
%load_ext tensorboard
%tensorboard
```

---

## Evaluation

```bash
python3 test_valid.py
```

Runs the model over the validation set and reports average BLEU score.

### Validation Observations

Validation accuracy remained nearly flat across epochs after an initial jump to ~53% after epoch 1. Likely causes: the validation set is very small (~2,000 samples vs ~1M training), the accuracy metric ignores synonyms, and out-of-vocabulary tokens map to `<UNK>`.

### Example Translations (trained model)

| English Input | Model Output | Notes |
|---------------|--------------|-------|
| Are you my friend? | teman ku ? | Partially correct — missing subject |
| he is over there. | dia di sana . . . | Correct translation |
| I can speak English | bisa bahasa Inggris . | Missing subject pronoun |
| Please take my money. | mengambil uang saya . | Slightly incorrect |
| There are lots of ways to say things. | Banyak hal yang harus dikatakan . | Meaning misinterpreted |
| that car is red | merah mobil itu . | Word order reversed |

### Error Analysis

Logging incorrect predictions across the validation set showed the most common mispredictions were `.`, `<EOS>`, `,`, and high-frequency words (`Aku`, `akan`, `tidak`, `kau`). This points to difficulty with punctuation placement, sentence termination, and context-dependent common words — likely amplified by the limitations of the simple tokenization scheme.

---

## Known Limitations & Future Work

**Tokenization:** Simple whitespace/punctuation splitting. Byte pair encoding (BPE) would handle morphology and rare words better, and may improve grammatical output.

**Vocab coupling:** Model weights are permanently tied to the original `vocab.json` token order. There is no mechanism to remap or migrate a trained model after vocab regeneration.

**Training duration:** 30 epochs is insufficient for a 130k-class sequence prediction task. 50+ epochs at progressively lower learning rates would likely yield significant gains — large accuracy jumps were observed at the start of each epoch, suggesting more epochs matter.

**Validation split imbalance:** ~2,000 validation samples vs ~1M training (~0.2%) is far below a typical 10–15% split. Supplementing with a portion of training data would give more reliable generalisation metrics.

**Model architecture:** A Transformer with attention would better handle long-range source-target dependencies. A GRU would be a simpler upgrade at lower compute cost.

---

## Dependencies

```bash
pip install datasets torch tqdm nltk tensorboard
```

| Library | Use |
|---------|-----|
| `datasets` (HuggingFace) | Dataset download and Arrow-format storage |
| `torch` | Model, training, and inference |
| `tqdm` | Progress bars |
| `nltk` | BLEU score (`sentence_bleu`) |
| `tensorboard` | Training metrics visualisation |

---

## References

- [LSTM — Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [torch.nn.LSTM — PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [torch.nn.Embedding — PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [Teacher Forcing — Wikipedia](https://en.wikipedia.org/wiki/Teacher_forcing)
- [Gradient Descent — Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [BLEU Score — Wikipedia](https://en.wikipedia.org/wiki/BLEU)
- [Dataset: kaitchup/opus-Indonesian-to-English](https://huggingface.co/datasets/kaitchup/opus-Indonesian-to-English)