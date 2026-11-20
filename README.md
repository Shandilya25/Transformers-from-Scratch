# Transformer From Scratch

Building the original Transformer from *Attention Is All You Need* in PyTorch, piece by piece, and training it to **translate English to Italian**.

## The big idea

A Transformer has two halves:

- **Encoder**: reads the English sentence and builds an understanding of it.
- **Decoder**: writes the Italian sentence one word at a time, looking at the encoder's output and at the words it has already written.

```
English sentence -> tokens -> Encoder --+
                                        v
     [SOS] + Italian so far -> Decoder -> next Italian word
```

## Files

| File | What's in it |
|---|---|
| `config.py` | All the settings in one place: batch size, epochs, learning rate, `seq_len`, languages, where to save weights. |
| `dataset.py` | `BilingualDataset`: turns one sentence pair into fixed-size tensors and masks. Also `casual_mask`, which stops the decoder from looking at future words. |
| `model.py` | The building blocks of the model (see below). |
| `training.py` | Loads the data, builds the tokenizers, and contains the training loop and greedy decoding. |

## Building blocks (`model.py`)

- **Input Embedding**: turns each token ID into a vector.
- **Positional Encoding**: adds sine/cosine patterns so the model knows the order of words.
- **Layer Normalization**: keeps the numbers in a stable range.
- **Feed-Forward Network**: two linear layers with ReLU in between.
- **Multi-Head Attention**: lets each word look at the other words.
- **Residual Connection**: adds a layer's input back to its output (with layer norm before the layer), which helps training.
- **Encoder / Decoder blocks**: the pieces above stacked together.
- **Final Linear Layer**: turns the decoder output into a score for every word in the vocabulary.

## What one training example looks like (`dataset.py`)

Every sentence is padded to the same length, `seq_len`:

| Tensor | Example |
|---|---|
| `encoder_input` | `[SOS] Hello world [EOS] [PAD] [PAD]` |
| `decoder_input` | `[SOS] Ciao mondo [PAD] [PAD] [PAD]` |
| `label` | `Ciao mondo [EOS] [PAD] [PAD] [PAD]` |

The label is the decoder input shifted by one, so at every position the model learns to **predict the next word**.

Special tokens:

- `[SOS]`: start of sentence
- `[EOS]`: end of sentence
- `[PAD]`: filler that makes every sentence the same length
- `[UNK]`: any word not in the vocabulary

## Data and tokenizers (`training.py`)

- Dataset: [`opus_books`](https://huggingface.co/datasets/opus_books) from HuggingFace (English–Italian).
- One word-level tokenizer per language, trained on the data and saved as `tokenizer_en.json` / `tokenizer_it.json`.
- Split: 90% training, 10% validation.

## Status

Work in progress.

- [x] Config
- [x] Dataset and masks
- [x] Tokenizers and data loading
- [ ] Model: building blocks written, still fixing bugs and adding `build_transformer`
- [ ] Training loop and validation: written, still fixing bugs
- [ ] First full training run

## Tech stack

- Python
- PyTorch
- HuggingFace `datasets` and `tokenizers`
- TensorBoard (training logs)
- tqdm (progress bars)

## Reference

Vaswani et al., *Attention Is All You Need* (2017). [Read the paper](https://arxiv.org/abs/1706.03762)
