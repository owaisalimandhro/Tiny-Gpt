# Character-Level GPT / Transformer from Scratch

A **character-level language model built from scratch using PyTorch**, implementing the core components of a GPT-style Transformer.

This project is designed to understand how Transformer-based language models work internally rather than simply using a pre-built Transformer implementation.

The model is trained on the **Tiny Shakespeare** dataset and learns to generate Shakespeare-like text character by character.

---

## 🚀 Features

* Character-level tokenization
* Custom vocabulary creation
* Training/validation dataset split
* Mini-batch training
* Token embeddings
* Positional embeddings
* Causal self-attention
* Multi-head self-attention
* Feed-forward neural network
* Layer normalization
* Residual connections
* Dropout
* Cross-entropy loss
* AdamW optimizer
* Training and validation loss estimation
* Autoregressive text generation
* CUDA support when available

---

## 🧠 Model Architecture

The model follows the general structure of a small GPT-style Transformer:

```text
Input Characters
       ↓
Character Encoding
       ↓
Token Embeddings + Positional Embeddings
       ↓
Transformer Blocks
       │
       ├── LayerNorm
       ├── Multi-Head Self-Attention
       ├── Residual Connection
       │
       ├── LayerNorm
       ├── Feed-Forward Network
       └── Residual Connection
       ↓
Final LayerNorm
       ↓
Language Model Head
       ↓
Vocabulary Logits
       ↓
Next Character Prediction
```

The Transformer blocks use residual connections so that the original representation can be preserved while the attention and feed-forward layers add additional information.

---

## 📂 Project Structure

```text
.
├── train.py
├── tinyshakespeare.txt
└── README.md
```

### `train.py`

Contains the complete model implementation, training loop, evaluation logic, and text generation.

### `tinyshakespeare.txt`

Training dataset containing Shakespeare's works. The dataset is read and converted into a character-level vocabulary.

---

## ⚙️ Requirements

Install Python and PyTorch.

```bash
pip install torch
```

The project uses:

* Python
* PyTorch
* CUDA (optional)

The model automatically uses CUDA when it is available and otherwise falls back to CPU.

---

## 📥 Dataset

The project expects a file named:

```text
tinyshakespeare.txt
```

The complete text is converted into a character vocabulary.

Each unique character receives an integer ID using two mappings:

```python
stoi
itos
```

`stoi` converts characters into integers, while `itos` converts integer IDs back into characters.

The encoded dataset is then split into:

* **90% training data**
* **10% validation data**

---

## 🔢 Training Configuration

The current configuration is:

| Parameter             | Value |
| --------------------- | ----: |
| Batch size            |    32 |
| Block size            |     8 |
| Maximum iterations    |  5000 |
| Evaluation interval   |   500 |
| Evaluation iterations |   200 |
| Learning rate         | 0.001 |
| Embedding dimension   |    32 |

These values are defined near the beginning of the training script.

---

## 🔤 Character-Level Tokenization

Unlike modern large language models that commonly use subword tokenization, this implementation works directly with **individual characters**.

For example:

```text
hello
```

is represented conceptually as:

```text
h → e → l → l → o
```

The model learns to predict the next character based on the characters it has already seen.

---

## 📦 Batches and Context

The `get_batch()` function creates random training examples.

With:

```python
block_size = 8
```

the model receives up to 8 characters as context and attempts to predict the next character at every position.

For example:

```text
Input:  H e l l o ...
Target: e l l o ...
```

Each target is shifted one character relative to the input.

---

# 🧩 Transformer Components

## 1. Self-Attention Head

The `Head` class implements a single causal self-attention head.

It creates three projections:

```python
self.key
self.query
self.value
```

These are used to determine:

* what information a token is looking for (`Query`)
* what information a token offers (`Key`)
* what information should actually be passed forward (`Value`)

The attention weights are calculated using:

```python
wei = q @ k.transpose(-2,-1) * C**-0.5
```

A causal mask prevents a token from looking at future tokens.

This is important for language generation because when predicting the next character, the model should not have access to future characters.

---

## 2. Multi-Head Attention

Multiple attention heads operate independently:

```python
self.heads = nn.ModuleList(
    [Head(head_size) for _ in range(num_heads)]
)
```

Their outputs are concatenated and projected back into the embedding dimension.

Multiple heads allow the model to learn different relationships between tokens simultaneously.

---

## 3. Feed-Forward Network

After attention, the representation passes through a feed-forward neural network:

```text
Embedding
   ↓
Linear
   ↓
ReLU
   ↓
Linear
   ↓
Dropout
```

The hidden dimension is expanded to `4 × n_emb_d` before being projected back to the original embedding size.

---

## 4. Layer Normalization

Each Transformer block uses LayerNorm before the attention and feed-forward operations:

```python
self.ln1 = nn.LayerNorm(n_emb_d)
self.ln2 = nn.LayerNorm(n_emb_d)
```

This helps keep the activations stable during training.

---

## 5. Residual Connections

The Transformer block uses:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

Instead of replacing the original representation, the output of attention/feed-forward processing is added back to the original `x`.

This allows information from earlier stages to flow through the network more easily.

---

## 6. Token and Positional Embeddings

The model uses two embeddings:

```python
self.token_embedding_table
self.position_embedding_table
```

Token embeddings represent **what the character/token is**, while positional embeddings provide information about **where it occurs in the sequence**.

They are added together before being passed into the Transformer blocks.

---

# 🎯 Language Model Head

After passing through the Transformer blocks and final LayerNorm, the representation is passed through:

```python
self.lm_head = nn.Linear(n_emb_d, vocab_size)
```

This converts the model's internal representation into a score for every character in the vocabulary.

The resulting logits are used with cross-entropy loss to measure how well the model predicts the correct next character.

---

# 📉 Training

The model uses the **AdamW optimizer**:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

During each training iteration:

1. A batch is sampled.
2. The model generates logits.
3. Cross-entropy loss is calculated.
4. Gradients are calculated using backpropagation.
5. The optimizer updates the model parameters.

The training loop runs for `5000` iterations in the current configuration.

---

# 📊 Loss Evaluation

The `estimate_loss()` function evaluates both:

```text
train loss
validation loss
```

It temporarily switches the model into evaluation mode, calculates the loss across multiple batches, averages the results, and then returns the model to training mode.

During training, the losses are printed every `500` iterations:

```text
step 0: train loss ..., val loss ...
step 500: train loss ..., val loss ...
...
```

This makes it possible to monitor whether the model is learning and whether it is generalizing to unseen validation data.

---

# ✍️ Text Generation

After training, the model starts with an empty context:

```python
context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)
```

It then generates characters one at a time.

The generation process is:

```text
Current context
      ↓
Transformer
      ↓
Next-character probabilities
      ↓
Sample next character
      ↓
Add character to context
      ↓
Repeat
```

The current script generates up to **500 new characters**.

The `generate()` function uses softmax probabilities and samples the next token using `torch.multinomial()`.

---

# ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install PyTorch:

```bash
pip install torch
```

Make sure the dataset exists:

```text
tinyshakespeare.txt
```

Then run:

```bash
python train.py
```

The model will train and periodically print training and validation loss.

After training, generated Shakespeare-like text will be printed in the terminal.

---

# 🛠️ Current Limitations

This is an educational implementation rather than a production-scale GPT model.

Current limitations include:

* Very small embedding dimension
* Short context window
* Character-level tokenization
* Small training dataset
* Relatively small number of training iterations
* No pretrained weights
* No checkpoint saving/loading
* No advanced tokenizer
* No large-scale distributed training

The purpose of the project is primarily to understand the **internal mechanics of Transformers and GPT-style language models**.

---

# 📚 Concepts Demonstrated

This project provides hands-on implementation of:

* Neural language modeling
* Character-level tokenization
* Embeddings
* Positional embeddings
* Query, Key, and Value
* Self-attention
* Causal masking
* Multi-head attention
* Feed-forward networks
* Layer normalization
* Residual connections
* Dropout
* Cross-entropy loss
* Backpropagation
* AdamW optimization
* Training vs. validation loss
* Autoregressive generation

---

# 🎓 Learning Goal

The main goal of this project is to understand **how a GPT-style language model works internally by implementing its major components rather than treating the Transformer as a black box**.

The implementation follows the fundamental ideas behind decoder-only Transformer language models and provides a small environment for experimenting with attention, embeddings, optimization, and text generation.

---

## 📌 Acknowledgment

This project was developed as a learning implementation inspired by the educational work of **Andrej Karpathy**, particularly the process of building a GPT-style language model from scratch.

---

## 📄 License

This project is intended for educational and learning purposes.
