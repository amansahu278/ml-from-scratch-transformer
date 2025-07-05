# ml-from-scratch-transformer
# 🧠 Transformer From Scratch

This project is a **from-scratch implementation** of the original [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) paper using **pure PyTorch**. It's part of my ongoing GitHub series: [`ml-from-scratch-xyz`](https://github.com/amansahu278?tab=repositories&q=ml-from-scratch).

> 🔬 Goal: To deeply understand and re-implement the Transformer architecture without relying on high-level libraries like HuggingFace or PyTorch Lightning.

---

## 🚀 Features

- Encoder & Decoder architecture
- Multi-Head Attention with masking
- Scaled Dot-Product Attention
- Sinusoidal positional encodings
- Position-wise Feedforward Networks
- Layer Normalization
- Autoregressive `generate()` method
- Optional support for returning:
  - `attention_probs`
  - `past_key_values`

---

## 🏗️ Project Structure

```
.
├── transformer.py         # Main model class
├── encoder.py             # Encoder & EncoderLayer
├── decoder.py             # Decoder & DecoderLayer
├── attention.py           # MHA & Scaled Dot Product
├── positional_encoding.py # Sinusoidal encoding
├── utils.py               # Position-wise FFN, Linear, LayerNorm
└── README.md
```

---

## 📝 Usage

1. **Install dependencies**

   Requires: `torch`, `datasets` (for the demo script)

   ```bash
   pip install torch datasets
   ```

2. **Run the training script**

   The included `main.py` demonstrates training the Transformer on a character-level English→French transliteration task using the HuggingFace `opus_books` dataset:

   ```bash
   python main.py
   ```

   You should see training and validation loss printed for each epoch, and a sample generation at the end.

---

## 📚 Learning Notes

What I learned while implementing this Transformer from scratch:

- How multi-head attention and masking work under the hood, turns out i had a completely flawed understanding of MHA
- How positional encodings are implemented
- How to implement autoregressive generation, especially with managing past key values

---

## 🤔 Why From Scratch?

- To understand how Transformers work (doing is learning!)
- For educational and research purposes

---

## 📖 References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Part of the [`ml-from-scratch-xyz`](https://github.com/amansahu278?tab=repositories&q=ml-from-scratch) series.**
