# ml-from-scratch-transformer
# 🧠 Transformer From Scratch

This project is a **from-scratch implementation** of the original [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) paper using **pure PyTorch**. It's part of my ongoing GitHub series: [`ml-from-scratch-xyz`](https://github.com/amansahu278?tab=repositories&q=ml-from-scratch).

> 🔬 Goal: To understand and re-implement the Transformer architecture without relying on high-level libraries like HuggingFace or PyTorch Lightning.

---

## 📚 Features

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

```bash
.
├── transformer.py         # Main model class
├── encoder.py             # Encoder & EncoderLayer
├── decoder.py             # Decoder &DecoderLayer
├── attention.py           # MHA & Scaled Dot Product
├── positional_encoding.py # Sinusoidal encoding
├── utils.py               # Position-wise FFN, Linear, LayerNorm
├── main.py               # Training loop
└── README.md
