# 🔍 Word2Vec CBOW: Real vs. Fake Embedding Implementations

This repository demonstrates and debunks a common myth in the NLP community:  
> "CBOW with One-Hot input vectors behaves just like real Word2Vec."

We implement and compare:
1. 🧠 **Fake CBOW** — using concatenated one-hot vectors as input (a widespread misunderstanding)
2. 💡 **Real CBOW** — using an embedding lookup table that is learned during training (true Word2Vec-style)

We also visualize the learned word clusters to show why proper embeddings are critical for meaningful vector space representations.

---

## 📁 Dataset

A simple, naive dataset consisting of 3 semantic clusters:
- 🟥 Fruits: `apple`, `banana`, `mango`, `orange`, ...
- 🟦 Countries: `india`, `china`, `france`, ...
- 🟩 Music Genres: `jazz`, `rock`, `pop`, `classical`, ...

Stop words have been removed for clarity and stronger signal during training.

---

## 🧪 CBOW Variants

### ✅ Fake CBOW (One-Hot Concatenation)

- Context words are converted to one-hot vectors
- Concatenated into a large sparse input layer
- Connected to a hidden layer → output vocab distribution
- Very inefficient and **not representative of Word2Vec**

### ✅ Real CBOW (Embedding Lookup)

- Context word indices are used to look up dense vectors
- Embeddings are averaged and passed through a linear layer
- Softmax + Cross-Entropy used for prediction
- Embedding matrix gets updated via backpropagation

---

## 🔁 Training

Both models are trained using:
- Optimizer: Vanilla SGD
- Loss: Negative Log-Likelihood (Cross Entropy)
- Context window: 2

---

## 📊 Visualization

We reduce the learned embeddings to 2D using **t-SNE** or **PCA**, and plot them using Matplotlib.
