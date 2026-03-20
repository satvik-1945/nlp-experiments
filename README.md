# 🔍 Word2Vec CBOW: Real vs. Fake Embedding Implementations

This repository demonstrates and debunks a common myth in the NLP community:  
> "CBOW with One-Hot input vectors behaves just like real Word2Vec."

We implement and compare:
1. 🧠 **Fake CBOW** — using concatenated one-hot vectors as input (a widespread misunderstanding)
2. 💡 **Real CBOW** — using an embedding lookup table that is learned during training (true Word2Vec-style)

We also visualize the learned word clusters and test analogy completion (e.g. *king − queen + woman ≈ man*) to show why proper embeddings matter.

---

## ⚡ Quick Setup

```bash
# 1. Clone and enter the project
cd nlp-experiments

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train both models
cd src/word2vec_real_vs_fake_implementation
python cbow_using_real_implementation.py
python cbow_using_fake_implementation.py

# 5. Run analogy tests
python test_analogy.py

# 6. Visualize clusters (opens plot window)
python graph_visualization/real_cbow_implementation.py
python graph_visualization/fake_cbow_implementation.py
```

---

## 📁 Project Structure

```
nlp-experiments/
├── requirements.txt
├── README.md
└── src/word2vec_real_vs_fake_implementation/
    ├── corpus.txt                    # ~5000-word essay corpus
    ├── cbow_using_real_implementation.py
    ├── cbow_using_fake_implementation.py
    ├── test_analogy.py               # king - queen + woman = ?
    ├── cbow_models/
    │   ├── EmbeddingCBOW.py          # Real CBOW (embedding lookup)
    │   └── OneHotCBOW.py             # Fake CBOW (one-hot concat)
    ├── saved_embedding_models/       # Trained models (.pkl)
    └── graph_visualization/
        ├── real_cbow_implementation.py
        ├── fake_cbow_implementation.py
        ├── real_cbow_plot.png
        └── fake_cbow_plot.png
```

---

## 📁 Dataset

A **~5000-word essay** (`corpus.txt`) with natural sentences designed for analogy testing:

- **Royalty**: king, queen, prince, princess, emperor, empress, duke, duchess, actor, actress
- **Family**: man, woman, boy, girl, father, mother, brother, sister, uncle, aunt
- **Fruits**: apple, banana, mango, orange, grape, pineapple, cherry, strawberry, peach, pear
- **Countries**: France, Japan, India, China, Germany, Italy, Spain, England, Russia
- **Music genres**: jazz, rock, pop, classical, techno, blues

Related words appear in context (e.g. *"The king lived in the kingdom with his queen"*, *"There was a mango in the basket"*).

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
- **Corpus**: `corpus.txt` (~5000 words, ~800 sentences)
- **Optimizer**: Vanilla SGD
- **Loss**: Negative Log-Likelihood (Cross Entropy)
- **Context window**: 2 (4 context words per target)

---

## 📊 Visualization

We reduce learned embeddings to 2D using **t-SNE** and plot **only selected word groups** to reveal clusters:

- Royalty (male/female)
- Family (male/female)
- Fruits, countries, music genres

Run the scripts or view the saved PNGs in `graph_visualization/`.

---

## 🧩 Analogy Testing

Test classic analogies like *king − queen + woman ≈ man*:

