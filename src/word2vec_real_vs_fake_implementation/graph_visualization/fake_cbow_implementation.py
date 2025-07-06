import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.join(script_dir, "../saved_embedding_models/cbow_embeddings.pkl")
with open(abspath, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
word2idx = data["word2idx"]
idx2word = data["idx2word"]


def plot_embeddings_2d(embeddings, idx2word, highlight_groups=None, method="tsne"):
    # Reduce dimensions
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=5)
        
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    for i, (x, y) in enumerate(reduced):
        word = idx2word[i]
        color = "black"
        
        if highlight_groups:
            for group_name, word_list in highlight_groups.items():
                if word in word_list:
                    color = highlight_groups[group_name][0]  
        plt.scatter(x, y, c=color)
        plt.text(x + 0.01, y + 0.01, word, fontsize=9)

    plt.title(f"Word Embeddings ({method.upper()}) Visualization")
    plt.grid(True)
    plt.show()

highlight_groups = {
    "fruit":      ["red", ["apple", "banana", "mango", "orange", "pineapple", "fruits", "smoothie", "salad", "fruit"]],
    "country":    ["blue", ["india", "china", "france", "germany", "japan", "countries", "union"]],
    "genre":      ["green", ["jazz", "rock", "pop", "classical", "techno", "music", "clubs"]],
}

plot_embeddings_2d(embeddings, idx2word, highlight_groups)