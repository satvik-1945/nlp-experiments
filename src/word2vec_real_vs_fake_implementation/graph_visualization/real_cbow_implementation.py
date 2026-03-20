import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.join(script_dir, "../saved_embedding_models/cbow_embeddings_real.pkl")
with open(abspath, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
word2idx = data["word2idx"]
idx2word = data["idx2word"]


def plot_selected_words(embeddings, word2idx, idx2word, selected_groups, method="tsne"):
    """Plot only selected word groups to reduce clutter and reveal clusters."""
    # Collect selected words and their indices
    selected_indices = []
    word_to_group = {}
    for group_name, (color, word_list) in selected_groups.items():
        for word in word_list:
            if word in word2idx:
                selected_indices.append(word2idx[word])
                word_to_group[word2idx[word]] = (group_name, color)

    if not selected_indices:
        print("No selected words found in vocab.")
        return

    # Extract embeddings for selected words only
    selected_embeddings = embeddings[selected_indices]

    # Reduce dimensions (fit on selected subset for cleaner clustering)
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        perplexity = min(30, len(selected_indices) - 1)
        perplexity = max(5, perplexity)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced = reducer.fit_transform(selected_embeddings)

    plt.figure(figsize=(10, 8))
    for group_name, (color, _) in selected_groups.items():
        group_indices = [i for i in selected_indices if word_to_group[i][0] == group_name]
        if group_indices:
            pts = [reduced[selected_indices.index(i)] for i in group_indices]
            xs, ys = zip(*pts)
            plt.scatter(xs, ys, c=color, s=80, alpha=0.8, label=group_name)
    for idx, (x, y) in zip(selected_indices, reduced):
        word = idx2word[idx]
        plt.annotate(word, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)

    plt.legend(loc="upper left", fontsize=8)
    plt.title(f"Word Embeddings — Selected Clusters ({method.upper()})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

selected_groups = {
    "royalty_male":   ("#e74c3c", ["king", "prince", "emperor", "duke", "lord", "actor"]),
    "royalty_female": ("#3498db", ["queen", "princess", "empress", "duchess", "lady", "actress"]),
    "family_male":    ("#e67e22", ["father", "brother", "uncle", "boy", "man"]),
    "family_female":  ("#2ecc71", ["mother", "sister", "aunt", "girl", "woman"]),
    "fruit":          ("#9b59b6", ["apple", "banana", "mango", "orange", "pineapple", "grape", "cherry", "strawberry", "peach", "pear"]),
    "country":        ("#8b4513", ["india", "china", "france", "germany", "japan", "italy", "spain", "england", "russia"]),
    "genre":          ("#1abc9c", ["jazz", "rock", "pop", "classical", "techno", "blues"]),
}

plot_selected_words(embeddings, word2idx, idx2word, selected_groups)