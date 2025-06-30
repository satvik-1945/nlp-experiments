import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random
import os

data_dir = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(data_dir, 'nltk_data')
nltk.download('gutenberg', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)
text = nltk.corpus.gutenberg.raw("shakespeare-hamlet.txt")
tokens = word_tokenize(text.lower(), language='english', preserve_line=True)    
tokens = [t for t in tokens if t.isalpha()] 

word_freq = defaultdict(int)
for word in tokens:
    word_freq[word] += 1

vocab = sorted(word_freq, key=word_freq.get,reverse=True)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

# generate traingin data
def generate_training_data(tokens, window_size=2):
    data = []
    for i in range(window_size, len(tokens) - window_size):
        context = [tokens[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        target = tokens[i]
        data.append((context, target))
    return data

window_size = 2 
training_data = generate_training_data(tokens, window_size)
print("Sample (context, target):", training_data[:2])

from CBOWModel import CBOWModel

embedding_dim = 100
cbow = CBOWModel(vocab_size, embedding_dim)
cbow.train(training_data[:10000], word_to_index, epochs=100, learning_rate=0.01)

word_embeddings = cbow.W1