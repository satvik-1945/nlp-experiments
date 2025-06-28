import nltk
import os

#download nltk data if not already present
# Uncomment the following lines if you want to download the NLTK data programmatically
# nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
# nltk.data.path.append(nltk_data_dir)

# nltk.download('gutenberg', download_dir=nltk_data_dir)
# nltk.download('punkt', download_dir=nltk_data_dir)
# nltk.download('wordnet', download_dir=nltk_data_dir)
# nltk.download('omw-1.4', download_dir=nltk_data_dir)

from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize

text = gutenberg.raw("shakespeare-hamlet.txt")
tokens = word_tokenize(text.lower(), language='english', preserve_line=True)

# print(tokens[:100])

#  making sentence
from nltk.util import bigrams

sentences = text.split('\n')
tokenized_sentences = [word_tokenize(sentence.lower(), language='english', preserve_line=True) for sentence in sentences if sentence.strip()]
print(tokenized_sentences[:5])

from gensim.models import Word2Vec

model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "shakespeare_word2vec.model")
model.save(model_path)

import numpy as np
embedding_matrix = np.array([model.wv[word] for word in model.wv.index_to_key])
vocab = model.wv.index_to_key

print(f"Embedding matrix shape: {embedding_matrix.shape}")