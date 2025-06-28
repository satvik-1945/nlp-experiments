import os
from gensim.models import Word2Vec

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "shakespeare_word2vec.model")

model = Word2Vec.load(model_path)

# print(model)

vocab = model.wv.index_to_key
# print(vocab[:10])  

print("hamlet" in model.wv)

print(model.wv.most_similar("remember", topn=1))
print(model.wv.similarity("hamlet", "king"))
print(model.wv.similarity("hamlet", "queen"))