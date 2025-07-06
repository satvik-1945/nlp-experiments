from cbow_models.OneHotCBOW import OneHotCBOW
import pickle
import os

sentences = [
    "apple banana mango pineapple",              # 🟥 fruits
    "china india france japan germany",          # 🟦 countries
    "jazz classical rock techno pop",            # 🟩 genres
    "banana mango apple",                        # 🟥 fruits
    "france india japan",                        # 🟦 countries
    "rock jazz pop", 
]

def tokenize_sentences(sentences):
    tokens = []
    for sentence in sentences:
        tokens.append(sentence.lower().split())
    return tokens

tokenized_sentences = tokenize_sentences(sentences)
# Build vocab

def build_vocab(tokenized_sentences):
    vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab, word2idx, idx2word

vocab, word2idx, idx2word = build_vocab(tokenized_sentences)
vocab_size = len(vocab)
print(f"Vocab Size: {vocab_size}")
def generate_cbow_training_data(tokenized_sentences, word2idx, window_size=2):
    training_data = []
    # The desired constant context size
    # This is (window_size on left) + (window_size on right)
    fixed_context_size = 2 * window_size 

    for sentence in tokenized_sentences:
        for idx in range(len(sentence)):
            target_word_idx = word2idx[sentence[idx]]
            current_context = []

            for i in range(1, window_size + 1):
                left_idx = idx - i
                if left_idx >= 0:
                    current_context.append(word2idx[sentence[left_idx]])
            for i in range(1, window_size + 1):
                right_idx = idx + i
                if right_idx < len(sentence):
                    current_context.append(word2idx[sentence[right_idx]])
            if len(current_context) == fixed_context_size:
                training_data.append((current_context, target_word_idx))
    return training_data

train_data = generate_cbow_training_data(tokenized_sentences, word2idx)
print(f'training data sample: {train_data[:5]}')
# exit()

context_size = 4  
print(len(vocab)*context_size)
embed_dim = 10

model = OneHotCBOW(vocab_size, context_size, embed_dim=embed_dim, lr=0.05)
print("starting training...")
model.train(train_data, epochs=1000)
print("training complete. saving model...")
# print(os.getcwd())
scriptdir = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(scriptdir,"saved_embedding_models/cbow_embeddings.pkl")

with open(output_file_path, "wb") as f:
    pickle.dump({
        "embeddings": model.get_embedding_matrix(),
        "word2idx": word2idx,
        "idx2word": idx2word
    }, f)

print(f"Model saved to: {output_file_path}")