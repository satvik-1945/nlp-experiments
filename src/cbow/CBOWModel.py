import numpy as np

class CBOWModel:
    def __init__(self, vocab_size, embedding_dim):
       self.vocab_size = vocab_size
       self.embedding_dim = embedding_dim
       self.W1 = np.random.rand(vocab_size, embedding_dim) 
       self.W2 = np.random.rand(embedding_dim, vocab_size)

    def one_hot(self, word_index):
        one_hot_vector = np.zeros(self.vocab_size)
        one_hot_vector[word_index] = 1
        return one_hot_vector

    def forward(self, context_indices):
        x = np.mean([self.one_hot(idx) for idx in context_indices], axis=0)
        h = np.dot(x, self.W1)  # Hidden layer
        u = np.dot(h, self.W2)  # Output layer
        y_pred = self.softmax(u)
        return y_pred, h, x
    
    def softmax(self, u):
        exp_u = np.exp(u - np.max(u))
        return exp_u / np.sum(exp_u)

    def train(self, data, word_to_idx, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            loss = 0
            for context, target in data:
                context_indices = [word_to_idx[word] for word in context]
                target_index = word_to_idx[target]

                y_pred, h, x = self.forward(context_indices)

                e = y_pred
                e[target_index] -= 1  # y_pred - y_true

                dW2 = np.outer(h, e)
                dW1 = np.outer(x, np.dot(self.W2, e))

                self.W1 -= learning_rate * dW1
                self.W2 -= learning_rate * dW2

                loss += -np.log(y_pred[target_index] + 1e-9)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

                