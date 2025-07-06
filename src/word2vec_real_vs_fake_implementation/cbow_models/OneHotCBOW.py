import numpy as np

class OneHotCBOW:

    def __init__(self, vocab_size, context_size, embed_dim=10, lr=0.05):
        self.vocab_size = vocab_size
        self.context_size = context_size  
        self.embed_dim = embed_dim
        self.lr = lr

        input_dim = context_size * vocab_size

        self.W1 = np.random.randn(input_dim, embed_dim) * 0.01  
        self.W2 = np.random.randn(embed_dim, vocab_size) * 0.01 

    def one_hot(self, idx):
        vec = np.zeros(self.vocab_size)
        vec[idx] = 1
        return vec

    def create_input_vector(self, context_idxs):
        return np.concatenate([self.one_hot(i) for i in context_idxs])

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def forward(self, context_idxs):
        x = self.create_input_vector(context_idxs)       
        h = np.dot(x, self.W1)                           
        u = np.dot(h, self.W2)                           
        y_pred = self.softmax(u)                         
        return x, h, u, y_pred

    def train(self, train_data, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for context_idxs, target_idx in train_data:
                x, h, u, y_pred = self.forward(context_idxs)

                # Cross-entropy loss
                loss = -np.log(y_pred[target_idx] + 1e-9)
                total_loss += loss

                # Gradients
                delta = y_pred.copy()
                delta[target_idx] -= 1

                dW2 = np.outer(h, delta)                  # [embed_dim, vocab_size]
                dW1 = np.outer(x, np.dot(self.W2, delta))  # [input_dim, embed_dim]

                self.W2 -= self.lr * dW2
                self.W1 -= self.lr * dW1

            if epoch % 100 == 0:
                avg_loss = total_loss / len(train_data)
                print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    def get_embedding_matrix(self):
        return self.W2.T  # Each row: vector for one word