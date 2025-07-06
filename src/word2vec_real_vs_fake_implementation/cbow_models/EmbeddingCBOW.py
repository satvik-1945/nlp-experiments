import numpy as np

class EmbeddingCBOW:
    def __init__(self, vocab_size, embed_dim=10, lr=0.05):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr

        self.embedding_matrix = np.random.randn(vocab_size, embed_dim) * 0.01  
        self.linear_weight = np.random.randn(embed_dim, vocab_size) * 0.01      
        self.linear_bias = np.zeros(vocab_size)                                

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def forward(self, context_idxs):
        embeds = self.embedding_matrix[context_idxs]
        mean_embed = embeds.mean(axis=0)

        logits = np.dot(mean_embed, self.linear_weight) + self.linear_bias

        probs = self.softmax(logits)
        return embeds, mean_embed, logits, probs
    
    def train(self, train_data, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0

            for context_idxs, target_idx in train_data:
                embeds, mean_embed, logits, probs = self.forward(context_idxs)

                loss = -np.log(probs[target_idx] + 1e-9)
                total_loss += loss

                d_logits = probs.copy()
                d_logits[target_idx] -= 1

                dW = np.outer(mean_embed, d_logits)           
                db = d_logits                                 

                d_embed = np.dot(self.linear_weight, d_logits) / len(context_idxs)  

                self.linear_weight -= self.lr * dW
                self.linear_bias -= self.lr * db

                for idx in context_idxs:
                    self.embedding_matrix[idx] -= self.lr * d_embed

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss / len(train_data):.4f}")

    
    def get_embedding_matrix(self):
        return self.embedding_matrix
