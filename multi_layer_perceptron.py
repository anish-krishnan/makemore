import string
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.contrib import itertools


class MultiLayerPerceptron:
    SPECIAL_TOKEN = "."
    EMBEDDING_DIM = 10
    HIDDEN_LAYER_SIZE = 200
    MINI_BATCH_SIZE = 32

    def __init__(self, context_length=4):
        self.context_length = context_length
        alphabet = list(self.SPECIAL_TOKEN + string.ascii_lowercase)
        self.K = len(alphabet)
        self.char_to_int = {char: idx for idx, char in enumerate(alphabet)}
        self.int_to_char = {idx: char for idx, char in enumerate(alphabet)}
        self.embedding = torch.randn((self.K, self.EMBEDDING_DIM))
        generator = torch.Generator().manual_seed(217483647)
        self.W1 = torch.randn(
            (
                self.context_length * self.EMBEDDING_DIM,
                self.HIDDEN_LAYER_SIZE,
            ),
            generator=generator,
        )
        self.b1 = torch.randn(self.HIDDEN_LAYER_SIZE, generator=generator)
        self.W2 = (
            torch.randn((self.HIDDEN_LAYER_SIZE, self.K), generator=generator) * 0.01
        )
        self.b2 = torch.zeros(self.K)

        self.parameters = [self.embedding, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

    def get_context_and_targets_from_words(self, words):
        for word in words:
            padded_word = (
                self.SPECIAL_TOKEN * self.context_length + word + self.SPECIAL_TOKEN
            )
            padded_word = [self.char_to_int[c] for c in padded_word]
            for i in range(len(padded_word) - self.context_length):
                context = padded_word[i : i + self.context_length]
                target = padded_word[i + self.context_length]
                yield context, target

    def train(self, words):
        unzip = lambda x: zip(*x)
        contexts, targets = unzip(self.get_context_and_targets_from_words(words))
        contexts = torch.tensor(contexts)
        targets = torch.tensor(targets)
        N = contexts.shape[0]

        def forward():
            # N x (BLOCK_SIZE * EMBEDDING_DIM)
            mini_batch = torch.randint(0, N, (self.MINI_BATCH_SIZE,))
            mini_contexts = contexts[mini_batch]
            mini_targets = targets[mini_batch]
            viewed_contexts = self.embedding[mini_contexts].view(
                (-1, self.context_length * self.EMBEDDING_DIM)
            )
            h = torch.tanh(viewed_contexts @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            loss = F.cross_entropy(logits, mini_targets)
            return loss, logits

        def backward(loss, learning_rate):
            for p in self.parameters:
                p.grad = None
            loss.backward()
            for p in self.parameters:
                p.data += -learning_rate * p.grad

        learning_rates = [0.1, 0.01]
        num_training_iterations_per_learning_rate = 100_000
        for learning_rate, _ in itertools.product(
            learning_rates, range(num_training_iterations_per_learning_rate)
        ):
            loss, _ = forward()
            backward(loss, learning_rate)

    def generate_word(self):
        word = ""
        # Start with just [SPECIAL_TOKEN]s
        current_context = [self.char_to_int[self.SPECIAL_TOKEN]] * self.context_length
        current_char_idx = self.SPECIAL_TOKEN

        def get_next_char_idx(context):
            embedding = self.embedding[torch.tensor([context])]
            h = torch.tanh(embedding.view(1, -1) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            distribution = F.softmax(logits, dim=1)
            return torch.multinomial(
                distribution, num_samples=1, replacement=True
            ).item()

        while (
            current_char_idx := get_next_char_idx(current_context)
        ) != self.char_to_int[self.SPECIAL_TOKEN]:

            word += self.int_to_char[current_char_idx]
            current_context = current_context[1:] + [current_char_idx]

        return word

    def visualize(self):
        plt.scatter(self.embedding[:, 0].data, self.embedding[:, 1].data, s=100)
        for idx in range(self.K):
            char = self.int_to_char[idx]
            plt.text(
                self.embedding[idx, 0],
                self.embedding[idx, 1],
                char,
                ha="center",
                va="center",
                color="white",
            )
        plt.show()


words = open("names.txt", "r").read().splitlines()
mlp = MultiLayerPerceptron()
mlp.train(words)

for _ in range(10):
    print(mlp.generate_word())

mlp.visualize()
