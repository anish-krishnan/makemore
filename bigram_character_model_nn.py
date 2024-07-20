import matplotlib.pyplot as plt
import string
import torch
import torch.nn.functional as F
from tqdm import tqdm


class BigramCharacterModelNN:
    LEARNING_RATE = 50
    REGULARIZATION_RATIO = 0.01
    SPECIAL_TOKEN = "-"
    IMAGE_SIZE = 8
    FONT_SIZE = 5

    def __init__(self, verbose=False):
        self.verbose = verbose
        alphabet = list(self.SPECIAL_TOKEN + string.ascii_lowercase)
        self.K = len(alphabet)
        self.char_to_int = {char: idx for idx, char in enumerate(alphabet)}
        self.int_to_char = {idx: char for idx, char in enumerate(alphabet)}
        generator = torch.Generator().manual_seed(217483647)
        self.weights = torch.randn(
            (self.K, self.K), generator=generator, requires_grad=True
        )

    def get_bigrams_from_words(self, words):
        for word in words:
            word = f"{self.SPECIAL_TOKEN}{word}{self.SPECIAL_TOKEN}"
            for char1, char2 in zip(word, word[1:]):
                yield self.char_to_int[char1], self.char_to_int[char2]

    def forward(self, inputs, targets=None):
        # logits are our log counts (when exponentiated, they are similar to
        # the counts in the classical Bigram character model). We can normalize
        # these to form a valid probability distribution.
        #
        # we're basically 'plucking' out the rows in [self.weights] that
        # correspond to the the letters represented by the one-hot encodings
        # in [inputs].
        N = len(inputs)
        logits = inputs @ self.weights

        # softmax
        counts = logits.exp()
        probabilities = counts / counts.sum(dim=1, keepdim=True)
        loss = None
        if targets is not None:
            y_probabilities = probabilities[torch.arange(N), targets]

            regularization = self.REGULARIZATION_RATIO * (self.weights**2).mean()
            loss = -y_probabilities.log().mean() + regularization
        return probabilities, loss

    def get_loss(self, words):
        unzip = lambda x: zip(*x)
        xs, ys = unzip(self.get_bigrams_from_words(words))
        xs = torch.tensor(xs)
        xs_encoded = F.one_hot(xs, num_classes=self.K).float()
        ys = torch.tensor(ys)

        _, loss = self.forward(xs_encoded, ys)
        return loss

    def train(self, words):
        unzip = lambda x: zip(*x)
        xs, ys = unzip(self.get_bigrams_from_words(words))
        xs = torch.tensor(xs)
        xs_encoded = F.one_hot(xs, num_classes=self.K).float()
        ys = torch.tensor(ys)

        # backwards pass
        def backward_pass(loss):
            self.weights.grad = None
            loss.backward()
            self.weights.data += -self.LEARNING_RATE * self.weights.grad

        for i in tqdm(range(200)):
            _, loss = self.forward(xs_encoded, ys)
            if self.verbose:
                print(f"loss({i}): {loss.item()}")
            backward_pass(loss)

    def generate_word(self):
        word = ""
        current_char_idx = self.char_to_int[self.SPECIAL_TOKEN]

        def get_next_char_idx(current_char_idx):
            current_char_idx_one_hot = F.one_hot(
                torch.tensor(current_char_idx).view(1), num_classes=self.K
            ).float()
            distribution, _ = self.forward(current_char_idx_one_hot)
            return torch.multinomial(
                distribution, num_samples=1, replacement=True
            ).item()

        while (
            current_char_idx := get_next_char_idx(current_char_idx)
        ) != self.char_to_int[self.SPECIAL_TOKEN]:
            word += self.int_to_char[current_char_idx]
        return word

    def visualize(self, image_size=IMAGE_SIZE, font_size=FONT_SIZE):
        weights = self.weights.detach().exp()
        weights = F.one_hot(torch.arange(self.K), num_classes=self.K).float() @ weights
        plt.rcParams.update({"font.size": font_size})
        plt.figure(figsize=(image_size, image_size))
        plt.imshow(weights, cmap="Blues")
        for i in range(self.K):
            for j in range(self.K):
                bigram = self.int_to_char[i] + self.int_to_char[j]
                weight = weights[i, j].item()
                weight = f"{weight:.2f}"
                plt.text(j, i, bigram, ha="center", va="bottom", color="gray")
                plt.text(
                    j,
                    i,
                    weight,
                    ha="center",
                    va="top",
                    color="gray",
                )
        plt.show()
