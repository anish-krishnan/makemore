import matplotlib.pyplot as plt
import string
import torch


class BigramCharacterModel:
    SPECIAL_TOKEN = "-"
    IMAGE_SIZE = 8
    FONT_SIZE = 5

    def __init__(self):
        alphabet = list(self.SPECIAL_TOKEN + string.ascii_lowercase)
        self.K = len(alphabet)
        self.char_to_int = {char: idx for idx, char in enumerate(alphabet)}
        self.int_to_char = {idx: char for idx, char in enumerate(alphabet)}
        self.bigram_counts = torch.zeros((self.K, self.K), dtype=torch.int32)
        self.bigram_probabilities = torch.tensor((self.K, self.K), dtype=torch.float32)

    def get_bigrams_from_words(self, words):
        for word in words:
            word = f"{self.SPECIAL_TOKEN}{word}{self.SPECIAL_TOKEN}"
            for char1, char2 in zip(word, word[1:]):
                idx1, idx2 = self.char_to_int[char1], self.char_to_int[char2]
                yield ((char1, idx1), (char2, idx2))

    def train(self, words):
        # Initialize the counts with 1's to smoothen the model. This also
        # prevents us from dealing with zero probabilities (infinite loss) if
        # certain bigrams never appear in the training set.
        self.bigram_counts = torch.ones((self.K, self.K), dtype=torch.int32)
        for (_, idx1), (_, idx2) in self.get_bigrams_from_words(words):
            self.bigram_counts[idx1, idx2] += 1

        # Normalize the counts into probabilities
        bigram_counts = self.bigram_counts.float()
        self.bigram_probabilities = bigram_counts / bigram_counts.sum(
            dim=1, keepdim=True
        )

    def visualize(self, image_size=IMAGE_SIZE, font_size=FONT_SIZE):
        plt.rcParams.update({"font.size": font_size})
        plt.figure(figsize=(image_size, image_size))
        plt.imshow(self.bigram_counts, cmap="Blues")
        for i in range(self.K):
            for j in range(self.K):
                bigram = self.int_to_char[i] + self.int_to_char[j]
                count = self.bigram_counts[i, j].item()
                plt.text(j, i, bigram, ha="center", va="bottom", color="gray")
                plt.text(
                    j,
                    i,
                    count,
                    ha="center",
                    va="top",
                    color="gray",
                )
        plt.show()

    def generate_word(self):
        def sample_next_char_idx(char_idx):
            distribution = self.bigram_probabilities[char_idx]
            return torch.multinomial(
                distribution, num_samples=1, replacement=True
            ).item()

        word = ""
        special_token_idx = self.char_to_int[self.SPECIAL_TOKEN]
        current_char_idx = self.char_to_int[self.SPECIAL_TOKEN]
        while (
            current_char_idx := sample_next_char_idx(current_char_idx)
        ) != special_token_idx:
            word += self.int_to_char[current_char_idx]
        return word

    def get_loss(self, words):
        log_likelihood = 0.0
        N = 0
        for (_, idx1), (_, idx2) in self.get_bigrams_from_words(words):
            N += 1
            log_likelihood += torch.log(self.bigram_probabilities[idx1, idx2])
        return -log_likelihood / N
