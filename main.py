import argparse
from bigram_character_model import BigramCharacterModel


def main(filename, num_words_to_generate, show_bigram_model_visualization):
    words = open(filename, "r").read().splitlines()

    # Train the model and maybe show a visualization
    bigram_character_model = BigramCharacterModel()
    bigram_character_model.train(words)
    if show_bigram_model_visualization:
        bigram_character_model.visualize()

    # Generate the model
    for _ in range(num_words_to_generate):
        print(bigram_character_model.generate_word())

    # Print the loss against training data
    print(f"\nloss (NLL): {bigram_character_model.get_loss(words)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some words.")
    parser.add_argument(
        "-filename",
        required=True,
        help="file containing a list of newline-separated words for training.",
    )
    parser.add_argument(
        "-num-words-to-generate",
        default=10,
        help="""number of new words to generate after training the model
                (default: %(default)s).""",
    )
    parser.add_argument(
        "-show-bigram-model-visualization",
        default=False,
        help="""shows a visualization contains the counts of all bigrams in the
                training data. Note that '-' is the special token used to
                designate the start and end of a word.""",
    )
    args = parser.parse_args()
    main(
        args.filename, args.num_words_to_generate, args.show_bigram_model_visualization
    )
