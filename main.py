import argparse
from bigram_character_model import BigramCharacterModel
from bigram_character_model_nn import BigramCharacterModelNN


def main(model, filename, num_words_to_generate, show_bigram_model_visualization):
    words = open(filename, "r").read().splitlines()

    # Train the model and maybe show a visualization
    model = model()
    model.train(words)
    if show_bigram_model_visualization:
        model.visualize()

    # Generate the model
    for _ in range(num_words_to_generate):
        print(model.generate_word())

    # Print the loss against training data
    print(f"\nloss (NLL): {model.get_loss(words)}")


if __name__ == "__main__":
    STR_TO_MODEL = {"BIGRAM": BigramCharacterModel, "BIGRAM_NN": BigramCharacterModelNN}
    parser = argparse.ArgumentParser(description="Process some words.")

    parser.add_argument("-model", choices=STR_TO_MODEL.keys(), help="model to use")
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
        STR_TO_MODEL[args.model],
        args.filename,
        args.num_words_to_generate,
        args.show_bigram_model_visualization,
    )
