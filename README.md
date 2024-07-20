# makemore

A series of language models based on [Andrej Karpathy](https://karpathy.ai)'s [video series](https://youtu.be/PaCmpygFfXo?si=dJprQlzr0Sazt8UC) on [makemore](https://github.com/karpathy/makemore).

I have implemented the following models:
- Bigram Lanuage Model
    - Manually by maintaining the frequency of the training set
    - With a single layer neural network to incrementally learn the frequency distribution
- Multilayer Perceptron with a configurable context length and high dimensional embedding space for better prediction
