This implements a sigmoid gated convolutional network, as per https://arxiv.org/pdf/1612.08083v1.pdf.

![x](https://raw.githubusercontent.com/astanway/gated-conv-nets/master/arch.png)

## To run
`python model.py --train True`

## Discussion:
  The general technique in this paper is an attempt to set up a convolutional network to achieve the same sorts of things an LSTM or RNN is traditionally good at, while taking advantage of the CNNs non-temporal nature to effect big speed gains. For the language modeling task, this means that the inputs are a sequence of learned word embeddings, and the outputs are that same sequence, but shifted to the left. The final output embedding for a word (a vector within the final hidden layer) is thus trying to predict the word in front of it, a probability which is calculated with the softmax for each vector in the final hidden layer.

  In this way, all output projections are computed simultaneously through the convolutional layers. Care must be taken to properly pad the inputs such that each convolution kernel cannot see any context in front of it, as this would constitute data leakage.

  As with most language modeling tasks, the most expensive part of the computation is the softmax stage, where each output vector must calculate the probability with respect to the target word. This is expensive because all of the output probabilities must sum to one, so the total across all possible tokens in the vocabulary must be computed as the divisor. The authors of used a newer technique called the adaptive softmax to approximate this for speed. I have opted for Tensorflow's native implementation of sampled softmax for now until I have the time to read the adaptive softmax paper (https://arxiv.org/pdf/1609.04309v2.pdf) and understand how it works. The sampled softmax works by simply approximating the total by sampling over a handful of classes in the vocabulary. This tends to work because most vocabularies have a Zipfian distribution.

  It takes about an hour and a half to train one epoch of [WikiText-2](http://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/) on an Amazon p2.xlarge Tesla K80.
