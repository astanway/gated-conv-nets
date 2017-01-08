import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import tensorflow as tf
import tensorflow as tf
import numpy as np
import random
from collections import defaultdict

#https://arxiv.org/pdf/1611.09482v1.pdf
# https://arxiv.org/pdf/1612.08083v1.pdf
#https://arxiv.org/pdf/1308.0850v5.pdf

# Placeholders for input, output and dropout

# Bug in TF sampled softmax implementation
@tf.RegisterGradient("LogUniformCandidateSampler")
def _LogUniformCandidateSamplerGrad(op,grad,foo,bar):
  return [tf.cast(tf.zeros_like(foo), tf.int64)]


vocabulary = set()
vocab_mapping = {}
sequence_length = 20
vocab_size = 1
embedding_size = 128
minibatch_size = 750
train = True

def get_data(test=False):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        return [l[i:i + n] for i in range(0, len(l)) if len(l[i:i + n]) == n]

    global vocab_size
    global vocabulary
    y = []
    x = []
    lines = []

    # Open both train and test data so that they can share a unified vocabulary
    with open('big.wiki.train.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if not test: # Add the data to lines if it isn't a test
                    lines.append(l)
                for w in l:
                    vocabulary.add(w)

    with open('big.wiki.test.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if test: # Add data to lines if this is a test run
                    lines.append(l)
                for w in l:
                    vocabulary.add(w)

    vocab_mapping = {i:x for x, i in enumerate(vocabulary)}
    vocab_size = len(vocabulary)
    clist = [chunks(l, sequence_length) for l in lines]

    for chunks in clist:
        for chunk in chunks:
            x.append([vocab_mapping[word] for word in chunk])
            del chunk[0]
            y.append([vocab_mapping[word] for word in chunk])

    return x, y

o = tf.Print(train, [train])
input_x = tf.placeholder(tf.int32, shape=(minibatch_size, sequence_length), name="input_x")
input_y = tf.placeholder(tf.float32, shape=(minibatch_size, sequence_length - 1), name="input_y")

def init_vars():

    # Embedding layer
    Wembedding= tf.Variable(tf.random_normal([vocab_size, embedding_size], stddev=.01), name="Wembedding")
    embedded_words = tf.nn.embedding_lookup(Wembedding, input_x)
    embedded_words_expanded = tf.expand_dims(embedded_words, 1) # give it height of 1

    def glu(filter_shape, layer_input, layer_name, res=False, last_layer=False):
        global padded
        kernel_width = filter_shape[1]
        left_pad = kernel_width - 1
        paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
        padded_input = tf.pad(layer_input, paddings, "CONSTANT")

        W = tf.Variable(tf.random_normal(filter_shape, stddev=np.sqrt(2.0 / (filter_shape[0] * filter_shape[1]))), name="W%s" % layer_name)

        # mask the kernel to future words from leaking into the kernel
        #center_w = filter_shape[1] // 2
        #mask = np.ones((filter_shape), dtype=np.float32)
        #mask[:, center_w+1: ,: ,:] = 0.
        #W *= tf.constant(mask, dtype=tf.float32)

        b = tf.Variable(tf.zeros(shape=[filter_shape[2] * filter_shape[3]]), name="b%s" % layer_name)
        conv1 = tf.nn.depthwise_conv2d(
            padded_input,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv1")

        conv1 = tf.nn.bias_add(conv1, b)

        V = tf.Variable(tf.random_normal(filter_shape, stddev=np.sqrt(2.0 / (filter_shape[0] * filter_shape[1]))), name="V%s" % layer_name)
        c = tf.Variable(tf.zeros(shape=[filter_shape[2] * filter_shape[3]]), name="c%s" % layer_name)
        conv2 = tf.nn.depthwise_conv2d(
            padded_input,
            V,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv2")
        conv2 = tf.sigmoid(tf.nn.bias_add(conv2, c))

        h = tf.multiply(conv1, conv2)

        # residual layer - add input to outpu
        if res:
            h = tf.add(h, padded_input)

        return h


    # [filter_height, filter_width, in_channels, out_channels]
    filter_shape = [1, 3, embedding_size, 1]
    h1 = glu(filter_shape, embedded_words_expanded, 1)

    filter_shape = [1, 3, 128, 1]
    h1a = glu(filter_shape, h1, 2)

    filter_shape = [1, 3, 128, 1]
    h2 = glu(filter_shape, h1a, 2)

    filter_shape = [1, 5, 128, 2]
    h3 = glu(filter_shape, h2, 3)

    filter_shape = [1, 5, 256, 2]
    h4 = glu(filter_shape, h3, 4)

    # filter_shape = [4, 128, 1, 128]
    # h5 = glu(filter_shape, paddings, h4, 5)
    #
    # filter_shape = [4, 128, 1, 128]
    # h6 = glu(filter_shape, paddings, h5, 6)

    # filter_shape = [4, 128, 1, 128]
    # h7 = glu(filter_shape, paddings, h6, 7)
    #
    # filter_shape = [4, 128, 1, 128]
    # h8 = glu(filter_shape, paddings, h7, 8)
    #
    # filter_shape = [4, 128, 1, 128]
    # h9 = glu(filter_shape, paddings, h8, 9)
    # h9 = tf.add(h8, h9) # residual layer

    output_embedding_size = 1024
    filter_shape = [1, 5, 512, 2]
    h10 = glu(filter_shape, h4, 10, last_layer=True)
    # Remove the last element, as the next word is in a new sequence and we do not predict it
    # Todo: this isn't really right
    h10 = tf.slice(h10, [0, 0, 0, 0], [-1, -1, sequence_length-1, -1])
    h10 = tf.squeeze(h10)

    def compute_sampled_softmax(hidden):
        labels = tf.cast(hidden[:, -1], tf.int64)
        labels = tf.expand_dims(labels, 1)
        hidden = tf.slice(hidden, [0, 0], [-1, output_embedding_size])
        losses = tf.nn.sampled_softmax_loss(output_embedding, output_bias, hidden, labels, 25, vocab_size, num_true=1, remove_accidental_hits=True, partition_strategy='mod', name='sampled_softmax_loss')
        return losses

    def compute_full_softmax(word):
        # Extract the label that we sent earlier
        label = tf.cast(word[-1], tf.int32)
        word = tf.slice(word, [0], [embedding_size])

        # Calculate total
        word = tf.transpose(tf.expand_dims(word, 1))
        total = tf.reduce_sum(tf.map_fn(lambda output_slice: tf.exp(tf.matmul(word, tf.expand_dims(output_slice, 1))), output_embedding))

        # Get the probability of the labeled word.
        output_slice = tf.nn.embedding_lookup(output_embedding, label)
        labeled_prob = tf.exp(tf.reduce_sum(tf.matmul(word, tf.expand_dims(output_slice, 1)))) / total

        #### FOR GENERATION WITHOUT LABELS #####
        # Get individual probabilities per word
        # probs = tf.map_fn(lambda output_slice: tf.exp(tf.reduce_sum(tf.matmul(word, tf.expand_dims(output_slice, 1)))) / total, output_embedding)

        # Make sure they add to 1
        # sum_probs = tf.reduce_sum(probs)

        # Word with highest probability is predicted word
        # predicted_word = tf.argmax(probs, 0)

        # Get the probability of the predicted word
        # predicted_word_prob = probs[tf.cast(predicted_word, tf.int32)]

        return -tf.log(labeled_prob)

    output_embedding = tf.Variable(tf.random_normal([vocab_size, output_embedding_size], stddev=.001), name="output_embedding")
    output_bias = tf.Variable(tf.zeros([vocab_size]), name="output_bias")

    # Concate the labels onto the context index and send off to be evaluated
    concated = tf.concat(2, [h10, tf.expand_dims(input_y, 2)])

    losses = tf.map_fn(lambda sequence: compute_sampled_softmax(sequence), concated)
    loss = tf.reduce_mean(losses)

    sum_losses = tf.map_fn(lambda sequence_loss: tf.reduce_sum(sequence_loss), losses)
    batch_perplexities = tf.map_fn(lambda sum_loss: tf.exp(1.0/sequence_length) * sum_loss, sum_losses)
    perplexity = tf.reduce_mean(batch_perplexities)
    p = tf.Print(perplexity, [perplexity], summarize=5000, message="perplexity")
    l = tf.Print(loss, [loss], summarize=5000, message="loss")

    if train:
        optimizer = tf.train.MomentumOptimizer(.5, .99)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -.1, .1), var) for grad, var in gvs if grad is not None]
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = optimizer.apply_gradients(capped_gvs, global_step)
        return train_step, global_step, p, l
    else:
        return p, l

def run():
    if train:
        x, y = get_data()
        train_step, global_step, p, l = init_vars()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        tf.global_variables_initializer().run()

    else:
        x, y = get_data(test=True)
        ckpt = tf.train.get_checkpoint_state('.')
        p, l = init_vars()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        saver.restore(sess, ckpt.model_checkpoint_path)

    print minibatch_size
    print len(x) / minibatch_size
    print len(vocabulary)

    for epoch in range(0, 50):
        print "epoch  %s" % epoch
        indices = range(0, len(x))
        for minibatch in range(0, len(x)):
            print "%s/%s" % (minibatch, len(indices)/minibatch_size)
            m_x = []
            m_y = []
            for x_i in range(0, minibatch_size):
                if len(indices) == 0:
                    break

                index = random.randrange(len(indices))

                m_x.append(x[index])
                m_y.append(y[index])
                del indices[index]

            m_x = np.array(m_x)
            m_y = np.array(m_y)

            if len(m_x) < minibatch_size:
                break

            if train:
                sess.run([train_step, global_step, p, l], feed_dict={input_x: m_x, input_y: m_y})
                if minibatch % 100 == 0:
                    saver.save(sess, 'model.ckpt', global_step=global_step)
            else:
                sess.run([p, l], feed_dict={input_x: m_x, input_y: m_y})

run()
# to test perpleity, feed each sequence into the model and then add up the final scores.
# output: not one word, n-1 words. so you can pad. don't need to reuce dimensionarliy of sequence length.