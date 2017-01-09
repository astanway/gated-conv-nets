import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import tensorflow as tf
import tensorflow as tf
import numpy as np
import random
from collections import defaultdict

# Bug in TF sampled softmax implementation
@tf.RegisterGradient("LogUniformCandidateSampler")
def _LogUniformCandidateSamplerGrad(op,grad,foo,bar):
  return [tf.cast(tf.zeros_like(foo), tf.int64)]

flags = tf.flags
flags.DEFINE_bool("train", True, "Train the model or run it on the test set only")
FLAGS = flags.FLAGS

# Config
sequence_length = 20
embedding_size = 128
minibatch_size = 750

def get_data():
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        return [l[i:i + n] for i in range(0, len(l)) if len(l[i:i + n]) == n]

    vocabulary = set()
    y = []
    x = []
    lines = []

    # Open both train and test data so that they can share a unified vocabulary
    with open('wiki.train.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if FLAGS.train: # Add the data to lines we are training
                    lines.append(l)
                for w in l:
                    vocabulary.add(w)

    with open('wiki.test.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if not FLAGS.train: # Add data to lines if this is a test run
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

    return x, y, vocab_mapping

def glu(kernel_shape, layer_input, layer_name):
    """ Gated Linear Unit """
    # Pad the left side to prevent kernels from viewing future context
    kernel_width = kernel_shape[1]
    left_pad = kernel_width - 1
    paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
    padded_input = tf.pad(layer_input, paddings, "CONSTANT")

    # First convolutional layer, Kaiming intialization
    W = tf.Variable(tf.random_normal(kernel_shape, stddev=np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1]))), name="W%s" % layer_name)
    b = tf.Variable(tf.zeros(shape=[kernel_shape[2] * kernel_shape[3]]), name="b%s" % layer_name)
    conv1 = tf.nn.depthwise_conv2d(
        padded_input,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv1")
    conv1 = tf.nn.bias_add(conv1, b)

    # Gating sigmoid layer, Kaiming intialization
    V = tf.Variable(tf.random_normal(kernel_shape, stddev=np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1]))), name="V%s" % layer_name)
    c = tf.Variable(tf.zeros(shape=[kernel_shape[2] * kernel_shape[3]]), name="c%s" % layer_name)
    conv2 = tf.nn.depthwise_conv2d(
        padded_input,
        V,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv2")
    conv2 = tf.sigmoid(tf.nn.bias_add(conv2, c))

    h = tf.multiply(conv1, conv2)

    return h

def compute_sampled_softmax(output_weights, output_bias, sequence, output_weights_size, vocab_size):
    """ Compute sampled softmax for training"""
    labels = tf.cast(sequence[:, -1], tf.int64)
    labels = tf.expand_dims(labels, 1)
    sequence= tf.slice(sequence, [0, 0], [-1, output_weights_size])
    losses = tf.nn.sampled_softmax_loss(output_weights, output_bias, sequence, labels, 25, vocab_size, num_true=1, remove_accidental_hits=True, partition_strategy='mod', name='sampled_softmax_loss')
    return losses

def setup_model(vocab_mapping, epoch_steps):
    """ Setup the model after we have imported the data and know the vocabulary size """
    # Embedding layer
    vocab_size = len(vocab_mapping)
    all_word_embeddings = tf.Variable(tf.random_normal([vocab_size, embedding_size], stddev=.01), name="all_word_embeddings")
    input_embeddings = tf.nn.embedding_lookup(all_word_embeddings, input_x)
    input_embeddings_expanded = tf.expand_dims(input_embeddings, 1) # give it height of 1

    # [height, width, in_channels, out_channels]
    kernel_shape = [1, 3, embedding_size, 1]
    h0 = glu(kernel_shape, input_embeddings_expanded, 1)

    kernel_shape = [1, 3, 128, 1]
    h1 = glu(kernel_shape, h0, 2)

    kernel_shape = [1, 3, 128, 1]
    h2 = glu(kernel_shape, h1, 3)

    h2 = tf.add(h2, h0) # skip two residual layer

    kernel_shape = [1, 5, 128, 2]
    h3 = glu(kernel_shape, h2, 4)

    kernel_shape = [1, 5, 256, 1]
    h4 = glu(kernel_shape, h3, 5)

    kernel_shape = [1, 5, 256, 1]
    h5 = glu(kernel_shape, h4, 6)

    h5 = tf.add(h5, h3) # skip two residual layer

    kernel_shape = [1, 5, 256, 2]
    h6 = glu(kernel_shape, h5, 7)

    kernel_shape = [1, 5, 512, 2]
    last_hidden = glu(kernel_shape, h6, 8)

    # Output word embeddings. Note: these are not the same as the input word embeddings.
    output_weights_size = kernel_shape[2] * kernel_shape[3]
    output_weights = tf.Variable(tf.random_normal([vocab_size, output_weights_size], stddev=.001), name="output_weights")
    output_bias = tf.Variable(tf.zeros([vocab_size]), name="output_bias")

    # Remove the last element, as the next word is in a new sequence and we do not predict it
    last_hidden = tf.slice(last_hidden, [0, 0, 0, 0], [-1, -1, sequence_length-1, -1])
    last_hidden = tf.squeeze(last_hidden)

    # Concat the labels onto the context index and send off to be evaluated
    concated = tf.concat(2, [last_hidden, tf.expand_dims(input_y, 2)])

    # Evaluate losses with a sampled softmax
    losses = tf.map_fn(lambda sequence: compute_sampled_softmax(output_weights, output_bias, sequence, output_weights_size, vocab_size), concated)
    loss = tf.reduce_mean(losses)

    # Find the perplexity
    sum_losses = tf.map_fn(lambda sequence_loss: tf.reduce_sum(sequence_loss), losses)
    batch_perplexities = tf.map_fn(lambda sum_loss: tf.exp(1.0/sequence_length) * sum_loss, sum_losses)
    perplexity = tf.reduce_mean(batch_perplexities)
    p = tf.Print(perplexity, [perplexity], summarize=5000, message="perplexity")
    l = tf.Print(loss, [loss], summarize=5000, message="loss")

    # If we are training a model, proceed to optimize gradients and backprop.
    # Gradient clipping set to -.1, .1.
    if FLAGS.train:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(0.5, global_step, epoch_steps, 0.8, staircase=True) # decay the learning every epoch
        optimizer = tf.train.MomentumOptimizer(learning_rate, .99)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -.1, .1), var) for grad, var in gvs if grad is not None]
        train_step = optimizer.apply_gradients(capped_gvs, global_step)
        return train_step, global_step, p, l
    else:
        return p, l

if __name__=="__main__":
    input_x = tf.placeholder(tf.int32, shape=(minibatch_size, sequence_length), name="input_x")
    input_y = tf.placeholder(tf.float32, shape=(minibatch_size, sequence_length - 1), name="input_y")
    x, y, vocab_mapping = get_data()

    print minibatch_size
    print len(x) / minibatch_size
    print len(vocab_mapping)
    epoch_steps = len(x) / minibatch_size

    if FLAGS.train:
        train_step, global_step, p, l = setup_model(vocab_mapping, epoch_steps)
        saver = tf.train.Saver()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        tf.global_variables_initializer().run()

    else:
        p, l = setup_model(vocab_mapping)
        saver = tf.train.Saver()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        ckpt = tf.train.get_checkpoint_state('.')
        saver.restore(sess, ckpt.model_checkpoint_path)

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

            if FLAGS.train:
                sess.run([train_step, global_step, p, l], feed_dict={input_x: m_x, input_y: m_y})
                if minibatch % 100 == 0:
                    saver.save(sess, 'model.ckpt', global_step=global_step)
            else:
                sess.run([p, l], feed_dict={input_x: m_x, input_y: m_y})