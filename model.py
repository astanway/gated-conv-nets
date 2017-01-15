import os
import glob
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
flags.DEFINE_bool("train", False, "Train the model or run it on the test set only")
flags.DEFINE_bool("valid", False, "Validate")
flags.DEFINE_bool("test", False, "Test")
FLAGS = flags.FLAGS

sequence_length = 25
embedding_size = 128
minibatch_size = 750
candidates = 10000

def get_data():
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        return [l[i:i + n] for i in range(0, len(l)) if len(l[i:i + n]) == n]

    vocabulary = set()
    lines = []
    y = []
    x = []

    # Open train, test, and validation data so that they can share a unified vocabulary and model
    with open('wiki.train.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if FLAGS.train:
                    lines.append(l)
                for w in l:
                    vocabulary.add(w)

    with open('wiki.valid.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if FLAGS.valid:
                    lines.append(l)
                for w in l:
                    vocabulary.add(w)

    with open('wiki.test.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if FLAGS.test:
                    lines.append(l)
                for w in l:
                    vocabulary.add(w)

    vocab_mapping = {i:x for x, i in enumerate(vocabulary)}
    vocab_size = len(vocabulary)

    clist = [chunks(c, sequence_length) for c in lines]
    for c in clist:
        for chunk in c:
            x.append([vocab_mapping[word] for word in chunk])
            del chunk[0]
            y.append([vocab_mapping[word] for word in chunk])

    return x, y, vocab_mapping

def glu(kernel_shape, layer_input, layer_name, residual=None):
    """ Gated Linear Unit """
    # Pad the left side to prevent kernels from viewing future context
    kernel_width = kernel_shape[1]
    left_pad = kernel_width - 1
    paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
    padded_input = tf.pad(layer_input, paddings, "CONSTANT")

    # Kaiming intialization
    stddev = np.sqrt(2.0 / (kernel_shape[1] * kernel_shape[2]))

    # First conv layer
    W_v = tf.Variable(tf.random_normal(kernel_shape, stddev=stddev), name="W%s" % layer_name)
    W = tf.Variable(stddev, dtype=tf.float32) * W_v / tf.nn.l2_normalize(W_v, 0)
    b = tf.Variable(tf.zeros([kernel_shape[2] * kernel_shape[3]]), name="b%s" % layer_name)
    conv1 = tf.nn.depthwise_conv2d(
        padded_input,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv1")
    conv1 = tf.nn.bias_add(conv1, b)

    # Second gating sigmoid layer
    V_v = tf.Variable(tf.random_normal(kernel_shape, stddev=stddev), name="V%s" % layer_name)
    V = tf.Variable(stddev, dtype=tf.float32) * V_v / tf.nn.l2_normalize(V_v, 0)
    c = tf.Variable(tf.zeros([kernel_shape[2] * kernel_shape[3]]), name="c%s" % layer_name)
    conv2 = tf.nn.depthwise_conv2d(
        padded_input,
        V,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv2")
    conv2 = tf.nn.bias_add(conv2, c)

    # Preactivation residual
    if residual is not None:
        conv1 = tf.add(conv1, residual)
        conv2 = tf.add(conv2, residual)

    h = tf.multiply(conv1, tf.sigmoid(conv2, name="sig"))

    return h

def setup_model(vocab_mapping, epoch_steps):
    """ Setup the model after we have imported the data and know the vocabulary size """

    vocab_size = len(vocab_mapping)
    all_word_embeddings = tf.Variable(tf.random_normal([vocab_size, embedding_size], stddev=.01), name="all_word_embeddings")
    input_embeddings = tf.nn.embedding_lookup(all_word_embeddings, input_x)
    input_embeddings_expanded = tf.expand_dims(input_embeddings, 1)

    # [height, width, in_channels, out_channels]
    kernel_shape = [1, 3, embedding_size, 1]
    h0 = glu(kernel_shape, input_embeddings_expanded, 0)
    h1 = glu(kernel_shape, h0, 1)
    h2 = glu(kernel_shape, h1, 2)
    h3 = glu(kernel_shape, h2, 3)
    h4 = glu(kernel_shape, h3, 4, h0)

   #  For larger models with output projections:
    kernel_shape = [1, 3, 128, 2]
    h4a = glu(kernel_shape, h4, '14a')

    kernel_shape = [1, 3, 256, 1]
    h5 = glu(kernel_shape, h4a, 5)
    h6 = glu(kernel_shape, h5, 6)
    h7 = glu(kernel_shape, h6, 7)
    h8 = glu(kernel_shape, h7, 8)
    h9 = glu(kernel_shape, h8, 9, h4a)

    # Output embeddings
    output_weights_size = kernel_shape[2] * kernel_shape[3]
    stddev = np.sqrt(2.0 / (kernel_shape[1] * kernel_shape[2]))
    output_weights = tf.Variable(tf.random_normal([vocab_size, output_weights_size], stddev=stddev), name="output_weights")
    output_bias = tf.Variable(tf.zeros([vocab_size]), name="output_bias")

    # Remove the last element, as the next word is in a new sequence and we do not predict it
    # Reshape labels and hidden layer as we're only interested in scoring at the word level
    last_hidden = h9
    last_hidden = tf.slice(last_hidden, [0, 0, 0, 0], [-1, -1, sequence_length-1, -1])
    last_hidden = tf.squeeze(last_hidden)
    last_hidden = tf.reshape(last_hidden, [minibatch_size * (sequence_length - 1), output_weights_size])
    labels = tf.expand_dims(tf.reshape(input_y, [-1]), 1)

   # Todo: sampled softmax for larger vocabularies
   # losses = tf.nn.sampled_softmax_loss(output_weights, output_bias, last_hidden, labels, candidates, vocab_size, num_true=1, partition_strategy='mod', name='ssl')

    multiplied = tf.matmul(last_hidden, tf.transpose(output_weights)) + output_bias
    logits = tf.nn.softmax(multiplied) # adds to 1, for each word
    rows = tf.expand_dims(tf.constant(range(0, minibatch_size * (sequence_length - 1))), 1)
    indices = tf.concat(1, [rows, labels])
    probs = tf.gather_nd(logits, indices)
    losses = -tf.log(probs)
    loss = tf.reduce_mean(losses)
    perplexity = tf.exp(loss)
    l = tf.Print(loss, [loss], summarize=21, message="")
    p = tf.Print(perplexity, [perplexity], summarize=21, message="")
    tf.summary.scalar('perplexity', perplexity)
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('losses', losses)

    # Clip and optimize
    if FLAGS.train:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(.5, global_step, 1, 0.9, staircase=False)
        optimizer = tf.train.MomentumOptimizer(.08, .8)
        gvs = optimizer.compute_gradients(losses)
        capped_gvs = [(tf.clip_by_norm(grad, .1), var) for grad, var in gvs if grad is not None]
        train_step = optimizer.apply_gradients(capped_gvs, global_step)
        return train_step, global_step, p, l
    else:
        return p, l

if __name__=="__main__":
    input_x = tf.placeholder(tf.int32, shape=(minibatch_size, sequence_length), name="input_x")
    input_y = tf.placeholder(tf.int32, shape=(minibatch_size, sequence_length - 1), name="input_y")
    x, y, vocab_mapping = get_data()

    print minibatch_size
    print len(x) / minibatch_size
    print len(vocab_mapping)
    epoch_steps = len(x) / minibatch_size

    if FLAGS.train:
        logdir = 'train_summaries'
        train_step, global_step, p, l = setup_model(vocab_mapping, epoch_steps)
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()
    else:
        if FLAGS.valid:
            logdir = 'valid_summaries'
        elif FLAGS.test:
            logdir = 'test_summaries'

        p, l = setup_model(vocab_mapping, epoch_steps)
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config=config)
        ckpt = tf.train.get_checkpoint_state('./train_summaries')
        saver.restore(sess, ckpt.model_checkpoint_path)

    files = glob.glob('/home/ubuntu/gated-conv-nets/' + logdir + '/*')
    for f in files:
        os.remove(f)

    writer = tf.summary.FileWriter(logdir, sess.graph)
    merged = tf.summary.merge_all()

    for epoch in range(0, 10000):
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
                saver.save(sess, logdir + '/model.ckpt', global_step=global_step)
                summary, t_, g_, p_, l_ = sess.run([merged, train_step, global_step, p, l], feed_dict={input_x: m_x, input_y: m_y})
                writer.add_summary(summary, minibatch)
                writer.flush()
            else:
                summary, p_, l_ = sess.run([merged, p, l], feed_dict={input_x: m_x, input_y: m_y})
                writer.add_summary(summary, minibatch)
                writer.flush()