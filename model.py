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
candidates = 500
validating = False

def get_data():
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        return [l[i:i + n] for i in range(0, len(l)) if len(l[i:i + n]) == n]

    vocabulary = set()
    lines = []
    y = []
    x = []
    vlines = []
    v_x = []
    v_y = []

    # Open train, test, and validation data so that they can share a unified vocabulary and model
    with open('wiki.train.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if FLAGS.train: # Add the data to lines we are training
                    lines.append(l)
                for w in l:
                    vocabulary.add(w)

    with open('wiki.valid.tokens') as f:
        for line in f:
            line = "<S> " + line + " </S>"
            l = line.lower().strip().split()
            if len(l) >= sequence_length:
                if FLAGS.train: # Add data to lines if this is a test run
                    vlines.append(l)
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
    for c in clist:
        for chunk in c:
            x.append([vocab_mapping[word] for word in chunk])
            del chunk[0]
            y.append([vocab_mapping[word] for word in chunk])

    vlist = [chunks(v, sequence_length) for v in vlines]
    for c in vlist:
        for chunk in c:
            v_x.append([vocab_mapping[word] for word in chunk])
            del chunk[0]
            v_y.append([vocab_mapping[word] for word in chunk])
    return x, y, v_x, v_y, vocab_mapping

def glu(kernel_shape, layer_input, layer_name):
    """ Gated Linear Unit """
    # Pad the left side to prevent kernels from viewing future context
    kernel_width = kernel_shape[1]
    left_pad = kernel_width - 1
    paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
    padded_input = tf.pad(layer_input, paddings, "CONSTANT")

    # First convolutional layer, Kaiming intialization, weight normalized
    stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1]))
    W_v = tf.Variable(tf.random_normal(kernel_shape, stddev=stddev), name="W%s" % layer_name)
    W = tf.Variable(1.0 / stddev, dtype=tf.float32) * W_v / tf.nn.l2_normalize(W_v, 0)
    b = tf.Variable(tf.zeros(shape=[kernel_shape[2] * kernel_shape[3]]), name="b%s" % layer_name)
    conv1 = tf.nn.depthwise_conv2d(
        padded_input,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv1")
    conv1 = tf.nn.bias_add(conv1, b)

    # Gating sigmoid layer, Kaiming intialization, weight normalized
    V_v = tf.Variable(tf.random_normal(kernel_shape, stddev=stddev), name="V%s" % layer_name)
    V = tf.Variable(1.0 / stddev, dtype=tf.float32) * V_v / tf.nn.l2_normalize(V_v, 0)
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

def setup_model(vocab_mapping, epoch_steps):
    """ Setup the model after we have imported the data and know the vocabulary size """
    # Embedding layer
    vocab_size = len(vocab_mapping)
    all_word_embeddings = tf.Variable(tf.random_normal([vocab_size, embedding_size], stddev=.01), name="all_word_embeddings")
    input_embeddings = tf.nn.embedding_lookup(all_word_embeddings, input_x)
    input_embeddings_expanded = tf.expand_dims(input_embeddings, 1)

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

    kernel_shape = [1, 5, 512, 1]
    last_hidden = glu(kernel_shape, h6, 8)

    # Output word embeddings. Note: these are not the same as the input word embeddings.
    output_weights_size = kernel_shape[2] * kernel_shape[3]
    output_weights = tf.Variable(tf.random_normal([vocab_size, output_weights_size], stddev=.001), name="output_weights")
    output_bias = tf.Variable(tf.zeros([vocab_size]), name="output_bias")

    # Remove the last element, as the next word is in a new sequence and we do not predict it
    last_hidden = tf.slice(last_hidden, [0, 0, 0, 0], [-1, -1, sequence_length-1, -1])
    last_hidden = tf.squeeze(last_hidden)

    # Evaluate losses with a sampled softmax for training and a full softmax for validation and test
    if FLAGS.train and validating == False:
        last_hidden = tf.reshape(last_hidden, [minibatch_size * (sequence_length - 1), output_weights_size])
        labels = tf.expand_dims(tf.reshape(input_y, [-1]), 1)
        losses = tf.nn.sampled_softmax_loss(output_weights, output_bias, last_hidden, labels, candidates, vocab_size, num_true=1, remove_accidental_hits=True, partition_strategy='mod', name='sampled_softmax_loss')
        loss = tf.reduce_mean(losses)
    else:
        last_hidden = tf.reshape(last_hidden, [minibatch_size * (sequence_length - 1), output_weights_size])
        labels = tf.reshape(input_y, [-1])
        logits = tf.nn.softmax(tf.matmul(last_hidden, tf.transpose(output_weights)) + output_bias)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        loss = tf.reduce_mean(losses)

    # Calculate perplexities
    perplexity = tf.exp(loss)

    l = tf.Print(loss, [loss], summarize=5000, message="loss")
    p = tf.Print(perplexity, [perplexity], summarize=5000, message="log perplexity")
    tf.summary.scalar('loss', loss)
    tf.summary.tensor_summary('losses', losses)
    tf.summary.tensor_summary('perplexity', perplexity)

    # If we are training a model, proceed to optimize gradients and backprop.
    # Gradient clipping set to -.1, .1.
    if FLAGS.train:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(1.0, global_step, epoch_steps, 0.99999, staircase=False) # decay the learning every epoch
        optimizer = tf.train.MomentumOptimizer(learning_rate, .99)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -.1, .1), var) for grad, var in gvs if grad is not None]
        train_step = optimizer.apply_gradients(capped_gvs, global_step)
        return train_step, global_step, p, l
    else:
        return p, l

if __name__=="__main__":
    input_x = tf.placeholder(tf.int32, shape=(minibatch_size, sequence_length), name="input_x")
    input_y = tf.placeholder(tf.int32, shape=(minibatch_size, sequence_length - 1), name="input_y")
    x, y, v_x, v_y, vocab_mapping = get_data()

    print minibatch_size
    print len(x) / minibatch_size
    print len(vocab_mapping)
    epoch_steps = len(x) / minibatch_size

    if FLAGS.train:
        logdir = './train_summaries'
        train_step, global_step, p, l = setup_model(vocab_mapping, epoch_steps)
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config=config)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, sess.graph)
        tf.global_variables_initializer().run()

    else:
        logdir = './test_summaries'
        p, l = setup_model(vocab_mapping)
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config=config)
        ckpt = tf.train.get_checkpoint_state('.')
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, sess.graph)
        saver.restore(sess, ckpt.model_checkpoint_path)

    for epoch in range(0, 10):
        validating = False
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
                summary, t_, g_, p_, l_ = sess.run([merged, train_step, global_step, p, l], feed_dict={input_x: m_x, input_y: m_y})
                #writer.add_summary(summary)

		# Check validation perplexity every N steps
                if minibatch % 101 == 0:
                    print "validation perplexity:"
                    saver.save(sess, logdir + '/model.ckpt', global_step=global_step)
                    vindices = range(0, len(v_x))
                    m_x = []
                    m_y = []

                    for x_i in range(0, minibatch_size):
                        vindex = random.randrange(len(vindices))
                        m_x.append(v_x[vindex])
                        m_y.append(v_y[vindex])
                        del vindices[vindex]

                    m_x = np.array(m_x)
                    m_y = np.array(m_y)

                    sess.run([p, l], feed_dict={input_x: m_x, input_y: m_y})
                    validating = False
            else:
                sess.run([p, l], feed_dict={input_x: m_x, input_y: m_y})

        # Run the validation set on model to get validation perplexity for this epoch
        # Break after one run for now. TODO: proper softmax loss instead of messing with the candidate size
        if FLAGS.train:
            print "full validation perplexity:"
            validating = True
            indices = range(0, len(v_x))
            for minibatch in range(0, len(v_x)):
                print "%s/%s" % (minibatch, len(indices)/minibatch_size)
                m_x = []
                m_y = []
                for x_i in range(0, minibatch_size):
                    if len(indices) == 0:
                        break

                    index = random.randrange(len(indices))

                    m_x.append(v_x[index])
                    m_y.append(v_y[index])
                    del indices[index]

                m_x = np.array(m_x)
                m_y = np.array(m_y)

                if len(m_x) < minibatch_size:
                    break

                sess.run([p, l], feed_dict={input_x: m_x, input_y: m_y})
            validating = False