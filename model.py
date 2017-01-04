import tensorflow as tf
import numpy as np
import random
from collections import defaultdict

#https://arxiv.org/pdf/1611.09482v1.pdf
# https://arxiv.org/pdf/1612.08083v1.pdf
#https://arxiv.org/pdf/1308.0850v5.pdf

# Placeholders for input, output and dropout

lines = []
vocabulary = set()
index = 0
sequence_length = 20

y = []
x = []

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    return [l[i:i + n] for i in range(0, len(l)) if len(l[i:i + n]) == n]

with open('wiki.train.tokens') as f:
    for line in f:
        if index == 1000:
            break
        index += 1
        l = line.lower().strip().split()
        if len(l) >= sequence_length:
            lines.append(l)
            for w in l:
                vocabulary.add(w)

embedding_size = 128
vocab_mapping = {i:x for x, i in enumerate(vocabulary)}
vocab_size = len(vocabulary)
block_size = 3
num_units = 128
layers = 128
kernel_width = 4

clist = [chunks(l, sequence_length + 1) for l in lines]

for chunks in clist:
    for chunk in chunks:
        y.append(vocab_mapping[chunk[-1]])
        del chunk[-1]
        x.append([vocab_mapping[word] for word in chunk])

x = [x[i] for i in range(0, len(x))]
y = [y[i] for i in range(0, len(y))]

print len(x)
print len(vocabulary)

input_x = tf.placeholder(tf.int32, shape=(None, sequence_length), name="input_x")
input_y = tf.placeholder(tf.int64, shape=(None), name="input_y")

# Embedding layer
Wembedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -.1, .1), name="Wembedding")
embedded_words = tf.nn.embedding_lookup(Wembedding, input_x)
embedded_words_expanded = tf.expand_dims(embedded_words, -1)
paddings = [[0,0],[0,0],[0,0],[0,0]]
def glu(filter_shape, paddings, layer_input, layer_name, res=False):
    global padded
    padded_input = tf.pad(layer_input, paddings, "CONSTANT")

    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=np.sqrt(1.0 / (filter_shape[0] * filter_shape[1]))), name="W%s" % layer_name)

    # mask the kernel to future words from leaking into the kernel
    # center_w = filter_shape[1] // 2
    # mask = np.ones((filter_shape), dtype=np.float32)
    # mask[:, center_w+1: ,: ,:] = 0.
    # W *= tf.constant(mask, dtype=tf.float32)

    b = tf.Variable(tf.zeros(shape=[filter_shape[-1]]), name="b%s" % layer_name)
    conv1 = tf.nn.conv2d(
        padded_input,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv1")

    conv1 = tf.nn.bias_add(conv1, b)

    V = tf.Variable(tf.truncated_normal(filter_shape, stddev=np.sqrt(1.0 / (filter_shape[0] * filter_shape[1]))), name="V%s" % layer_name)
    c = tf.Variable(tf.zeros(shape=[filter_shape[-1]]), name="c%s" % layer_name)
    conv2 = tf.nn.conv2d(
        padded_input,
        V,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv2")
    conv2 = tf.sigmoid(tf.nn.bias_add(conv2, c))

    h = tf.multiply(conv1, conv2)

    # residual layer - add input to output
    if res:
        h = tf.add(h, padded_input)

    h = tf.squeeze(h)
    h = tf.expand_dims(h, -1)

    return h


# [filter_height, filter_width, in_channels, out_channels]
filter_shape = [3, 128, 1, 128]
h1 = glu(filter_shape, paddings, embedded_words_expanded, 1)

filter_shape = [3, 128, 1, 128]
h1a = glu(filter_shape, paddings, h1, 2)

filter_shape = [3, 128, 1, 128]
h2 = glu(filter_shape, paddings, h1a, 2)

filter_shape = [4, 128, 1, 128]
h3 = glu(filter_shape, paddings, h2, 3)

#filter_shape = [4, 128, 1, 128]
#h4 = glu(filter_shape, paddings, h3, 4)

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

filter_shape = [11, 128, 1, 128]
h10 = glu(filter_shape, paddings, h3, 10)

def compute_loss(hidden):
    """ Softmax """
    shaped = tf.squeeze(hidden)
    total = tf.reduce_sum(tf.map_fn(lambda x: tf.exp(tf.reduce_sum(tf.mul(shaped, x))), output_embedding))
    output = tf.exp(tf.reduce_sum(tf.mul(shaped, tf.nn.embedding_lookup(output_embedding, input_y)))) / total
    return -tf.log(output)


# Init weights, bias, (all zeros first)
print "init output embedding"
output_embedding = tf.Variable(tf.zeros([vocab_size, 128], name="output_embedding"))

# Compute average loss across minibatch
print "init losses"
losses = tf.map_fn(lambda x: compute_loss(x), h10)
o = tf.Print(losses, [losses], summarize=128)
print "init loss"
loss = tf.reduce_mean(losses)

# Trainer
print "init optimizer"
optimizer = tf.train.MomentumOptimizer(.7, .99)
print "init gradients"
gvs = optimizer.compute_gradients(loss, gate_gradients=0)
print "init clipper"
capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
print "init train step"
train_step = optimizer.apply_gradients(capped_gvs)
print "inited."

saver = tf.train.Saver()
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
print "initializing..."
tf.global_variables_initializer().run()
print "initialized."

def run():
    for minibatch in range(0, 10000):
        print minibatch
        m_x = []
        m_y = []
        for x_i in range(0, 10):
            if len(x) == 0:
                return

            index = random.randrange(len(x))
            m_x.append(x[index])
            m_y.append(y[index])
            del x[index]
            del y[index]

        m_x = np.array(m_x)
        m_y = np.array(m_y)

        sess.run([train_step, o], feed_dict={input_x: m_x, input_y: m_y})

        if minibatch % 20 == 0:
            saver.save(sess, 'model.ckpt', global_step=i)

run()