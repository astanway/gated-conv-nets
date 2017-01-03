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

y = []
x = []

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    return [l[i:i + n] for i in range(0, len(l)) if len(l[i:i + n]) == n]

with open('wiki.train.tokens') as f:
    for line in f:
        if index == 5:
            break
        index += 1
        print index
        l = line.lower().strip().split()
        if len(l) >= 33:
            lines.append(l)
            for w in l:
                vocabulary.add(w)

sequence_length = 32
embedding_size = 128
vocab_mapping = {i:x for x, i in enumerate(vocabulary)}
vocab_size = len(vocabulary)
block_size = 3
num_units = 128
layers = 128
kernel_width = 4

clist = [chunks(l, 33) for l in lines]

index = 0
for chunks in clist:
    index += 1
    print index
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

def glu(filter_shape, paddings, layer_input, layer_name):
    global padded
    padded_input = tf.pad(layer_input, paddings, "CONSTANT")
    print "padded input", padded_input

    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=1.0/filter_shape[2]), name="W%s" % layer_name)
    
    # mask the kernel to future words from leaking into the kernel
    # center_w = filter_shape[1] // 2
    # mask = np.ones((filter_shape), dtype=np.float32)
    # mask[:, center_w+1: ,: ,:] = 0.
    # W *= tf.constant(mask, dtype=tf.float32)

    #padded = tf.Print(W, [W], summarize=1000)

    b = tf.Variable(tf.constant(1.0 / filter_shape[2], shape=[filter_shape[-1]]), name="b%s" % layer_name)
    conv1 = tf.nn.conv2d(
        padded_input,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv1")

    conv1 = tf.nn.bias_add(conv1, b)

    V = tf.Variable(tf.truncated_normal(filter_shape, stddev=1.0/filter_shape[2]), name="V%s" % layer_name)
    c = tf.Variable(tf.constant(1.0 / filter_shape[2], shape=[filter_shape[-1]]), name="c%s" % layer_name)
    conv2 = tf.nn.conv2d(
        padded_input,
        V,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv2")
    conv2 = tf.sigmoid(tf.nn.bias_add(conv2, c))

    h = tf.multiply(conv1, conv2)
    h = tf.squeeze(h)
    h = tf.expand_dims(h, -1)

    return h

# mask the kernels
# do not zero pad. zero padding retains the same size
# if mask_type is not None:

# [filter_height, filter_width, in_channels, out_channels]
# use rectangular kernels
filter_shape = [3, 128, 1, 128]
#paddings = [[0,0], [1, 0], [0, 0], [0, 0]]
paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h1 = glu(filter_shape, paddings, embedded_words_expanded, 1)
print "first layer output", h1

filter_shape = [3, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h1a = glu(filter_shape, paddings, h1, 2)
print "second_layer output", h1a

filter_shape = [3, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h2 = glu(filter_shape, paddings, h1a, 2)
print "second_layer output", h2

filter_shape = [4, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h3 = glu(filter_shape, paddings, h2, 3)
print "third layer output", h3

filter_shape = [4, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h4 = glu(filter_shape, paddings, h3, 4)
print "third layer output", h4

filter_shape = [4, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h5 = glu(filter_shape, paddings, h4, 5)
print "third layer output", h5

filter_shape = [4, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h6 = glu(filter_shape, paddings, h5, 6)
print "third layer output", h6

filter_shape = [4, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h7 = glu(filter_shape, paddings, h6, 7)
print "third layer output", h7

filter_shape = [4, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h8 = glu(filter_shape, paddings, h7, 8)
print "third layer output", h8

filter_shape = [4, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h9 = glu(filter_shape, paddings, h8, 9)
print "third layer output", h9

filter_shape = [5, 128, 1, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h10 = glu(filter_shape, paddings, h9, 10)
print "third layer output", h10

#padded = tf.Print(shaped, [shaped], summarize=1000)
# TODO: Add the residual every 5 units
#if (i % 5 == 0):
#    gated = tf.add(gated, embedded_words_expanded)
## need fully connected layer

def compute_loss(hidden):
    shaped = tf.squeeze(hidden)
    sh = tf.Print(shaped, [shaped], summarize=128, message="sh")
    total = 0.0
    for i in range(0, vocab_size):
        output_embedding = tf.nn.embedding_lookup(softmax_w, i)
        total += tf.exp(tf.reduce_sum(tf.mul(shaped, output_embedding))) # dot product

    output = tf.exp(tf.reduce_sum(tf.mul(shaped, tf.nn.embedding_lookup(softmax_w, input_y)))) / total
    # m = tf.reduce_sum(tf.mul(shaped, tf.nn.embedding_lookup(softmax_w, input_y)))
    # muled = tf.Print(m, [m], summarize=128, message="m")
    # o = tf.Print(output, [output], message="o")
    # padded = tf.Print(total, [total], message="total")
    return -tf.log(output)


# Init weights, bioas, (all zeros first)
#softmax_w = tf.Variable(tf.random_uniform([vocab_size, 128], -.001, .001), name="softmax_w")
softmax_w = tf.Variable(tf.zeros([vocab_size, 128]), name="softmax_w")
# softmax

losses = tf.map_fn(lambda x: compute_loss(x), h10)


# Average loss across minibatch
loss = tf.reduce_mean(losses)
l = tf.Print(loss, [loss], message="loss")
s = tf.Print(tf.nn.embedding_lookup(softmax_w, input_y), [tf.nn.embedding_lookup(softmax_w, input_y)], summarize=128, message="s")


#loss_sum = tf.scalar_summary("loss", loss)

optimizer = tf.train.MomentumOptimizer(0.5, .99)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -.01, .01), var) for grad, var in gvs if grad != None]
train_step = optimizer.apply_gradients(capped_gvs)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def run():
    for minibatch in range(0, 1000):
        m_x = []
        m_y = []
        for x_i in range(0, 20):
            if len(x) == 0:
                return

            index = random.randrange(len(x))
            m_x.append(x[index])
            m_y.append(y[index])
            del x[index]
            del y[index]

        m_x = np.array(m_x)
        m_y = np.array(m_y)
        sess.run([train_step, l], feed_dict={input_x: m_x, input_y: m_y})

run()