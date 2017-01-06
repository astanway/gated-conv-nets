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

lines = []
vocabulary = set()
index = 0
sequence_length = 20

y = []
x = []

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    return [l[i:i + n] for i in range(0, len(l)) if len(l[i:i + n]) == n]

index = 0
with open('wiki.train.tokens') as f:
    for line in f:
        index += 1
        #if index == 10:
        #    break
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
minibatch_size = 200
print minibatch_size

clist = [chunks(l, sequence_length) for l in lines]

for chunks in clist:
    for chunk in chunks:
        x.append([vocab_mapping[word] for word in chunk])
        del chunk[0]
        y.append([vocab_mapping[word] for word in chunk])


print len(x) / minibatch_size
print len(vocabulary)

input_x = tf.placeholder(tf.int32, shape=(None, sequence_length), name="input_x")
input_y = tf.placeholder(tf.int64, shape=(None, sequence_length - 1), name="input_y")
rs = tf.placeholder(tf.int64, shape=(None), name="random_sample")

# Embedding layer
#Wembedding = tf.Variable(tf.random_normal([vocab_size, embedding_size], -1, 1), name="Wembedding")
Wembedding= tf.Variable(tf.random_normal([vocab_size, embedding_size], stddev=.1), name="Wembedding")
embedded_words = tf.nn.embedding_lookup(Wembedding, input_x)
embedded_words_expanded = tf.expand_dims(embedded_words, 1) # give it height of 1
print embedded_words_expanded
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

    #h = tf.squeeze(h)

    #if not last_layer:
    #    h = tf.expand_dims(h, -1)

    return h


# [filter_height, filter_width, in_channels, out_channels]
filter_shape = [1, 3, 128, 1]
h1 = glu(filter_shape, embedded_words_expanded, 1)

filter_shape = [1, 3, 128, 1]
h1a = glu(filter_shape, h1, 2)

filter_shape = [1, 3, 128, 1]
h2 = glu(filter_shape, h1a, 2)

filter_shape = [1, 5, 128, 1]
h3 = glu(filter_shape, h2, 3)

filter_shape = [1, 5, 128, 2]
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

output_embedding_size = 256
filter_shape = [1, 5, 256, 1]
h10 = glu(filter_shape, h4, 10, last_layer=True)
print h10
print "hi"
# Remove the first element, we don't predict it.
h10 = tf.slice(h10, [0, 0, 1, 0], [-1, -1, -1, -1])
#print h10
h10 = tf.squeeze(h10)

#print h10

#o1 = tf.Print(h10, [h10], summarize=256, message="hidden")


def compute_loss(hidden):
    """ Softmax """
    # Calculate total with a sample
    #total = tf.reduce_sum(tf.map_fn(lambda output_slice: tf.exp(tf.reduce_sum(tf.mul(hidden, output_slice))), sampled_output_embedding))

    # Calculate actual loss from the full vocab
    output = tf.exp(tf.reduce_sum(tf.mul(hidden, tf.nn.embedding_lookup(output_embedding, input_y) + output_bias)))
    return -tf.log(output)

# Init weights, bias, (all zeros first)
print "init output embedding"
output_embedding = tf.Variable(tf.random_normal([vocab_size, output_embedding_size], stddev=.01), name="output_embedding")
#o2 = tf.Print(tf.nn.embedding_lookup(output_embedding, input_y), [tf.nn.embedding_lookup(output_embedding, input_y)], summarize=1000, message="output embedding")
#output_embedding = tf.Variable(tf.zeros([vocab_size, output_embedding_size], name="output_embedding"))
output_bias = tf.Variable(tf.fill([output_embedding_size], -7.0), name="output_bias")
#o3 = tf.Print(output_bias, [output_bias], summarize=1000, message="output bias")
#sampled_output_embedding = tf.gather(output_embedding, rs)

# Compute average loss across minibatch
print "init losses"
losses = tf.map_fn(lambda x: compute_loss(x), h10)
print "init loss"
loss = tf.reduce_mean(losses)
o = tf.Print(losses, [losses], summarize=128, message="loss")

# Trainer
print "init optimizer"
optimizer = tf.train.MomentumOptimizer(.5, .99)
print "init gradients"
gvs = optimizer.compute_gradients(loss, gate_gradients=0)
print "init clipper"
capped_gvs = [(tf.clip_by_value(grad, -.1, .1), var) for grad, var in gvs]
print "init train step"
train_step = optimizer.apply_gradients(capped_gvs)
print "inited."

saver = tf.train.Saver()
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
print "initializing..."
tf.global_variables_initializer().run()
print "initialized."

def run():
    for minibatch in range(0, 100000):
        print minibatch
        m_x = []
        m_y = []
        for x_i in range(0, minibatch_size):
            if len(x) == 0:
                return

            index = random.randrange(len(x))
            m_x.append(x[index])
            m_y.append(y[index])
            del x[index]
            del y[index]

        m_x = np.array(m_x)
        m_y = np.array(m_y)

        random_sample = np.arange(vocab_size)
        np.random.shuffle(random_sample)
        sample_size = int(np.ceil(vocab_size * .2))
        random_sample = random_sample[:sample_size]

        sess.run([train_step, o], feed_dict={input_x: m_x, input_y: m_y, rs: random_sample})

        if minibatch % 20 == 0:
            saver.save(sess, 'model.ckpt', global_step=minibatch)

run()
# to test perpleity, feed each sequence into the model and then add up the final scores.
# output: not one word, n-1 words. so you can pad. don't need to reuce dimensionarliy of sequence length.