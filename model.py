import tensorflow as tf
import numpy as np
from collections import defaultdict

#https://arxiv.org/pdf/1611.09482v1.pdf
# https://arxiv.org/pdf/1612.08083v1.pdf
#https://arxiv.org/pdf/1308.0850v5.pdf


corpus = "Smith served as the news editor for 1UP.com , the sister site to the now @-@ defunct magazine Electronic Gaming Monthly . During his tenure at 1UP , Smith wrote extensively about the Halo video game franchise , as was considered a leading fan voice ; in one article , he declared Halo was the only game series he cared about . Smith wrote a feature story for 1UP in 2005 called  Broken Halo  , in which he explained how developer Bungie could fix problems he perceived with the game ; Crecente said the article put Smith  on the map  . Smith also became one of the panelists of the 1UP Yours show , a weekly video games podcast featuring gaming editors and experts . In 2006 , Edge named him one of gaming s top 50 journalists . <unk> magazine credited Smith with inspiring gamers to learn more about the game industry and not accept company promotion , as well as turning 1UP from  the bastard child of EGM  to an important part of the Ziff Davis Internet company s gaming network . His style has been described as a  robust , direct approach  to journalism and is known for his scathing attacks on the industry . Smith , however , felt disheartened by the state of game journalism .  Video game journalism is just weird . You have guys married to women in marketing for the games they cover . Video game journalism is still very young , very early , still trying to find out what it is ,  he said . In an interview with Michael Zenke of The Escapist , Smith said he felt game journalists were treated by developers as another part of the PR plan , with developers sending out information and the journalists  regurgitating  it . Worse , Smith felt that gamers had become used to this sort of information ;  We have to be responsible for our actions and held accountable when we manipulate the expectations of gamers ,  he told Zenke . While he was becoming more frustrated with the field at 1UP , game developer Bungie contacted Smith about employment . After sending the company his resume , Smith stopped writing about Bungie and Microsoft to avoid a conflict of interest . Smith accepted a job offer a month later ."
corpus = corpus.split(' ')

# Placeholders for input, output and dropout
sequence_length = 32
embedding_size = 128
vocabulary = set(corpus)
vocab_mapping = {i:x for x, i in enumerate(vocabulary)}
vocab_size = len(vocabulary)
block_size = 3
num_units = 128
layers = 128
kernel_width = 4

input_x = tf.placeholder(tf.int32, shape=(None, sequence_length), name="input_x")
input_y = tf.placeholder(tf.int64, shape=(), name="input_y")

# [batch, in_height, in_width, in_channels]
input_shape = [None, sequence_length, embedding_size, 1]



# Embedding layer
Wembedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="Wembedding")
embedded_words = tf.nn.embedding_lookup(Wembedding, input_x)
embedded_words_expanded = tf.expand_dims(embedded_words, -1)
#   padded = tf.Print(embedded_words_expanded, [embedded_words_expanded], summarize=1000)

def glu(filter_shape, paddings, layer_input, layer_name):
    global padded
    padded_input = tf.pad(layer_input, paddings, "CONSTANT") 
    print "padded input", padded_input
    #padded = tf.Print(padded_input, [padded_input], summarize=1000)

    center_w = filter_shape[1] // 2 
    mask = np.ones((filter_shape), dtype=np.float32)
    mask[:, center_w+1: ,: ,:] = 0.

    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=1.0/filter_shape[2]), name="W%s" % layer_name)
   # padded = tf.Print(W, [W], summarize=1000)
    W *= tf.constant(mask, dtype=tf.float32) # mask the kernel
    
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

    return h
    
# mask the kernels
# do not zero pad. zero padding retains the same size
# if mask_type is not None:

# [filter_height, filter_width, in_channels, out_channels]
# use rectangular kernels
filter_shape = [3, 3, 1, 128]
#paddings = [[0,0], [1, 0], [0, 0], [0, 0]]
paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h1 = glu(filter_shape, paddings, embedded_words_expanded, 1)
print "first layer output", h1

filter_shape = [3, 3, 128, 128]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h1a = glu(filter_shape, paddings, h1, 2)
print "second_layer output", h1a

filter_shape = [3, 3, 128, 256]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h2 = glu(filter_shape, paddings, h1a, 2)
print "second_layer output", h2

filter_shape = [4, 4, 256, 512]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h3 = glu(filter_shape, paddings, h2, 3)
print "third layer output", h3

filter_shape = [4, 4, 512, 512]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h4 = glu(filter_shape, paddings, h3, 4)
print "third layer output", h4

filter_shape = [4, 10, 512, 1024]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h5 = glu(filter_shape, paddings, h4, 5)
print "third layer output", h5

filter_shape = [4, 10, 1024, 1024]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h6 = glu(filter_shape, paddings, h5, 6)
print "third layer output", h6

filter_shape = [4, 15, 1024, 1024]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h7 = glu(filter_shape, paddings, h6, 7)
print "third layer output", h7

filter_shape = [4, 20, 1024, 1024]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h8 = glu(filter_shape, paddings, h7, 8)
print "third layer output", h8

filter_shape = [4, 30, 1024, 1024]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h9 = glu(filter_shape, paddings, h8, 9)
print "third layer output", h9

filter_shape = [5, 36, 1024, 1024]
#paddings = [[0,0], [0, 0], [0, 0], [0, 0]]
h10 = glu(filter_shape, paddings, h9, 10)
print "third layer output", h10

shaped = tf.squeeze(h10)
print shaped
#padded = tf.Print(shaped, [shaped], summarize=1000)
# TODO: Add the residual every 5 units
#if (i % 5 == 0):
#    gated = tf.add(gated, embedded_words_expanded)
## need fully connected layer

# Init weights, bioas, (all zeros first) 
softmax_w = tf.Variable(tf.zeros([vocab_size, 1024]), name="softmax_w")

# softmax
total = 0.0
print vocab_size
for i in range(0, vocab_size):
    output_embedding = tf.nn.embedding_lookup(softmax_w, i)
    print output_embedding
    total += tf.exp(tf.reduce_sum(tf.mul(shaped, output_embedding))) # dot product

# why make the whole output? just make what you want.
#output = [tf.exp(tf.reduce_sum(tf.mul(shaped, tf.nn.embedding_lookup(softmax_w, i)))) / total for i in range(0, vocab_size)]
#output = tf.constant(output)

output = tf.exp(tf.reduce_sum(tf.mul(shaped, tf.nn.embedding_lookup(softmax_w, input_y)))) / total

print output
loss = -tf.log(output)
loss_sum = tf.scalar_summary("loss", loss)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for index in range(32, 300):
    print index
    x = corpus[index - sequence_length:index]
    x = np.array(
    [
        [vocab_mapping[word] for word in x]
    ])

    # y = [0 for v in range(0, vocab_size)]
    # y[vocab_mapping[corpus[index + 1]]] = 1
    # y = np.array([y])
    writer = tf.train.SummaryWriter("log", sess.graph_def)
    _, l = sess.run([train_step, loss_sum], feed_dict={input_x: x, input_y: vocab_mapping[corpus[index + 1]]})
    write.add_summary(l)
    print("%f" % l)