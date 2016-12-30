import tensorflow as tf
import numpy as np
from collections import defaultdict

#https://arxiv.org/pdf/1611.09482v1.pdf
# https://arxiv.org/pdf/1612.08083v1.pdf
#https://arxiv.org/pdf/1308.0850v5.pdf


corpus = "Smith served as the news editor for 1UP.com , the sister site to the now @-@ defunct magazine Electronic Gaming Monthly . During his tenure at 1UP , Smith wrote extensively about the Halo video game franchise , as was considered a leading fan voice ; in one article , he declared Halo was the only game series he cared about . Smith wrote a feature story for 1UP in 2005 called  Broken Halo  , in which he explained how developer Bungie could fix problems he perceived with the game ; Crecente said the article put Smith  on the map  . Smith also became one of the panelists of the 1UP Yours show , a weekly video games podcast featuring gaming editors and experts . In 2006 , Edge named him one of gaming s top 50 journalists . <unk> magazine credited Smith with inspiring gamers to learn more about the game industry and not accept company promotion , as well as turning 1UP from  the bastard child of EGM  to an important part of the Ziff Davis Internet company s gaming network . His style has been described as a  robust , direct approach  to journalism and is known for his scathing attacks on the industry . Smith , however , felt disheartened by the state of game journalism .  Video game journalism is just weird . You have guys married to women in marketing for the games they cover . Video game journalism is still very young , very early , still trying to find out what it is ,  he said . In an interview with Michael Zenke of The Escapist , Smith said he felt game journalists were treated by developers as another part of the PR plan , with developers sending out information and the journalists  regurgitating  it . Worse , Smith felt that gamers had become used to this sort of information ;  We have to be responsible for our actions and held accountable when we manipulate the expectations of gamers ,  he told Zenke . While he was becoming more frustrated with the field at 1UP , game developer Bungie contacted Smith about employment . After sending the company his resume , Smith stopped writing about Bungie and Microsoft to avoid a conflict of interest . Smith accepted a job offer a month later ."
corpus = corpus.split(' ')

# Placeholders for input, output and dropout
sequence_length = 1
embedding_size = 128
vocabulary = set(corpus)
vocab_mapping = {i:x for x, i in enumerate(vocabulary)}
vocab_size = len(vocabulary)
block_size = 3
num_units = 128
layers = 128
kernel_width = 4

input_x = tf.placeholder(tf.int32, shape=(None, sequence_length), name="input_x")
input_y = tf.placeholder(tf.int32, shape=(None, vocab_size), name="input_y")

# [batch, in_height, in_width, in_channels]
input_shape = [None, sequence_length, embedding_size, 1]

# [filter_height, filter_width, in_channels, out_channels]
filter_shape = [2, embedding_size, 1, 1]

# Embedding layer
with tf.device('/cpu:0'), tf.name_scope("embedding"):

    W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="Wembedding")
    
    # (?, 20, 128)
    embedded_words = tf.nn.embedding_lookup(W, input_x)
    
    # (?, 20, 128, 1)
    embedded_words_expanded = tf.expand_dims(embedded_words, -1)

    # embedded_output = tf.nn.embedding_lookup(W, input_y)
    # embedded_output_expanded = tf.expand_dims(embedded_output, -1)


W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
conv1 = tf.nn.conv2d(
    embedded_words_expanded,
    W,
    strides=[1, 1, 1, 1],
    padding="SAME", # todo: http://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
    name="conv1")

conv1 = tf.nn.bias_add(conv1, b)

V = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="V")
c = tf.Variable(tf.constant(0.1, shape=[1]), name="c")
conv2 = tf.nn.conv2d(
    embedded_words_expanded,
    V,
    strides=[1, 1, 1, 1],
    padding="SAME", # todo: http://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
    name="conv2")
conv2 = tf.sigmoid(tf.nn.bias_add(conv2, c))

h = tf.multiply(conv1, conv2)

# TODO: Add the residual every 5 units
#if (i % 5 == 0):
#    gated = tf.add(gated, embedded_words_expanded)

# Activation
h = tf.nn.relu(h, name="relu")

#print h
i = 1
conv1 = tf.nn.conv2d(
    h,
    tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%s" % i),
    strides=[1, 1, 1, 1],
    padding="SAME", # todo: http://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
    name="conv1%s" % i)
conv1 = tf.nn.bias_add(conv1, tf.Variable(tf.constant(0.1, shape=[1]), name="b%s" % i))

conv2 = tf.nn.conv2d(
    h,
    tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="V%s" % i),
    strides=[1, 1, 1, 1],
    padding="SAME", # todo: http://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
    name="conv2%s" % i)
conv2 = tf.sigmoid(tf.nn.bias_add(conv2, tf.Variable(tf.constant(0.1, shape=[1]), name="c%s" % i)))

# TODO: Add the residual every 5 units
#if (i % 5 == 0):
#    gated = tf.add(gated, embedded_words_expanded)

# Activation
h = tf.nn.relu(tf.multiply(conv1, conv2), name="relu")

# use a softmax layer on h to get the embedding of the next word

# Final (unnormalized) scores and predictions
# with tf.name_scope("output"):
#     W = tf.get_variable(
#         "W",
#         shape=[num_units, vocab_size],
#         initializer=tf.contrib.layers.xavier_initializer())
#     b = tf.Variable(tf.constant(0.1, shape=[vocab_size]), name="b")
#     scores = tf.nn.xw_plus_b(h_pool_flat, W, b, name="scores")

# CalculateMean cross-entropy loss
print h
print input_y

# softmax on h?
#prediction = tf.nn.softmax(h)


# Init weights, bioas, (all zeros first) 
W = tf.Variable(tf.zeros([vocab_size, embedding_size]))

# define softmax function
# first multiply x and w, then add b vector. apply softmax to get probabilities
#y = tf.nn.softmax(tf.matmul(h, W))

total = 0
for i in W:
    total += exp(h * W[i])

divisor = 1 / total
output = [exp(h * W[i]) / total for i in W]

loss = 

losses = tf.nn.softmax_cross_entropy_with_logits(h, input_y)
loss = tf.reduce_mean(losses)
#loss_summary = tf.summary.scalar('loss', loss)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for index in range(30, 300):
    x = corpus[index - sequence_length:index]
    x = np.array(
    [
        [vocab_mapping[word] for word in x]
    ])

    y = [0 for v in range(0, vocab_size)]
    y[vocab_mapping[corpus[index + 1]]] = 1
    y = np.array([y])
    _, l = sess.run([train_step, loss], feed_dict={input_x: x, input_y: y})
    print l