import tensorflow as tf
from data_reader import DataReader
import matplotlib.pyplot as plt


def conv_block(prev, size, k: tuple, s: tuple):
    size_policy = 'same' if s == (1, 1) else 'valid'
    return tf.layers.conv2d(prev, size, k, s, size_policy, activation=tf.nn.relu)


root_path = 'data'
data_reader = DataReader(dataset='jaco', context_size=5, root=root_path)
data = data_reader.read(batch_size=1)


def generative_pipeline_tower(x, v):
    input_img = x
    input_v = tf.broadcast_to(v, (1, 16, 16, 7))

    test = conv_block(input_img, 256, (2, 2), (2, 2))

    # first residual
    test2 = conv_block(test, 128, (3, 3), (1, 1))
    test3 = tf.concat([test, test2], 3)
    test4 = conv_block(test3, 256, (2, 2), (2, 2))

    # add v
    test5 = tf.concat([test4, input_v], 3)

    # second residual
    test6 = conv_block(test5, 128, (3, 3), (1, 1))
    test7 = tf.concat([test6, test5], 3)
    test8 = conv_block(test7, 256, (3, 3), (1, 1))

    # last conv
    return conv_block(test8, 256, (1, 1), (1, 1))


xd = generative_pipeline_tower(data[1], data[0][1])

with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(xd)
    print(data[0])

