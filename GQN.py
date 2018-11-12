import tensorflow as tf
from data_reader import DataReader
import matplotlib.pyplot as plt


def conv_block(prev, size, k: tuple, s: tuple):
    size_policy = 'same' if s == (1, 1) else 'valid'
    return tf.layers.conv2d(prev, size, k, s, size_policy, activation=tf.nn.relu)


root_path = 'data'
data_reader = DataReader(dataset='jaco', context_size=5, root=root_path)
data = data_reader.read(batch_size=1)

test = conv_block(data[1], 256, (2, 2), (2, 2))

with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(test)

