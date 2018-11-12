import tensorflow as tf
from data_reader import DataReader
import matplotlib.pyplot as plt


def conv_block(prev, size, k: tuple, s: tuple):
    size_policy = 'same' if s == (1, 1) else 'valid'
    return tf.layers.conv2d(prev, size, k, s, size_policy, activation=tf.nn.relu)


root_path = 'data'
data_reader = DataReader(dataset='jaco', context_size=5, root=root_path)
data = data_reader.read(batch_size=12)

with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(data)
    for img in d.target:
        plt.imshow(img)
        plt.show()