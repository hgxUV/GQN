import tensorflow as tf
from data_reader import DataReader
import matplotlib.pyplot as plt

root_path = 'data'
data_reader = DataReader(dataset='jaco', context_size=5, root=root_path)
data = data_reader.read(batch_size=12)

with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(data)
    for img in d.target:
        plt.imshow(img)
        plt.show()