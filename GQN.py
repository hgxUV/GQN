import tensorflow as tf
from data_reader import DataReader
import matplotlib.pyplot as plt


def conv_block(prev, size, k: tuple, s: tuple):
    size_policy = 'same' if s == (1, 1) else 'valid'
    after_conv = tf.layers.conv2d(prev, size, k, s, size_policy)
    return tf.nn.relu(after_conv)


# x shape: (1, 64, 64, 3), v shape: (1, 7)
def representation_pipeline_tower(x, v):
    assert x.shape == (1, 64, 64, 3)
    assert v.shape == (1, 7)

    input_v = tf.broadcast_to(tf.expand_dims(v, 0), (1, 16, 16, 7))

    test = conv_block(x, 256, (2, 2), (2, 2))

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


# frames shape: (5, 64, 64, 3), v shape: (5, 7)
def create_representation(frames, cameras):
    assert frames.shape[0] == cameras.shape[0]
    assert len(frames.shape) == 4
    assert len(cameras.shape) == 2

    iterations = frames.shape[0]
    r = [None] * iterations

    for i in range(iterations):
        r[i] = representation_pipeline_tower(tf.expand_dims(frames[i], 0), tf.expand_dims(cameras[i], 0))

    ret = tf.zeros(r[0].shape)

    for rr in r:
        ret = tf.math.add(ret, rr)

    return ret


def sample_gaussian(mu, sigma=1.):
    sampled = tf.random_normal((), mean=0., stddev=1.)
    return tf.multiply(tf.math.add(mu, sampled), sigma)

def prior_posterior(h_i, n_features):
    gaussianParams = conv_block(h_i, 2*n_features, (5, 5), (1, 1))
    means = gaussianParams[:, :, :, 0:n_features]
    stds = gaussianParams[:, :, :, n_features:]
    stds = tf.nn.softmax(stds)
    gaussianParamsTuple = (means, stds)
    latent = tf.map_fn(lambda stats : sample_gaussian(stats[0], stats[1]), gaussianParamsTuple, dtype=tf.float32)
    return latent, gaussianParams

def recon_loss(x_true, x_pred):
    tf.sigmoid_cross_entropy_with_logits(labels=x_true, logits=x_pred)

def regularization_loss(prior, posterior):
    pass

def loss():
    return recon_loss + regularization_loss

def observation_sample(u_L):
    means = conv_block(u_L, 3, (1, 1), (1, 1))
    x = tf.map_fn(lambda mean : sample_gaussian(mean), means, dtype=tf.float32)
    return x

def image_reconstruction(sampled):
    pass
    #dunno how to to that
    #x = tf.layers.conv2d_transpose(sampled, 128, 3, 16, 'SAME')


def lstm_cell(concat, c):
    forget = tf.nn.sigmoid(conv_block(concat, 256, (5, 5), (1, 1)))
    input = tf.nn.sigmoid(conv_block(concat, 256, (5, 5), (1, 1)))
    cell = tf.nn.tanh(conv_block(concat, 256, (5, 5), (1, 1)))
    output = tf.nn.sigmoid(conv_block(concat, 256, (5, 5), (1, 1)))

    c = tf.math.multiply(c, forget)
    c = tf.math.add(c, tf.math.multiply(input, cell))

    h = tf.math.multiply(output, tf.nn.tanh(c))

    return h, c

def body(h_g, c_g, u_g, r, v_q, x_q, h_i, c_i, i, ta):

    prior, prior_params = prior_posterior(h_g, 256)

    #generation
    concat_g = tf.concat([h_g, v_q, r, prior], 3)
    h_g, c_g = lstm_cell(concat_g, c_g)
    u_g = tf.math.add(tf.layers.conv2d_transpose(h_g, 256, 4, 4, 'SAME'), u_g)

    #inference
    concat_i = tf.concat([h_i, v_q, x_q], 3)
    h_i, c_i = lstm_cell(concat_i, c_i)
    posterior, posterior_params = prior_posterior(h_i, 256)

    return h_g, c_g, u_g, r, v_q, x_q, h_i, c_i, i, ta


def training_loop(x, v, v_q, x_q):
    h_g = tf.zeros([1, 16, 16, 256], 0, 1)
    c_g = tf.zeros([1, 16, 16, 256], 0, 1)
    u_g = tf.zeros([1, 64, 64, 256], 0, 1)

    r = tf.random_normal([1, 16, 16, 256], 0, 1)

    h_i = tf.zeros([1, 16, 16, 256], 0, 1)
    c_i = tf.zeros([1, 16, 16, 256], 0, 1)

    v_q = tf.random_normal([1, 16, 16, 7], 0, 1)
    x_q = tf.random_normal([1, 16, 16, 256], 0, 1)

    stuff = body(h_g, c_g, u_g, r, v_q, x_q, h_i, c_i, 12)

    return stuff

root_path = 'data'
data_reader = DataReader(dataset='rooms_ring_camera', context_size=5, root=root_path)
data = data_reader.read(batch_size=1)
print(data[1])

#xd = representation_pipeline_tower(data[1], data[0][1])
#someTensor = tf.random_normal([1, 16, 16, 256], 0, 1)
#test = prior_posterior(someTensor, someTensor.shape[-1])
#u_L = tf.random_normal([1, 64, 64, 256], 0, 1)
#output_images = observation_sample(u_L)
#output_images = tf.clip_by_value(output_images, 0, 1)

stuff = training_loop(1, 2, 3, 4)

with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(output_images)
    #plt.imshow(d[0, :, :, :])
    #plt.show()

a = 1
