import tensorflow as tf
from data_reader import DataReader
import matplotlib.pyplot as plt

SIGMA = 1
BATCH_SIZE = 10


def conv_block(prev, size, k: tuple, s: tuple):
    size_policy = 'same' if s == (1, 1) else 'valid'
    after_conv = tf.layers.conv2d(prev, size, k, s, size_policy)
    return tf.nn.relu(after_conv)


# x shape: (1, 64, 64, 3), v shape: (1, 7)
def representation_pipeline_tower(x, v):
    #assert x.shape == (1, 64, 64, 3)
    #assert v.shape == (1, 7)

    input_v = tf.broadcast_to(tf.expand_dims(tf.expand_dims(v, 1), 1), (x.shape[0], 16, 16, 7))

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
    #assert frames.shape[0] == cameras.shape[0]
    #assert len(frames.shape) == 4
    #assert len(cameras.shape) == 2

    iterations = frames.shape[1]
    r = [None] * iterations

    for i in range(iterations):
        r[i] = representation_pipeline_tower(frames[:, i, :, :, :], cameras[:, i, :])

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
    distributions = tf.distributions.Normal(loc=means, scale=stds)
    latent = distributions.sample()
    return (latent, gaussianParams)

def recon_loss(x_true, x_pred):
    tf.sigmoid_cross_entropy_with_logits(labels=x_true, logits=x_pred)

def regularization_loss(prior, posterior):
    #TODO this stuff here
    pass

def calculate_loss(priors, posteriors, x_pred, x_q):
    return recon_loss(x_q, x_pred) + regularization_loss(priors, posteriors)

def observation_sample(u_L):
    means = conv_block(u_L, 3, (1, 1), (1, 1))
    x = tf.map_fn(lambda mean : sample_gaussian(mean), means, dtype=tf.float32)
    return x

def image_reconstruction(u):
    x = conv_block(u, 3, (1, 1), (1, 1))
    stds = tf.multiply(tf.ones(x.shape), SIGMA)
    dist = tf.distributions.Normal(loc=x, scale=stds)
    x_pred = dist.sample()
    return x_pred


def lstm_cell(concat, c):
    forget = tf.nn.sigmoid(conv_block(concat, 256, (5, 5), (1, 1)))
    input = tf.nn.sigmoid(conv_block(concat, 256, (5, 5), (1, 1)))
    cell = tf.nn.tanh(conv_block(concat, 256, (5, 5), (1, 1)))
    output = tf.nn.sigmoid(conv_block(concat, 256, (5, 5), (1, 1)))

    c = tf.math.multiply(c, forget)
    c = tf.math.add(c, tf.math.multiply(input, cell))

    h = tf.math.multiply(output, tf.nn.tanh(c))

    return h, c

def body(h_g, c_g, u_g, r, v_q, x_q, h_i, c_i, n_reg_features, priors, posteriors, i):

    #generation
    prior_latent, prior_params = prior_posterior(h_g, n_reg_features)
    priors = priors.concat(prior_params)

    concat_g = tf.concat([h_g, v_q, r, prior_latent], 3)
    h_g, c_g = lstm_cell(concat_g, c_g)
    u_g = tf.math.add(tf.layers.conv2d_transpose(h_g, 256, 4, 4, 'SAME'), u_g)

    #inference
    concat_i = tf.concat([h_i, v_q, x_q], 3)
    h_i, c_i = lstm_cell(concat_i, c_i)
    #TODO find out what to do with posterior_latent
    posterior_latent, posterior_params = prior_posterior(h_i, n_reg_features)
    posteriors.concat(posterior_params)

    return h_g, c_g, u_g, r, v_q, x_q, h_i, c_i, n_reg_features, priors, posteriors, i


def architecture(x, v, v_q, x_q):
    h_g = tf.zeros([x.shape[0], 16, 16, 256])
    c_g = tf.zeros([x.shape[0], 16, 16, 256])
    u_g = tf.zeros([x.shape[0], 64, 64, 256])

    #TODO use tower architecture here
    r = create_representation(x, v)

    h_i = tf.zeros([x.shape[0], 16, 16, 256])
    c_i = tf.zeros([x.shape[0], 16, 16, 256])

    n_reg_features = 256
    i = 0
    priors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[None, 16, 16, n_reg_features])
    posteriors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[None, 16, 16, n_reg_features])

    variables = (h_g, c_g, u_g, r, v_q, x_q, h_i, c_i, n_reg_features, priors, posteriors, i)

    stuff = body(*variables)
    cond = lambda variables : tf.less(12, i)
    variables = tf.while_loop(stuff, cond, variables)
    h_g, c_g, u_g, r, v_q, x_q, h_i, c_i, n_reg_features, priors, posteriors, i = variables

    x_pred = image_reconstruction(u_g)
    #TODO reconstruction loss, distribution loss
    loss = calculate_loss(priors, posteriors, x_pred, x_q)

    return x_pred, loss

root_path = 'data'
data_reader = DataReader(dataset='rooms_ring_camera', context_size=5, root=root_path)
data = data_reader.read(batch_size=BATCH_SIZE)
x = data.query.context.frames
v = data.query.context.cameras
x_q = data.target
v_q = data.query.query_camera

#xd = representation_pipeline_tower(data[1], data[0][1])
#someTensor = tf.random_normal([1, 16, 16, 256], 0, 1)
#test = prior_posterior(someTensor, someTensor.shape[-1])
#u_L = tf.random_normal([1, 64, 64, 256], 0, 1)
#output_images = observation_sample(u_L)
#output_images = tf.clip_by_value(output_images, 0, 1)

stuff = architecture(x, v, v_q, x_q)

with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(stuff)
    #plt.imshow(d[0, :, :, :])
    #plt.show()

a = 1
