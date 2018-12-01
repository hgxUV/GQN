import tensorflow as tf
from data_reader import DataReader
import matplotlib.pyplot as plt

SIGMA = 1
BATCH_SIZE = 10
L = 12
n_reg_features = 256
EPOCHS = 10


gen_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[16, 16, 266], output_channels=256, kernel_shape=[5, 5], name='gen_cell')  # TODO: output channels, skip connection?
inf_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[64, 64, 1066], output_channels=256, kernel_shape=[5, 5], name='inf_cell')  # TODO: output channels, skip connection?


def conv_block(prev, size, k: tuple, s: tuple):
    size_policy = 'same' if s == (1, 1) else 'valid'
    after_conv = tf.layers.conv2d(prev, size, k, s, size_policy)
    return tf.nn.relu(after_conv)


# x shape: (1, 64, 64, 3), v shape: (1, 7)
def representation_pipeline_tower(x, v):
    v = tf.broadcast_to(tf.expand_dims(tf.expand_dims(v, 1), 1), (x.shape[0], 16, 16, 7))

    x = conv_block(x, 256, (2, 2), (2, 2))

    # first residual
    y = conv_block(x, 128, (3, 3), (1, 1))
    x = tf.concat([x, y], 3)
    x = conv_block(x, 256, (2, 2), (2, 2))

    # add v
    x = tf.concat([x, v], 3)

    # second residual
    y = conv_block(x, 128, (3, 3), (1, 1))
    x = tf.concat([x, y], 3)
    x = conv_block(x, 256, (3, 3), (1, 1))

    # last conv
    return conv_block(x, 256, (1, 1), (1, 1))


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

def prior_posterior(h_i):
    gaussianParams = conv_block(h_i, 2*n_reg_features, (5, 5), (1, 1))
    means = gaussianParams[:, :, :, 0:n_reg_features]
    stds = gaussianParams[:, :, :, n_reg_features:]
    stds = tf.nn.softmax(stds)
    distributions = tf.distributions.Normal(loc=means, scale=stds)
    latent = distributions.sample()
    return (latent, gaussianParams)

def recon_loss(x_true, x_pred):
    rec_loss = tf.losses.mean_squared_error(x_true, x_pred)
    return rec_loss

def regularization_loss(prior, posterior):
    prior = prior.concat()
    posterior = posterior.concat()
    distritution = lambda x : tf.distributions.Normal(loc=x[:, :, :, 0:n_reg_features],
                                                      scale=x[:, :, :, n_reg_features:])
    prior = distritution(prior)
    posterior = distritution(posterior)
    reg_loss = tf.distributions.kl_divergence(posterior, prior)
    reg_loss = tf.reduce_sum(reg_loss)
    return reg_loss

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


def body(state_g, u_g, r, v_q, x_q, state_i, priors, posteriors, i):
    #generation
    prior_latent, prior_params = prior_posterior(state_g.h)
    priors = priors.write(i, prior_params)

    concat_g = tf.concat([v_q, r, prior_latent], 3)
    h_g, state_g = gen_cell(concat_g, state_g)
    u_g = tf.math.add(tf.layers.conv2d_transpose(h_g, 256, 4, 4, 'SAME'), u_g)

    #inference
    concat_i = tf.concat([tf.broadcast_to(v_q, (v_q.shape[0], 64, 64, v_q.shape[-1])), x_q], 3)
    h_i, state_i = inf_cell(concat_i, state_i)

    #TODO find out what to do with posterior_latent
    posterior_input = tf.layers.max_pooling2d(h_i, (4, 4), (4, 4), padding='SAME')
    posterior_latent, posterior_params = prior_posterior(posterior_input)
    posteriors = posteriors.write(i, posterior_params)

    i += 1

    return state_g, u_g, r, v_q, x_q, state_i, priors, posteriors, i

def cond(state_g, u_g, r, v_q, x_q, state_i, priors, posteriors, i):
    dec = tf.less(L, i)
    return dec


def architecture(x, v, v_q, x_q):
    state_g = gen_cell.zero_state(BATCH_SIZE, dtype='float32') # c + h
    u_g = tf.zeros([x.shape[0], 64, 64, 256])

    r = create_representation(x, v)

    state_i = inf_cell.zero_state(BATCH_SIZE, dtype='float32') # c + h

    i = 0
    priors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[x.shape[0], 16, 16, n_reg_features*2])
    posteriors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[x.shape[0], 16, 16, n_reg_features*2])

    v_q = tf.broadcast_to(tf.expand_dims(tf.expand_dims(v_q, 1), 1), (x.shape[0], 16, 16, 7))

    variables = (state_g, u_g, r, v_q, x_q, state_i, priors, posteriors, i)
    #variables = body(*variables)

    variables = tf.while_loop(cond, body, variables)
    state_g, u_g, r, v_q, x_q, state_i, priors, posteriors, i = variables

    x_pred = image_reconstruction(u_g)
    #TODO reconstruction loss, distribution loss
    loss = calculate_loss(priors, posteriors, x_pred, x_q)

    return x_pred, loss

tf.reset_default_graph()

root_path = 'data'
data_reader = DataReader(dataset='rooms_ring_camera', context_size=5, root=root_path)
#xd = representation_pipeline_tower(data[1], data[0][1])
#someTensor = tf.random_normal([1, 16, 16, 256], 0, 1)
#test = prior_posterior(someTensor, someTensor.shape[-1])
#u_L = tf.random_normal([1, 64, 64, 256], 0, 1)
#output_images = observation_sample(u_L)
#output_images = tf.clip_by_value(output_images, 0, 1)

optimizer = tf.train.AdamOptimizer()
data = data_reader.read(batch_size=BATCH_SIZE)
x = data.query.context.frames
v = data.query.context.cameras
x_q = data.target
v_q = data.query.query_camera

#x = tf.placeholder(tf.float32, [BATCH_SIZE, 5, 64, 64, 3], 'x')
#v = tf.placeholder(tf.float32, [BATCH_SIZE, 5, 7], 'v')
#v_q = tf.placeholder(tf.float32, [BATCH_SIZE, 7], 'v_q')
#x_q = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], 'x_q')
image, loss = architecture(x, v, v_q, x_q)
train_op = optimizer.minimize(loss)

with tf.train.SingularMonitoredSession() as sess:
    for i in range(EPOCHS):
        for _ in range(4):
            image, train_op = sess.run([image, train_op])#, feed_dict={x: x, v: v, v_q: v_q, x_q: x_q})
            print(train_op)
            plt.imshow(image[0, ...])
            plt.show()

a = 1
