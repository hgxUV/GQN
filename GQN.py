import tensorflow as tf
from data_reader import DataReader
import matplotlib.pyplot as plt
import datetime
import os

SIGMA = 1
BATCH_SIZE = 20
L = 12
n_reg_features = 256
EPOCHS = 100
RECORDS_IN_TF_RECORD = 5000
TF_RECORDS = 20


#TODO ask someone qualified about these global thingies
gen_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[16, 16, 519], output_channels=256, kernel_shape=[5, 5], name='gen_cell')  # TODO: output channels, skip connection?
inf_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[16, 16, 775], output_channels=256, kernel_shape=[5, 5], name='inf_cell')  # TODO: output channels, skip connection?
training = False


def conv_block(prev, size, k: tuple, s: tuple):
    size_policy = 'same' if s == (1, 1) else 'valid'
    after_conv = tf.layers.conv2d(prev, size, k, s, size_policy)
    return tf.nn.relu(after_conv)


# x shape: (1, 64, 64, 3), v shape: (1, 7)
def representation_pipeline_tower(x, v=None, representation=True):
    if(representation):
        v = tf.broadcast_to(tf.expand_dims(tf.expand_dims(v, 1), 1), (x.shape[0], 16, 16, 7))

    x = conv_block(x, 256, (2, 2), (2, 2))

    # first residual
    y = conv_block(x, 128, (3, 3), (1, 1))
    x = tf.concat([x, y], 3)
    x = conv_block(x, 256, (2, 2), (2, 2))

    # add v
    if(representation):
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

    frames = tf.reshape(frames, (-1, 64, 64, 3))
    cameras = tf.reshape(cameras, (-1, 7))

    r = representation_pipeline_tower(frames, cameras)

    r = tf.reshape(r, (BATCH_SIZE, 5, 16, 16, 256))
    r = tf.math.reduce_sum(r, axis=1)

    return r


def sample_gaussian(mu, sigma=1.):
    sampled = tf.random_normal((), mean=0., stddev=1.)
    return tf.multiply(tf.math.add(mu, sampled), sigma)

def prior_posterior(h_i):
    gaussianParams = conv_block(h_i, 2*n_reg_features, (5, 5), (1, 1))
    means = gaussianParams[:, :, :, 0:n_reg_features]
    stds = gaussianParams[:, :, :, n_reg_features:]
    stds = tf.nn.softmax(stds)
    gaussianParams = tf.concat([means, stds], -1)
    distributions = tf.distributions.Normal(loc=means, scale=stds)
    latent = distributions.sample()
    return (latent, gaussianParams)

def recon_loss(x_true, x_pred):
    rec_loss = tf.losses.mean_squared_error(x_true, x_pred)
    tf.summary.scalar('reconstruction_loss', rec_loss)
    return rec_loss

def distribution_loss(prior, posterior):
    prior = prior.concat()
    posterior = posterior.concat()
    distritution = lambda x : tf.distributions.Normal(loc=x[:, :, :, 0:n_reg_features],
                                                      scale=x[:, :, :, n_reg_features:])
    prior = distritution(prior)
    posterior = distritution(posterior)
    dist_loss = tf.distributions.kl_divergence(posterior, prior)
    dist_loss = tf.reduce_mean(dist_loss)
    tf.summary.scalar('distribution_loss', dist_loss)
    return dist_loss

def calculate_loss(priors, posteriors, x_pred, x_q):
    loss = tf.add(recon_loss(x_q, x_pred), distribution_loss(priors, posteriors))
    tf.summary.scalar('loss', loss)
    return loss

def image_reconstruction(u):
    x = conv_block(u, 3, (1, 1), (1, 1))
    stds = tf.multiply(tf.ones(x.shape), SIGMA)
    dist = tf.distributions.Normal(loc=x, scale=stds)
    x_pred = dist.sample()
    return x_pred


def body(state_g, u, r, v_q, x_q, state_i, priors, posteriors, i):

    #inference
    if(training):
        concat_i = tf.concat([x_q,
                              tf.broadcast_to(v_q, (v_q.shape[0], 16, 16, v_q.shape[-1])),
                              r,
                              state_g.h],
                             3)
        h_i, state_i = inf_cell(concat_i, state_i)

        posterior_latent, posterior_params = prior_posterior(state_i.h)
        posteriors = posteriors.write(i, posterior_params)

    #generation
    prior_latent, prior_params = prior_posterior(state_g.h)
    priors = priors.write(i, prior_params)

    if(training):
        prior_latent = posterior_latent

    concat_g = tf.concat([v_q,
                          r,
                          prior_latent], 3)
    h_g, state_g = gen_cell(concat_g, state_g)
    u = tf.math.add(tf.layers.conv2d_transpose(h_g, 256, 4, 4, 'SAME'), u)

    i += 1

    return state_g, u, r, v_q, x_q, state_i, priors, posteriors, i

def cond(state_g, u, r, v_q, x_q, state_i, priors, posteriors, i):
    dec = tf.less(i, L)
    return dec


def generative_query_network(x, v, v_q, x_q, training_local):
    #with tf.variable_scope('gqn', reuse=tf.AUTO_REUSE):

    global training
    training = training_local
    x_gt = x_q

    state_i = inf_cell.zero_state(BATCH_SIZE, dtype='float32') # c + h
    state_g = gen_cell.zero_state(BATCH_SIZE, dtype='float32') # c + h

    u = tf.zeros([x.shape[0], 64, 64, 256])
    r = create_representation(x, v)
    i = 0
    x_q = representation_pipeline_tower(x_q, representation=False)
    v_q = tf.broadcast_to(tf.expand_dims(tf.expand_dims(v_q, 1), 1), (x.shape[0], 16, 16, 7))

    priors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[x.shape[0], 16, 16, n_reg_features*2])#,  clear_after_read=False)
    posteriors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[x.shape[0], 16, 16, n_reg_features*2])#,  clear_after_read=False)

    variables = (state_g, u, r, v_q, x_q, state_i, priors, posteriors, i)

    variables = tf.while_loop(cond, body, variables, parallel_iterations=1)

    state_g, u, r, v_q, x_q, state_i, priors, posteriors, i = variables

    x_pred = image_reconstruction(u)
    loss = calculate_loss(priors, posteriors, x_pred, x_gt)

    return x_pred, loss

tf.reset_default_graph()

root_path = 'data'
data_reader = DataReader(dataset='rooms_ring_camera', context_size=5, root=root_path)

optimizer = tf.train.AdamOptimizer()
data = data_reader.read(batch_size=BATCH_SIZE)
x = data.query.context.frames
v = data.query.context.cameras
x_q = data.target
v_q = data.query.query_camera


train_output, train_loss = generative_query_network(x, v, v_q, x_q, training_local=True)
train_op = optimizer.minimize(train_loss)

merged_summaries = tf.summary.merge_all()
now = datetime.datetime.now()
current_data = now.strftime("%Y-%m-%d-%H-%M")
path = os.path.join('tensorboard', current_data)
os.makedirs(os.path.join(path, 'test'))
train_writer = tf.summary.FileWriter(os.path.join('tensorboard', current_data, 'test'))


with tf.train.SingularMonitoredSession() as sess:
    train_writer.add_graph(sess.graph)
    for i in range(EPOCHS):
        total_loss = []
        j = 0
        for j in range(int((TF_RECORDS * RECORDS_IN_TF_RECORD) / BATCH_SIZE)):
            x_pred, x_gt, loss_value, summary = sess.run([train_output, x_q, train_loss, merged_summaries])  # , feed_dict={x: x, v: v, v_q: v_q, x_q: x_q})
            train_writer.add_summary(summary, j)
            #x_gt = sess.run(x_q)
            total_loss.append(loss_value)
            print(loss_value)
            #plt.imshow(x_gt[0, ...])
            #plt.show()
            #plt.imshow(x_pred[0, ...])
            #plt.show()
            j += 1
            print(j)
        print(j)
        print('Total epoch {0} loss: {1}'.format(i, sum(total_loss) / len(total_loss)))
