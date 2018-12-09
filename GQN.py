import tensorflow as tf
from data_reader import DataReader
import datetime
import os

SIGMA = 1
BATCH_SIZE = 20
L = 12
n_reg_features = 256
EPOCHS = 100
RECORDS_IN_TF_RECORD = 5000
TF_RECORDS = 20


def conv_block(prev, size, k: tuple, s: tuple, name=None):
    size_policy = 'same' if s == (1, 1) else 'valid'
    relu_name = (name + '_relu') if name != None else name
    after_conv = tf.layers.conv2d(prev, size, k, s, size_policy, name=name)
    return tf.nn.relu(after_conv, name=relu_name)


# x shape: (1, 64, 64, 3), v shape: (1, 7)
def representation_pipeline_tower(x, v):
    with tf.variable_scope("representation_tower"):

        exp_1 = tf.expand_dims(v, 1, name='exp_1')
        exp_2 = tf.expand_dims(exp_1, 1, name='exp_2')
        v = tf.broadcast_to(exp_2, (x.shape[0], 16, 16, 7), name='viewpoints_broadcast')

        x = conv_block(x, 256, (2, 2), (2, 2), name='conv1')

        # first residual
        y = conv_block(x, 128, (3, 3), (1, 1), name='conv2')
        x = tf.concat([x, y], 3, name='first_residual')
        x = conv_block(x, 256, (2, 2), (2, 2), name='conv3')

        # add v
        x = tf.concat([x, v], 3, name='viewpoint_concat')

        # second residual
        y = conv_block(x, 128, (3, 3), (1, 1), name='conv4')
        x = tf.concat([x, y], 3, name='second_residual')
        x = conv_block(x, 256, (3, 3), (1, 1), name='conv5')

        # last conv
        return conv_block(x, 256, (1, 1), (1, 1), name='conv6')


# frames shape: (5, 64, 64, 3), v shape: (5, 7)
def create_representation(frames, cameras):
    frames = tf.reshape(frames, (-1, 64, 64, 3), name='frames_reshape')
    cameras = tf.reshape(cameras, (-1, 7), name='cameras_reshape')

    r = representation_pipeline_tower(frames, cameras)

    r = tf.reshape(r, (BATCH_SIZE, 5, 16, 16, 256), name='r_rereshape')
    r = tf.math.reduce_sum(r, axis=1, name='reducing_sum_images')

    return r

def prior_posterior(h_i, name):
    gaussianParams = conv_block(h_i, 2*n_reg_features, (5, 5), (1, 1), name='conv')
    means = gaussianParams[:, :, :, 0:n_reg_features]
    stds = gaussianParams[:, :, :, n_reg_features:]
    stds = tf.nn.softmax(stds, name='std_normalization')
    gaussianParams = tf.concat([means, stds], -1, name='mean_std_concat')
    distributions = tf.distributions.Normal(loc=means, scale=stds,
                                                          name='normal_distribution')
    latent = distributions.sample()
    return latent, gaussianParams

def recon_loss(x_true, x_pred):
    with tf.variable_scope("recon_loss"):
        rec_loss = tf.losses.mean_squared_error(x_true, x_pred)
        tf.summary.scalar('reconstruction_loss', rec_loss)
        return rec_loss

def distribution_loss(prior, posterior):
    with tf.variable_scope("distribution_loss"):
        prior = prior.concat(name='prior_concat')
        posterior = posterior.concat(name='posterior_concat')
        distritution = lambda x : tf.distributions.Normal(loc=x[:, :, :, 0:n_reg_features],
                                                          scale=x[:, :, :, n_reg_features:],
                                                          name='normal_distribution')
        prior = distritution(prior)
        posterior = distritution(posterior)
        dist_loss = tf.distributions.kl_divergence(posterior, prior, name='kl_div')
        dist_loss = tf.reduce_mean(dist_loss, name='mean_kl')
        tf.summary.scalar('distribution_loss', dist_loss)
        return dist_loss

def calculate_loss(priors, posteriors, x_pred, x_q):
    with tf.variable_scope("loss"):
        loss = tf.add(recon_loss(x_q, x_pred), distribution_loss(priors, posteriors), name='loss_sum')
        tf.summary.scalar('loss', loss)
        return loss

def image_reconstruction(u):
    with tf.variable_scope("image_reconstruction"):
        x = conv_block(u, 3, (1, 1), (1, 1), name='image_reconstruction')
        stds = tf.multiply(tf.ones(x.shape), SIGMA, name='stds')
        dist = tf.distributions.Normal(loc=x, scale=stds, name='normal_distribution')
        x_pred = dist.sample(name='sampling')
        return x_pred


def generative_query_network(x, v, v_q, x_q, training):

    def body(state_g, u, r, v_q, x_q, state_i, priors, posteriors, i):

        # inference
        with tf.variable_scope("inference"):
            if (training):
                concat_i = tf.concat([x_q,
                                      v_q,
                                      r,
                                      state_g.h],
                                     3, name='concat_inference')
                h_i, state_i = inf_cell(concat_i, state_i)

                posterior_latent, posterior_params = prior_posterior(state_i.h, name='posterior')
                posteriors = posteriors.write(i, posterior_params)

        # generation
        with tf.variable_scope("generation"):
            prior_latent, prior_params = prior_posterior(state_g.h, name='prior')
            priors = priors.write(i, prior_params)

            if (training):
                prior_latent = posterior_latent

            concat_g = tf.concat([v_q,
                                  r,
                                  prior_latent],
                                 3, name='concat_generation')
            h_g, state_g = gen_cell(concat_g, state_g)
            u = tf.math.add(tf.layers.conv2d_transpose(h_g, 256, 4, 4, 'SAME', name='h_upsampling'), u, name='u_update')

        i = tf.math.add(i, 1, name='increment')

        return state_g, u, r, v_q, x_q, state_i, priors, posteriors, i

    def cond(state_g, u, r, v_q, x_q, state_i, priors, posteriors, i):
        dec = tf.less(i, L)
        return dec

    with tf.variable_scope('gqn', reuse=tf.AUTO_REUSE):
        with tf.variable_scope("init"):
            gen_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[16, 16, 519], output_channels=256,
                                                     kernel_shape=[5, 5],
                                                     name='gen_cell')  # TODO: output channels, skip connection?
            inf_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[16, 16, 552], output_channels=256,
                                                     kernel_shape=[5, 5],
                                                     name='inf_cell')  # TODO: output channels, skip connection?

            x_gt = x_q

            state_i = inf_cell.zero_state(BATCH_SIZE, dtype='float32') # c + h
            state_g = gen_cell.zero_state(BATCH_SIZE, dtype='float32') # c + h

            u = tf.zeros([x.shape[0], 64, 64, 256], name='u')
            i = 0

            x_q = tf.image.resize_images(x_q, (16, 16))
            exp_1 = tf.expand_dims(v_q, 1, name='exp_1')
            exp_2 = tf.expand_dims(exp_1, 1, name='exp_2')
            v_q = tf.broadcast_to(exp_2, (x.shape[0], 16, 16, 7), name='query_broadcast')

            priors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[x.shape[0], 16, 16, n_reg_features*2], name='priors_TA')
            posteriors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[x.shape[0], 16, 16, n_reg_features*2], name='posteriors_TA')

        r = create_representation(x, v)

        variables = (state_g, u, r, v_q, x_q, state_i, priors, posteriors, i)

        variables = tf.while_loop(cond, body, variables, parallel_iterations=1)

        state_g, u, r, v_q, x_q, state_i, priors, posteriors, i = variables

        x_pred = image_reconstruction(u)
        loss = calculate_loss(priors, posteriors, x_pred, x_gt)

        return x_pred, loss

tf.reset_default_graph()

root_path = 'data'
with tf.variable_scope('data_reader'):
    data_reader = DataReader(dataset='rooms_ring_camera', context_size=5, root=root_path)

with tf.variable_scope('adam'):
    optimizer = tf.train.AdamOptimizer()
with tf.variable_scope('data_init'):
    data = data_reader.read(batch_size=BATCH_SIZE)
    x = data.query.context.frames
    v = data.query.context.cameras
    x_q = data.target
    v_q = data.query.query_camera


train_output, train_loss = generative_query_network(x, v, v_q, x_q, training=True)
with tf.variable_scope('optimizer'):
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
        break
        total_loss = []
        j = 0
        for j in range(int((TF_RECORDS * RECORDS_IN_TF_RECORD) / BATCH_SIZE)):
            x_pred, x_gt, loss_value, summary = sess.run([train_output, x_q, train_loss, merged_summaries])
            train_writer.add_summary(summary, j)
            total_loss.append(loss_value)
            print(loss_value)
            j += 1
            print(j)
        print(j)
        print('Total epoch {0} loss: {1}'.format(i, sum(total_loss) / len(total_loss)))
