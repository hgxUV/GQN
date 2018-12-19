import tensorflow as tf
import tensorflow.contrib as tfc

SIGMA = 1
L = 12
n_reg_features = 256


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
    with tf.variable_scope("create_representation"):
        frames = tf.reshape(frames, (-1, 64, 64, 3), name='frames_reshape')
        cameras = tf.reshape(cameras, (-1, 7), name='cameras_reshape')

        r = representation_pipeline_tower(frames, cameras)

        r = tf.reshape(r, (-1, 5, 16, 16, 256), name='r_rereshape')
        r = tf.math.reduce_sum(r, axis=1, name='reducing_sum_images')

    return r


def prior_posterior(h_i, name):
    gaussianParams = conv_block(h_i, 2 * n_reg_features, (5, 5), (1, 1), name='conv')
    means = gaussianParams[:, :, :, 0:n_reg_features]
    stds = gaussianParams[:, :, :, n_reg_features:]
    stds = tf.nn.softmax(stds, name='std_normalization')
    gaussianParams = tf.concat([means, stds], -1, name='mean_std_concat')
    distributions = tf.distributions.Normal(loc=means, scale=stds,
                                            name='normal_distribution')
    latent = distributions.sample()
    return latent, gaussianParams


def distribution_loss(prior, posterior):
    with tf.variable_scope("distribution_loss"):
        prior = prior.concat(name='prior_concat')
        posterior = posterior.concat(name='posterior_concat')
        distritution = lambda x: tf.distributions.Normal(loc=x[:, :, :, 0:n_reg_features],
                                                         scale=x[:, :, :, n_reg_features:],
                                                         name='normal_distribution')
        prior = distritution(prior)
        posterior = distritution(posterior)
        dist_loss = tf.distributions.kl_divergence(posterior, prior, name='kl_div')
        dist_loss = tf.reduce_mean(dist_loss, name='mean_kl')
        tf.summary.scalar('distribution_loss', dist_loss)
        return dist_loss


def image_reconstruction(u):
    with tf.variable_scope("image_reconstruction"):
        x = conv_block(u, 3, (1, 1), (1, 1), name='image_reconstruction')
        stds = tf.multiply(tf.ones(x.shape), SIGMA, name='stds')
        dist = tf.distributions.Normal(loc=x, scale=stds, name='normal_distribution')
        x_pred = dist.sample(name='sampling')
        return x_pred


def generative_query_network(data, training):
    (x, v), (x_q, v_q) = data

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
        return tf.less(i, L)

    with tf.variable_scope('gqn', reuse=tf.AUTO_REUSE,
                           regularizer=tfc.layers.l2_regularizer(1e-4)):
        with tf.variable_scope("init"):
            gen_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[16, 16, 519], output_channels=256,
                                                     kernel_shape=[5, 5],
                                                     name='gen_cell')
            inf_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[16, 16, 552], output_channels=256,
                                                     kernel_shape=[5, 5],
                                                     name='inf_cell')

            state_i = inf_cell.zero_state(x.shape[0], dtype='float32')  # c + h
            state_g = gen_cell.zero_state(x.shape[0], dtype='float32')  # c + h

            u = tf.zeros([x.shape[0], 64, 64, 256], name='u')
            i = 0

            x_q = tf.image.resize_images(x_q, (16, 16))
            exp_1 = tf.expand_dims(v_q, 1, name='exp_1')
            exp_2 = tf.expand_dims(exp_1, 1, name='exp_2')
            v_q = tf.broadcast_to(exp_2, (x.shape[0], 16, 16, 7), name='query_broadcast')

            priors = tf.TensorArray(dtype=tf.float32, size=12,
                                    element_shape=[x.shape[0], 16, 16, n_reg_features * 2])  # , name='priors_TA')
            posteriors = tf.TensorArray(dtype=tf.float32, size=12, element_shape=[x.shape[0], 16, 16,
                                                                                  n_reg_features * 2])  # , name='posteriors_TA')

        x = tf.Print(x, [x], message='before net')
        r = create_representation(x, v)
        r = tf.Print(r, [r], message='representation done')
        variables = (state_g, u, r, v_q, x_q, state_i, priors, posteriors, i)
        variables = tf.while_loop(cond, body, variables)
        state_g, u, r, v_q, x_q, state_i, priors, posteriors, i = variables
        u = tf.Print(u, [u], message='loop done')
        x_pred = image_reconstruction(u)
        x_pred = tf.Print(x_pred, [x_pred], message='image reconstruction done')

        return x_pred, priors, posteriors
