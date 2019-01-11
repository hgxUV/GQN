import configargparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from GQN import generative_query_network, distribution_loss, n_reg_features
from data_reader import get_dataset
from utils import save, restore, prepare_writer

tf.set_random_seed(42)
np.random.seed(42)

RECORDS_IN_TF_RECORD = 5000
TF_RECORDS_TRAIN = 1
TF_RECORDS_VAL = 1
TRAIN_SIZE = int(TF_RECORDS_TRAIN * RECORDS_IN_TF_RECORD)
VAL_SIZE = int(TF_RECORDS_VAL * RECORDS_IN_TF_RECORD)


def loss(x, priors, posteriors, images, y):
    model_loss = tf.scalar_mul(1, tf.losses.mean_squared_error(y, x))
    dist_loss = tf.scalar_mul(1, distribution_loss(priors, posteriors))
    regularization_loss = tf.scalar_mul(1, tf.losses.get_regularization_loss())
    total_loss = model_loss + dist_loss  + regularization_loss
    return total_loss, model_loss, dist_loss, regularization_loss


def prepare_train_op(loss, init_eta, decay_step, train_beta, train_batch):
    global_step = tf.train.get_or_create_global_step()

    eta = tf.train.exponential_decay(
        init_eta,
        global_step,
        TRAIN_SIZE * decay_step / train_batch,
        train_beta)

    optimizer = tf.train.AdamOptimizer(eta)

    # todo think about gradient clipping
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, global_step)

    return train_op


def summaries(inputs, outputs, losses, training):
    pre = 'train/' if training else 'val/'
    (x, v), (x_q, v_q) = inputs
    x_pred, priors, posteriors, images = outputs
    total_loss, model_loss, dist_loss, regularization_loss = losses
    
    priors = priors.concat(name='prior_concat')
    posteriors = posteriors.concat(name='posterior_concat')
    images = images.stack(name='images_concat')
    images = tf.transpose(images, perm=[1, 0, 2, 3, 4], name='images_reshape')

    # todo: think about priors and posteriors visualization
    return tf.summary.merge([
        tf.summary.image(pre + 'x', x[0, :], 5),
        tf.summary.image(pre + 'x_query', x_q, 1),
        tf.summary.image(pre + 'x_pred', x_pred, 1),
        tf.summary.image(pre + 'LSTM', images[0, :], 12),
        tf.summary.scalar(pre + 'total_loss', total_loss),
        tf.summary.scalar(pre + 'model_loss', model_loss),
        tf.summary.scalar(pre + 'distribution_loss', dist_loss),
        tf.summary.scalar(pre + 'regularization_loss', regularization_loss),
        tf.summary.histogram(pre + 'prior_mean', priors[:, :, :, 0:n_reg_features]),
        tf.summary.histogram(pre + 'prior_sigma', priors[:, :, :, n_reg_features:]),
        tf.summary.histogram(pre + 'posterior_mean', posteriors[:, :, :, 0:n_reg_features]),
        tf.summary.histogram(pre + 'posterior_sigma', posteriors[:, :, :, n_reg_features:])
    ])


def main(args):
    t_data = get_dataset(args.dataset_path, args.dataset_name, args.context_size, args.train_batch_size, True)
    v_data = get_dataset(args.dataset_path, args.dataset_name, args.context_size, args.val_batch_size, False)

    sigma = tf.train.exponential_decay(
        args.sigma,
        tf.train.get_or_create_global_step(),
        TRAIN_SIZE * args.sigma_step / args.train_batch_size,
        args.sigma_decay
    )

    t_output = generative_query_network(t_data, sigma, True)
    v_output = generative_query_network(v_data, sigma, False)

    t_loss = loss(*t_output, t_data[1][0])
    v_loss = loss(*v_output, v_data[1][0])

    t_summary = summaries(t_data, t_output, t_loss, True)
    v_summary = summaries(v_data, v_output, v_loss, False)

    train_op = prepare_train_op(t_loss[0], args.eta, args.decay_step, args.train_beta, args.train_batch_size)

    init_local = tf.local_variables_initializer()
    init_global = tf.global_variables_initializer()

    train_writer = prepare_writer(args.logs_dir, args.out_name, 'train')
    val_writer = prepare_writer(args.logs_dir, args.out_name, 'val')

    train_iteration, val_iteration = 0, 0
        
    saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=args.save_path,
      save_secs=60,
      save_steps=None,
      saver=tf.train.Saver(),
      checkpoint_basename='model.ckpt',
      scaffold=None)
      
    saver = tf.train.Saver()

    with tf.train.MonitoredTrainingSession(hooks=[saver_hook], checkpoint_dir=args.save_path) as sess:
        sess.run(init_global)

        if args.restore_path is not None:
            ckpt = tf.train.get_checkpoint_state(args.restore_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('restoring', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

        for epoch in range(args.num_epochs):
            sess.run(init_local)
            
            with tqdm(ncols=80, total=TRAIN_SIZE,
                      bar_format='Training epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (epoch + 1)) as pbar:

                for i in range(0, TRAIN_SIZE, args.train_batch_size):
                    if train_iteration % args.train_summary_interval != 0:
                        _, tmp_loss = sess.run([train_op, t_loss[0]])
                    else:
                        _, tmp_loss, tmp_summary = sess.run([train_op, t_loss[0], t_summary])
                        train_writer.add_summary(tmp_summary, train_iteration)
                        train_writer.flush()
                    train_iteration += 1
                    pbar.update(args.train_batch_size)
                
            sess.run(init_local)
            # TODO calc some metrics? accuracy?
            if(epoch % 10 == 0):
                with tqdm(ncols=80, total=VAL_SIZE,
                          bar_format='Validation epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (epoch + 1)) as pbar:
                    for i in range(0, VAL_SIZE, args.val_batch_size):
                        if val_iteration % args.val_summary_interval != 0:
                            tmp_loss = sess.run([v_loss[0]])
                            # TODO do something
                        else:
                            tmp_loss, tmp_summary = sess.run([v_loss[0], v_summary])
                            val_writer.add_summary(tmp_summary, val_iteration)
                            val_writer.flush()
                        val_iteration += 1
                        pbar.update(args.val_batch_size)


if __name__ == '__main__':
    parser = configargparse.ArgParser(default_config_files=['*.confesl'])

    # dataset arguments
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--context-size', type=int, required=True)

    # train arguments
    parser.add_argument('--num-epochs', type=int, required=True)
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--decay-step', type=int, default=1)
    parser.add_argument('--train-beta', type=float, default=0.99)
    parser.add_argument('--train-batch-size', type=int, default=1)
    parser.add_argument('--val-batch-size', type=int, default=1)

    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--sigma-step', type=int, default=1)
    parser.add_argument('--sigma-decay', type=float, default=0.99)

    # summary arguments
    parser.add_argument('--logs-dir', type=str, default='./logs')
    parser.add_argument('--out-name', type=str, required=True)
    parser.add_argument('--train_summary_interval', type=int, default=50)
    parser.add_argument('--val_summary_interval', type=int, default=50)

    # restore arguments
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--restore-path', type=str)

    args, _ = parser.parse_known_args()
    
    assert args.save_path != args.restore_path, 'save path and restore path must be different'
    
    main(args)
