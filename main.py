from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from GQN import generative_query_network, distribution_loss
from data_reader import get_dataset
from utils import save, restore, prepare_writer

tf.set_random_seed(42)
np.random.seed(42)

RECORDS_IN_TF_RECORD = 5000
TF_RECORDS_TRAIN = 3
TF_RECORDS_VAL = 1
TRAIN_SIZE = int(TF_RECORDS_TRAIN * RECORDS_IN_TF_RECORD)
VAL_SIZE = int(TF_RECORDS_VAL * RECORDS_IN_TF_RECORD)


def loss(x, priors, posteriors, y):
    model_loss = tf.losses.mean_squared_error(y, x)
    dist_loss = distribution_loss(priors, posteriors)
    regularization_loss = tf.losses.get_regularization_loss()
    total_loss = model_loss + dist_loss + regularization_loss
    return total_loss, model_loss, dist_loss, regularization_loss


def prepare_train_op(loss, init_eta, decay_step, train_beta):
    global_step = tf.train.get_or_create_global_step()

    eta = tf.train.exponential_decay(
        init_eta,
        global_step,
        TRAIN_SIZE * decay_step,
        train_beta)

    optimizer = tf.train.AdamOptimizer(eta)

    # todo think about gradient clipping
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, global_step)

    return train_op


def summaries(inputs, outputs, losses, training):
    pre = 'train/' if training else 'val/'
    (x, v), (x_q, v_q) = inputs
    x_pred, priors, posteriors = outputs
    total_loss, model_loss, dist_loss, regularization_loss = losses

    # todo: think about priors and posteriors visualization
    return tf.summary.merge([
        tf.summary.image(pre + 'x', x[0, :], 5),
        tf.summary.image(pre + 'x_query', x_q, 1),
        tf.summary.image(pre + 'x_pred', x_pred, 1),
        tf.summary.scalar(pre + 'total_loss', total_loss),
        tf.summary.scalar(pre + 'model_loss', model_loss),
        tf.summary.scalar(pre + 'distribution_loss', dist_loss),
        tf.summary.scalar(pre + 'regularization_loss', regularization_loss)
    ])


def main(args):
    t_data = get_dataset(args.dataset_path, args.dataset_name, args.context_size, args.train_batch_size, True)
    v_data = get_dataset(args.dataset_path, args.dataset_name, args.context_size, args.val_batch_size, False)

    t_output = generative_query_network(t_data, True)
    v_output = generative_query_network(v_data, False)

    t_loss = loss(*t_output, t_data[1][0])
    v_loss = loss(*v_output, v_data[1][0])

    t_summary = summaries(t_data, t_output, t_loss, True)
    v_summary = summaries(v_data, v_output, v_loss, False)

    train_op = prepare_train_op(t_loss[0], args.eta, args.decay_step, args.train_beta)

    init_local = tf.local_variables_initializer()
    init_global = tf.global_variables_initializer()

    train_writer = prepare_writer(args.logs_dir, args.out_name, 'train')
    val_writer = prepare_writer(args.logs_dir, args.out_name, 'val')

    train_iteration, val_iteration = 0, 0

    with tf.Session() as sess:
        sess.run(init_global)

        if args.restore_path is not None:
            restore(args.restore_path)

        for epoch in range(args.num_epochs):
            sess.run(init_local)
            with tqdm(ncols=80, total=TRAIN_SIZE,
                      bar_format='Validation epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (epoch + 1)) as pbar:
                for i in range(TRAIN_SIZE):
                    if i % args.train_summary_interval == 0:
                        _, tmp_loss = sess.run([train_op, t_loss[0]])
                    else:
                        _, tmp_loss, tmp_summary = sess.run([train_op, t_loss[0], t_summary])
                        train_writer.add_summary(tmp_summary, train_iteration)
                        train_writer.flush()
                    train_iteration += 1
                    pbar.update(1)

            sess.run(init_local)
            # TODO calc some metrics? accuracy?
            with tqdm(ncols=80, total=VAL_SIZE,
                      bar_format='Validation epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (epoch + 1)) as pbar:
                for i in range(VAL_SIZE):
                    if i % args.val_summary_interval == 0:
                        tmp_loss = sess.run([v_loss[0]])
                    else:
                        tmp_loss, tmp_summary = sess.run([v_loss[0], v_summary])
                        val_writer.add_summary(tmp_summary, val_iteration)
                        val_writer.flush()
                    val_iteration += 1
                    pbar.update(1)

            save(args.save_path)


if __name__ == '__main__':
    parser = ArgumentParser()

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

    # summary arguments
    parser.add_argument('--logs-dir', type=str, default='./logs')
    parser.add_argument('--out-name', type=str, required=True)
    parser.add_argument('--train_summary_interval', type=int, default=50)
    parser.add_argument('--val_summary_interval', type=int, default=50)

    # restore arguments
    parser.add_argument('--restore-path', type=str)

    args, _ = parser.parse_known_args()
    main(args)
