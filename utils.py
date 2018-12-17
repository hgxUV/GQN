import os

import tensorflow as tf


def save(path):
    sess = tf.get_default_session()
    if sess is None:
        raise Exception('Run this method inside default session')

    if '.ckpt' not in path:
        path = os.path.join(path, 'model.ckpt')

    if not os.path.exists(path):
        os.makedirs(path)

    saver = tf.train.Saver()
    with sess.as_default():
        saver.save(sess, path)


def restore(path):
    sess = tf.get_default_session()
    if sess is None:
        raise Exception('Run this method inside default session')

    if '.ckpt' not in path:
        path = os.path.join(path, 'model.ckpt')

        if not os.path.exists(path):
            print('No such path!')
            return

    saver = tf.train.Saver()
    try:
        saver.restore(sess, path)
        print('Model loaded from ' + path)
    except Exception:
        print("Can't load graph - probably doesn't exists")
