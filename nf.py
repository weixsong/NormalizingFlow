#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# define network structure


def inputs(input_dim, hidden_dim):
    x = tf.placeholder(tf.float32, [None, input_dim], 'x')
    e = tf.placeholder(tf.float32, [None, hidden_dim], 'e')
    return x, e


def encoder(x, e, input_dim, hidden_dim, z_dim, K, initializer=tf.contrib.layers.xavier_initializer):
    '''
    :param x: input
    :param e:
    :param input_dim:
    :param hidden_dim:
    :param z_dim:
    :param K: number of normalizing flow
    :param initializer:
    :return:
    '''
    with tf.variable_scope('encoder'):
        w_h = tf.get_variable('w_h', [input_dim, hidden_dim], initializer=initializer())
        b_h = tf.get_variable('b_h', [hidden_dim])
        w_mu = tf.get_variable('w_mu', [hidden_dim, z_dim], initializer=initializer())
        b_mu = tf.get_variable('b_mu', [z_dim])
        w_v = tf.get_variable('w_v', [hidden_dim, z_dim], initializer=initializer())
        b_v = tf.get_variable('b_v', [z_dim])

        # Weights for outputting normalizing flow parameters
        w_us = tf.get_variable('w_us', [hidden_dim, K*z_dim])
        b_us = tf.get_variable('b_us', [K*z_dim])
        w_ws = tf.get_variable('w_ws', [hidden_dim, K*z_dim])
        b_ws = tf.get_variable('b_ws', [K*z_dim])
        w_bs = tf.get_variable('w_bs', [hidden_dim, K])
        b_bs = tf.get_variable('b_bs', [K])

        # compute hidden state
        h = tf.nn.tanh(tf.matmul(x, w_h) + b_h)
        mu = tf.matmul(h, w_mu) + b_mu
        log_var = tf.matmul(h, w_v) + b_v
        # re-parameterization
        z = mu + tf.sqrt(tf.exp(log_var)) * e

        # Normalizing Flow parameters
        us = tf.matmul(h, w_us) + b_us
        ws = tf.matmul(h, w_ws) + b_ws
        bs = tf.matmul(h, w_bs) + b_bs

        t = (us, ws, bs)

    return mu, log_var, z, t


def norm_flow(z, lambd, K, Z):
    us, ws, bs = lambd

    log_detjs = []
    for k in range(K):
        u, w, b = us[:, k*Z:(k+1)*Z], ws[:, k*Z:(k+1)*Z], bs[:, k]
        temp = tf.expand_dims(tf.nn.tanh(tf.reduce_sum(w*z, 1) + b), 1)
        temp = tf.tile(temp, [1, u.get_shape()[1].value])
        z = z + tf.multiply(u, temp)

        # Eqn. (11) and (12)
        temp = tf.expand_dims(dtanh(tf.reduce_sum(w*z, 1) + b), 1)
        temp = tf.tile(temp, [1, w.get_shape()[1].value])
        log_detj = tf.abs(1. + tf.reduce_sum(tf.multiply(u, temp*w), 1))
        log_detjs.append(log_detj)

    if K != 0:
        log_detj = tf.reduce_sum(log_detjs)
    else:
        log_detj = 0

    return z, log_detj


def dtanh(input):
    return 1.0 - tf.square(tf.tanh(input))


def decoder(z, D, H, Z, initializer=tf.contrib.layers.xavier_initializer, out_fn=tf.sigmoid):
    with tf.variable_scope('decoder'):
        w_h = tf.get_variable('w_h', [Z, H], initializer=initializer())
        b_h = tf.get_variable('b_h', [H])
        w_mu = tf.get_variable('w_mu', [H, D], initializer=initializer())
        b_mu = tf.get_variable('b_mu', [D])
        w_v = tf.get_variable('w_v', [H, 1], initializer=initializer())
        b_v = tf.get_variable('b_v', [1])

        h = tf.nn.tanh(tf.matmul(z, w_h) + b_h)
        out_mu = tf.matmul(h, w_mu) + b_mu
        out_log_var = tf.matmul(h, w_v) + b_v
        out = out_fn(out_mu)

    return out, out_mu, out_log_var


def make_loss(pred, actual, log_var, mu, log_detj, sigma=1.0):
    # kl loss
    kl = -tf.reduce_mean(0.5*tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1))
    # re-construct loss
    # TODO: re-construct loss should be computed by negative log-likelihood of Bernoulli distribution
    # , here is only L2 loss

    rec_err = 0.5*(tf.nn.l2_loss(actual - pred)) / sigma
    loss = tf.reduce_mean(kl + rec_err - log_detj)
    # TODO: I think it is wrong here to compute the loss, wrong sign for (kl + rec_err), need verify!
    # loss = tf.reduce_mean(-kl - rec_err - log_detj)  # test this loss
    return loss


def train_step(sess, input_data, train_op, loss_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    _, l = sess.run([train_op, loss_op], feed_dict={x_op: input_data, e_op: e_})
    return l


def reconstruct(sess, batch_size, out_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    x_rec = sess.run([out_op], feed_dict={x_op: input_data, e_op: e_})
    return x_rec


def show_reconstruction(actual, recon):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(actual.reshape(28, 28), cmap='gray')
    axs[1].imshow(recon.reshape(28, 28), cmap='gray')
    axs[0].set_title('actual')
    axs[1].set_title('reconstructed')
    plt.show()


def sample_latent(sess, input_data, z_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    zs = sess.run(z_op, feed_dict={x_op: input_data, e_op: e_})
    return zs


if __name__ == '__main__':
    N = 1000
    xs = np.vstack((
        np.random.uniform(-6, -2, size=(N//3, 2)),
        np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=N//3),
        np.random.multivariate_normal([5, -5], np.eye(2) / 2, size=N//3)
    ))
    ys = np.repeat(np.arange(3), N // 3)

    idxs = np.random.choice(range(xs.shape[0]), xs.shape[0])
    xs, ys = xs[idxs], ys[idxs]

    plt.scatter(xs[:, 0], xs[:, 1], c=ys)
    plt.title('original data')
    plt.show()

    tf.reset_default_graph()
    data = xs
    data_dim = xs.shape[1]

    enc_h = 128
    enc_z = 2
    dec_h = 128
    max_iters = 10000
    batch_size = data.shape[0]
    learning_rate = 0.001
    k = 3

    x, e = inputs(data_dim, enc_z)
    mu, log_var, z0, lambd = encoder(x, e, data_dim, enc_h, enc_z, k)
    z_k, log_detj = norm_flow(z0, lambd, k, enc_z)
    out_op, out_mu, out_log_var = decoder(z_k, data_dim, dec_h, enc_z, out_fn=tf.identity)
    loss_op = make_loss(out_op, x, log_var, mu, log_detj, z0)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    idx = 0
    for i in range(max_iters):
        x_ = data[idx:idx + batch_size]
        l = train_step(sess, x_, train_op, loss_op, x, e, enc_z)
        idx += batch_size
        if idx >= x_.shape[0]:
            idx = 0
        if i % 1000 == 0:
            print('iter: %d\tloss: %.2f' % (i, l))

    zs = sample_latent(sess, xs, z_k, x, e, enc_z)
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(zs[:, 0], zs[:, 1], c=ys)
    plt.title('latent z values for each point in the dataset')
    plt.show()

    k = 500

    # Take a data point from each class, replicate it k times
    x_ = np.repeat(data[[(ys == i).argmax() for i in range(3)]], k, axis=0)
    y_ = np.repeat(np.arange(3), k)
    e_ = np.random.normal(size=(x_.shape[0], enc_z))
    zs = sess.run(z_k, feed_dict={x: x_, e: e_})

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(zs[:, 0], zs[:, 1], c=y_)
    plt.title("Posterior samples")
    plt.show()

    reconstructed = reconstruct(sess, 1000, out_op, x, e, enc_z)[0]
    plt.scatter(reconstructed[:, 0], reconstructed[:, 1])
    plt.title('reconstructed data')
    plt.show()
