#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import synthetic_data


def compute_density(U_func, Z):
    neg_logp = U_func(Z)
    p = np.exp(-neg_logp)
    p /= np.sum(p)
    return p


def plot_density():
    fig, axes = plt.subplots(2, 2)
    U_list = [synthetic_data.U1, synthetic_data.U2, synthetic_data.U3, synthetic_data.U4]

    space = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(space, space)
    shape = X.shape
    X_flatten, Y_flatten = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))
    Z = np.concatenate([X_flatten, Y_flatten], 1)

    # ISSUE
    Y = -Y  # not sure why, but my plots are upside down compared to paper

    for U, ax in zip(U_list, axes.flatten()):
        density = compute_density(U, Z)
        density = np.reshape(density, shape)
        ax.pcolormesh(X, Y, density)
        ax.set_title(U.__name__)
        ax.axis('off')

    fig.tight_layout()
    plt.savefig('./data.png')


if __name__ == '__main__':
    plot_density()
