# -*- coding = utf-8 -*-
# @Time : 1/21/25 16:15
# @Author : Tracy
# @File : visualize.py
# @Software : PyCharm

import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def rolling_window(data, window_size, sparse=True):
    return np.array([data[i:i+window_size] for i in range(len(data) - window_size + 1)])


def visualize_window(real_ret, fake_ret, windows, fig_path):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

    if np.ndim(axs) == 1:
        axs = axs.reshape(2, 2)

    for i in range(len(windows)):
        row = i // 2
        col = i % 2

        real_dist = rolling_window(real_ret, windows[i], sparse=not (windows[i] == 1)).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_ret.T, windows[i], sparse=not (windows[i] == 1)).sum(axis=1).ravel()

        axs[row, col].hist(real_dist, bins=50, density=True, alpha=0.5, label='Historical returns')
        axs[row, col].hist(fake_dist, bins=50, density=True, alpha=0.5, label='Synthetic returns')

        axs[row, col].set_xlim(*np.quantile(fake_dist, [0.001, 0.999]))

        axs[row, col].set_title('{} day return distribution'.format(windows[i]), size=16)
        axs[row, col].yaxis.grid(True, alpha=0.5)
        axs[row, col].set_xlabel('Cumulative log return')
        axs[row, col].set_ylabel('Frequency')

    axs[0, 0].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'window.png'))
    plt.show()

def visualize_return(y, fig_path):
    plt.plot(np.cumsum(y[0:30], axis=1).T, alpha=0.75)
    plt.savefig(os.path.join(fig_path, 'ret.png'))
    plt.show()


pass

