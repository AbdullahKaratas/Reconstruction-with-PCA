'''

    Example inspired by:
    https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca

'''

import pylab as plt
import numpy as np
import keras
from keras.datasets import mnist


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255
    x_test = x_test.reshape(10000, 784) / 255
    return x_train, x_test, y_train, y_test


def pca(x_train):
    mu = x_train.mean(axis=0)
    U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
    Zpca = np.dot(x_train - mu, V.transpose())
    return Zpca, V, mu


def reconstruction(Zpca, V, mu, x_train, rec_length):
    Rpca = np.dot(Zpca[:, :rec_length], V[:rec_length, :]) + mu
    err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
    return Rpca, err


def plot_result(x_train, Rpca):
    plt.figure(figsize=(9, 3))
    toPlot = (x_train, Rpca)
    for i in range(10):
        for j in range(2):
            ax = plt.subplot(2, 10, 10*j+i+1)
            plt.imshow(toPlot[j][i].reshape(28, 28), interpolation="nearest",
                       vmin=0, vmax=1)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Step 1, load data
    x_train, x_test, y_train, y_test = load_data()
    # Step 2, use PCA
    Zpca, V, mu = pca(x_train)
    # Step 3, reconstruction with rec_length most important elements
    rec_length = 50
    Rpca, err = reconstruction(Zpca, V, mu, x_train, rec_length)
    # Step 4, plot result
    plot_result(x_train, Rpca)
