import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import roc_auc_score
from .utils import task, num_batch, batch


def get_encoder(input_shape=(128, 128, 1), d=8):
    model = keras.Sequential(name='encoder')
    model.add(keras.layers.Conv2D(8, 5, (2, 2), padding='same', activation=tf.nn.leaky_relu, input_shape=input_shape))

    model.add(keras.layers.Conv2D(8, 5, (2, 2), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2D(8, 3, (1, 1), padding='same', activation=tf.nn.leaky_relu))

    model.add(keras.layers.Conv2D(16, 5, (2, 2), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2D(16, 3, (1, 1), padding='same', activation=tf.nn.leaky_relu))

    model.add(keras.layers.Conv2D(16, 5, (2, 2), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2D(8, 3, (1, 1), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2D(8, 3, (1, 1), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2D(d, 8, (1, 1), padding='valid', activation=None))

    return model


def get_decoder(d=8):
    model = keras.Sequential(name='decoder')
    model.add(keras.layers.Conv2DTranspose(8, 8, (1, 1), padding='valid', activation=tf.nn.leaky_relu, input_shape=(1, 1, d)))
    model.add(keras.layers.Conv2DTranspose(8, 3, (1, 1), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2DTranspose(16, 5, (2, 2), padding='same', activation=tf.nn.leaky_relu))

    model.add(keras.layers.Conv2DTranspose(16, 3, (1, 1), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2DTranspose(16, 5, (2, 2), padding='same', activation=tf.nn.leaky_relu))

    model.add(keras.layers.Conv2DTranspose(8, 3, (1, 1), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2DTranspose(8, 5, (2, 2), padding='same', activation=tf.nn.leaky_relu))
    model.add(keras.layers.Conv2DTranspose(1, 5, (2, 2), padding='same', activation=tf.nn.sigmoid))
    return model


def batch(data, batch_size, N=None, strict=False, shuffle=False):
    if N is None:
        N = data.shape[0]

    if shuffle:
        inds = np.random.permutation(N)
    else:
        inds = np.arange(N)

    if isinstance(data, tuple):
        for i_batch in range(num_batch(N, batch_size, strict=strict)):
            inds_batch = inds[i_batch * batch_size: (i_batch + 1) * batch_size]
            d_batch = tuple(v[inds_batch] for v in data)
            yield i_batch, d_batch

    else:
        for i_batch in range(num_batch(N, batch_size, strict=strict)):
            inds_batch = inds[i_batch * batch_size: (i_batch + 1) * batch_size]
            x_batch = data[inds_batch]
            yield i_batch, x_batch


class Detector:
    def __init__(self, sess, d):
        enc = get_encoder(d=d)
        dec = get_decoder(d=d)
        ae = keras.Sequential([enc, dec], name='AE')

        X = keras.Input((128, 128, 1))
        Xh = ae(X)

        loss = tf.reduce_mean(tf.square(Xh - X), axis=[1, 2, 3])
        normal_score = -loss

        with task('Assign Fields'):
            self.enc = enc
            self.dec = dec
            self.ae = ae

            self.loss = loss
            self.normal_score = normal_score

            self.X = X
            self.Xh = Xh

            self.sess = sess

    def load(self):
        self.ae.load_weights('./ckpts/AE/AE')

    def predict(self, x):
        sess = self.sess
        scores = list()
        for i_batch, x_batch in batch(x, 512):
            score = sess.run(self.normal_score, feed_dict={self.X: x_batch})
            scores.append(score)
        return np.concatenate(scores)

    def recon(self, x):
        return self.ae.predict(x, batch_size=512)

    def evaluate(self, x, y):
        scores = self.predict(x)
        auc = roc_auc_score(y, scores)
        return auc

