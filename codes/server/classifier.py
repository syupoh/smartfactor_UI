import tensorflow as tf
import numpy as np
from .garam import SVM_boost_predict_test


class BinaryModel:
    def __init__(self, x):
        self.name = 'noname'
        self.x = x
        h = self.x
        with tf.variable_scope(self.name):
            with tf.variable_scope('layer1'):
                h = tf.contrib.layers.conv2d(h, 32, (7, 7))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.conv2d(h, 32, (7, 7))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer2'):
                h = tf.contrib.layers.conv2d(h, 32, (7, 7))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.conv2d(h, 32, (7, 7))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer3'):
                h = tf.contrib.layers.conv2d(h, 64, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.conv2d(h, 64, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer4'):
                h = tf.contrib.layers.conv2d(h, 128, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.conv2d(h, 128, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer5'):
                h = tf.contrib.layers.flatten(h)
                h = tf.contrib.layers.fully_connected(h, 128)

            with tf.variable_scope('layer6'):
                logit = tf.contrib.layers.fully_connected(h, 2, activation_fn=None)
                probs = tf.nn.softmax(logit)

                self.probs = probs
                self.logit = logit


class Classifier:
    def __init__(self):
        self.X = tf.placeholder(tf.uint8, [None, 128, 128, 3])
        X = tf.cast(self.X, tf.float32)
        X -= [150.62635218, 148.43409781, 144.14871928]
        X /= [11.60014713, 10.87613746, 11.82365726]

        self.model = BinaryModel(X)
        self.probs = self.model.probs

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='noname')
        saver = tf.train.Saver(var_list=vs)
        saver.restore(self.sess, './ckpts/green_model.ckpt')

    def infer_probs(self, images):
        probs = self.sess.run(self.model.probs, feed_dict={self.X: images})
        return probs[:, 1]

    def infer_garam(self, image):
        result = SVM_boost_predict_test(image)
        return result

    def infer(self, images):
        probs = self.infer_probs(images)
        # garam = [self.infer_garam(image) for image in images]
        garam = np.zeros_like(probs)
        return probs, garam

    def classify(self, images):
        probs, garam = self.infer(images)
        # result = (probs >= self.thres).astype(np.int32)
        return probs
