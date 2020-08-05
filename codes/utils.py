from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
import os
import math
from PIL import Image


__all__ = ['task', 'set_tf_log', 'num_batch', 'batch', 'rgb2gray', 'Dummy', 'resize_image']


@contextmanager
def task(_=''):
    yield


class Dummy:
    pass


def set_tf_log(level=5):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)


def num_batch(num_data, batch_size, strict=False) -> int:
    """
    :param int num_data:
    :param int batch_size:
    :param bool strict:
    """
    if strict:
        return num_data // batch_size
    else:
        return int(math.ceil(num_data / batch_size))


def batch(data, batch_size, N=None, strict=False, shuffle=False, verbose=False):
    if N is None:
        N = len(data)

    if shuffle:
        inds = np.random.permutation(N)
    else:
        inds = np.arange(N)

    i_batch_g = range(num_batch(N, batch_size, strict=strict))

    if verbose:
        i_batch_g = tqdm(i_batch_g)

    if isinstance(data, tuple):
        for i_batch in i_batch_g:
            inds_batch = inds[i_batch * batch_size: (i_batch + 1) * batch_size]
            d_batch = tuple(v[inds_batch] for v in data)
            yield i_batch, d_batch

    else:
        for i_batch in i_batch_g:
            inds_batch = inds[i_batch * batch_size: (i_batch + 1) * batch_size]
            x_batch = data[inds_batch]
            yield i_batch, x_batch


def rgb2gray(images, keep_dims=False) -> np.ndarray:
    """

    :param np.ndarray images:
    :param bool keep_dims:
    :return:
    """
    N, H, W, C = images.shape
    assert C == 3, 'C(%s) should be 3' % C

    R, G, B = images[..., 0], images[..., 1], images[..., 2]
    result = 0.2989 * R + 0.5870 * G + 0.1140 * B

    if keep_dims:
        result = np.expand_dims(result, axis=-1)

    return result.astype(images.dtype)


def resize_image(image, shape):
    return np.array(Image.fromarray(image).resize(shape[::-1]))
