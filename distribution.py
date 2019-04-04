import tensorflow as tf
import numpy as np


# Gumbel Softmax Trick
# >> implementation from Eric Jang
def sample_gumbel(shape, eps=1e-20):
    """sample from gumbel(0,1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """sample from gumbel-softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Args:
        logits: [batch, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, directly take the argmax
    Returns:
        [batch_size, n_class] sample from Gumbel-Softmax
        when hard is True, return will be one-hot
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
