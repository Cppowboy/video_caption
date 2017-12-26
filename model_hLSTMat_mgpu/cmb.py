import tensorflow as tf


class CMBCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, beta=1.0):
        super(CMBCell, self).__init__()
        self._num_units = num_units
        self._beta = beta

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, self._num_units), dtype=dtype)

    def __call__(self, inputs, state):
        with tf.variable_scope('CMB'):
            value = tf.layers.dense(tf.concat([inputs, state], axis=1), units=self._num_units * 3, use_bias=True,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.constant_initializer(0.0))
            i, m, o = tf.split(value=value, num_or_size_splits=3, axis=1)
            new_state = tf.nn.tanh(m) * state + self._beta * tf.nn.sigmoid(i)
            new_h = tf.nn.sigmoid(o) * new_state
        return new_state, new_h
