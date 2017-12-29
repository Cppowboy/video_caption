# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to result's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf


class Model(object):
    def __init__(self, word_to_idx, dim_feature=[196, 2048], dim_embed=512, dim_hidden=1024, n_time_step=16,
                 prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True, decay_c=1e-4):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of video feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.decay_c = decay_c
        self.selector = selector
        self.dropout = dropout
        self.num_words = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.n_time_step = n_time_step
        self._start = word_to_idx['<start>']
        self._null = word_to_idx['<pad>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

    def _get_initial_lstm(self, features_mean):
        with tf.variable_scope('initial_lstm'):
            # features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.dim_hidden], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.dim_hidden], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_m = tf.get_variable('w_m', [self.D, self.dim_hidden], initializer=self.weight_initializer)
            b_m = tf.get_variable('b_m', [self.dim_hidden], initializer=self.const_initializer)
            m = tf.nn.tanh(tf.matmul(features_mean, w_m) + b_m)

            _w_h = tf.get_variable('_w_h', [self.D, self.dim_hidden], initializer=self.weight_initializer)
            _b_h = tf.get_variable('_b_h', [self.dim_hidden], initializer=self.const_initializer)
            _h = tf.nn.tanh(tf.matmul(features_mean, _w_h) + _b_h)

            _w_m = tf.get_variable('_w_m', [self.D, self.dim_hidden], initializer=self.weight_initializer)
            _b_m = tf.get_variable('_b_m', [self.dim_hidden], initializer=self.const_initializer)
            _m = tf.nn.tanh(tf.matmul(features_mean, _w_m) + _b_m)

            return m, h, _m, _h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.num_words, self.dim_embed], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, n_time_step, dim_embed ) or (N, dim_embed)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.dim_hidden, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)  # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])  # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_mean(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.dim_hidden, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            sel = tf.nn.sigmoid(tf.matmul(h, w) + b, 'sel')  # (N, 1)
            context = tf.multiply(sel, context, name='selected_context')
            return context, sel

    def _mlp_layer(self, x, h, context, _h, beta, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.dim_hidden, self.dim_embed], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.dim_embed], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.dim_embed, self.num_words], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.num_words], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.prev2out:
                h_logits += x

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.dim_embed], initializer=self.weight_initializer)
                h_logits += tf.matmul(tf.multiply(beta, context), w_ctx2out)
                w__h = tf.get_variable('w__h', [self.dim_hidden, self.dim_embed], initializer=self.weight_initializer)
                h_logits += tf.matmul(tf.multiply(1 - beta, _h), w__h)
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def build_model(self, features, captions):
        batch_size = tf.shape(features)[0]
        x_in = captions[:, :self.n_time_step + 1]
        x_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(x_out, self._null))

        # batch normalize feature vectors
        # features = self._batch_norm(features, mode='train', name='conv_features')
        features_mean = tf.reduce_mean(features, 1)
        m, h, _m, _h = self._get_initial_lstm(features_mean=features_mean)
        y = self._word_embedding(inputs=x_in)
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        beta_list = []
        bottom_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)
        top_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)
        for t in range(self.n_time_step):
            with tf.variable_scope('bottom_lstm', reuse=(t != 0)):
                _, (m, h) = bottom_lstm_cell(inputs=tf.concat([_h, features_mean, y[:, t, :]], axis=1), state=[m, h])

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('top_lstm', reuse=(t != 0)):
                _, (_m, _h) = top_lstm_cell(inputs=tf.concat([context, h], axis=1), state=[_m, _h])

            # _context, beta = self._adjusted_layer(context, h, _h, reuse=(t != 0))
            logits = self._mlp_layer(y[:, t, :], h, context, _h, beta, dropout=self.dropout, reuse=(t != 0))
            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=x_out[:, t]) * mask[:, t])

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)  # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((1.0 - alphas_all) ** 2)
            loss += alpha_reg

        if self.decay_c > 0:
            weight_decay = 0
            for var in tf.trainable_variables():
                weight_decay += tf.reduce_sum(tf.square(var))
            loss += weight_decay * self.decay_c

        return loss / tf.to_float(batch_size)

    def build_sampler(self, features, max_len=20):

        # batch normalize feature vectors
        # features = self._batch_norm(features, mode='test', name='conv_features')
        features_mean = tf.reduce_mean(features, 1)
        m, h, _m, _h = self._get_initial_lstm(features_mean=features_mean)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        # sel_list = []
        beta_list = []
        bottom_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)
        top_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            with tf.variable_scope('bottom_lstm', reuse=(t != 0)):
                _, (m, h) = bottom_lstm_cell(inputs=tf.concat([_h, features_mean, x], axis=1), state=[m, h])

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                # sel_list.append(sel)
                beta_list.append(beta)
            with tf.variable_scope('top_lstm', reuse=(t != 0)):
                _, (_m, _h) = top_lstm_cell(inputs=tf.concat([context, h], axis=1), state=[_m, _h])

            # _context, beta = self._adjusted_layer(context, h, _h)
            # beta_list.append(beta)
            # logits = self._mlp_layer(x, h, _context, dropout=self.dropout, reuse=(t != 0))
            logits = self._mlp_layer(x, h, context, _h, beta, dropout=self.dropout, reuse=(t != 0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
        # sels = tf.transpose(tf.squeeze(sel_list), (1, 0))  # (N, T)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))  # (N, max_len)
        return alphas, betas, sampled_captions
