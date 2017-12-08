from __future__ import division

import tensorflow as tf


class Model(object):
    def __init__(self, word_to_idx, dim_feature=[28, 2048], dim_embed=512, dim_hidden=1024, n_time_step=30,
                 batch_size=64):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of video feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM, max length of sentence.
        """

        self.word_to_idx = word_to_idx
        # self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.num_words = len(word_to_idx)
        self.dim_feature = dim_feature
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.n_time_step = n_time_step
        self.batch_size = batch_size
        self._start = word_to_idx['<start>']
        self._null = word_to_idx['<pad>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.dim_feature[0], self.dim_feature[1]])
        self.captions = tf.placeholder(tf.int32, [None, self.n_time_step + 1])

    def _initial_lstm(self, features, reuse=False):
        with tf.variable_scope('initial_lstm', reuse=reuse):
            features_mean = tf.reduce_mean(features, 1)

            h = tf.layers.dense(features_mean, self.dim_hidden, use_bias=True,
                                kernel_initializer=self.weight_initializer, bias_initializer=self.const_initializer,
                                name='h_init')
            m = tf.layers.dense(features_mean, self.dim_hidden, use_bias=True,
                                kernel_initializer=self.weight_initializer, bias_initializer=self.const_initializer,
                                name='m_init')
            return h, m

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.num_words, self.dim_embed], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, n_time_step, dim_embed ) or (N, dim_embed)
            return x

    def _temporal_attention(self, features, h, reuse=False):
        with tf.variable_scope('temporal_attention', reuse=reuse):
            features_proj = tf.layers.dense(features, self.dim_feature[1], use_bias=False,
                                            kernel_initializer=self.weight_initializer)
            e = tf.layers.dense(h, self.dim_feature[1], use_bias=True,
                                kernel_initializer=self.weight_initializer,
                                bias_initializer=self.const_initializer)
            e = tf.nn.tanh(tf.expand_dims(e, axis=1) + features_proj)
            alpha = tf.layers.dense(e, 1, activation=tf.nn.softmax, use_bias=True,
                                    kernel_initializer=self.weight_initializer,
                                    bias_initializer=self.const_initializer)

            context = tf.reduce_mean(features * alpha, axis=1)
            return context, alpha

    def _adjusted_temporal_attention(self, h, reuse=False):
        with tf.variable_scope('adjusted_temporal_attention', reuse=reuse):
            beta = tf.layers.dense(h, 1, activation=tf.nn.sigmoid, use_bias=False,
                                   kernel_initializer=self.weight_initializer)
            # _context = beta * context + (1 - beta) * _h
            # return _context, beta
            return beta

    # def _adjusted_temporal_attention(self, context, h, _h, reuse=False):
    #     with tf.variable_scope('adjusted_temporal_attention', reuse=reuse):
    #         beta = tf.layers.dense(h, 1, activation=tf.nn.sigmoid, use_bias=False,
    #                                kernel_initializer=self.weight_initializer)
    #         # _context = beta * context + (1 - beta) * _h
    #         # return _context, beta
    #         return beta

    def _mlp(self, h, _h, context, beta, reuse=False):
        with tf.variable_scope('mlp', reuse=reuse):
            x = tf.concat((h, (1 - beta) * _h, beta * context), axis=1)
            emb = tf.layers.dense(x, self.dim_embed, activation=tf.nn.sigmoid, use_bias=True,
                                  kernel_initializer=self.weight_initializer,
                                  bias_initializer=self.const_initializer)
            p = tf.layers.dense(emb, self.num_words, activation=tf.nn.softmax, use_bias=True,
                                kernel_initializer=self.weight_initializer,
                                bias_initializer=self.const_initializer)
            return p

    def build_model(self):
        features = self.features  # (batch, frames, dim_feature)
        captions = self.captions  # (batch, max_length + 1)
        batch_size = self.batch_size  # batch

        x_in = captions[:, :self.n_time_step]  # (batch, max_length)
        x_out = captions[:, 1:]  # (batch, max_length)
        mask = tf.to_float(
            tf.not_equal(x_out, self._null))  # (batch, max_length), 1 if word is not null, 0 if word is null

        h, m = self._initial_lstm(
            features=features)  # (batch, dim_hidden) (batch, dim_hidden) (batch, dim_hidden) (batch, dim_hidden)
        y = self._word_embedding(inputs=x_in)  # (batch, dim_emb)

        loss = 0.0
        alpha_list = []
        beta_list = []
        bottom_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)
        top_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)
        for t in range(self.n_time_step):  # max_length
            with tf.variable_scope('bottom_lstm', reuse=(t != 0)):
                _, (m, h) = bottom_lstm_cell(inputs=y[:, t, :], state=[m, h])  # (batch, dim_hidden) (batch, dim_hidden)

            context, alpha = self._temporal_attention(features, h,
                                                      reuse=(t != 0))  # (batch, dim_hidden) (batch, dim_hidden)
            alpha_list.append(alpha)

            with tf.variable_scope('top_lstm', reuse=(t != 0)):
                _, (_m, _h) = top_lstm_cell(inputs=h, state=(
                    tf.zeros_like(h), tf.zeros_like(h)))  # (batch, dim_hidden) (batch, dim_hidden)

            # _context, beta = self._adjusted_temporal_attention(context, h, _h,
            #                                                    reuse=(t != 0))  # (batch, dim_hidden) (batch, 1)
            beta = self._adjusted_temporal_attention(h, reuse=(t != 0))  # (batch, dim_hidden) (batch, 1)
            beta_list.append(beta)

            logits = self._mlp(h, _h, context, beta, reuse=(t != 0))  # (batch, num_words)

            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=x_out[:, t]) * mask[:, t])

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features

        h, m = self._initial_lstm(features=features)  # (dim_hidden,) (dim_hidden,)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        bottom_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)
        top_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)

        sampled_word = self._start
        for t in range(max_len):
            # x = self._word_embedding(inputs=sampled_word, reuse=(t != 0))  # (dim_emb,)
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            with tf.variable_scope('bottom_lstm', reuse=(t != 0)):
                _, (m, h) = bottom_lstm_cell(inputs=x, state=[m, h])  # (dim_hidden,)

            context, alpha = self._temporal_attention(features, h, reuse=(t != 0))  # (dim_hidden,) (dim_hidden,)
            alpha_list.append(alpha)

            with tf.variable_scope('top_lstm', reuse=(t != 0)):
                _, (_m, _h) = top_lstm_cell(inputs=h,
                                            state=(tf.zeros_like(h), tf.zeros_like(h)))  # (dim_hidden,) (dim_hidden,)

            # _context, beta = self._adjusted_temporal_attention(context, h, _h, reuse=(t != 0))  # (dim_hidden,) (1,)
            beta = self._adjusted_temporal_attention(h, reuse=(t != 0))  # (dim_hidden,) (1,)
            beta_list.append(beta)

            # logits = self._mlp(h, _context, reuse=(t != 0))  # (num_words,)
            logits = self._mlp(h, _h, context, beta, reuse=(t != 0))  # (num_words,)
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2, 3))  # (max_length, num_words)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (max_length, 1)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))  # (max_length, num_words)
        return alphas, betas, sampled_captions
