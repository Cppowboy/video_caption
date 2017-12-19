import tensorflow as tf
import numpy as np
import time
import os
from utils import decode_captions, write_bleu
from bleu import evaluate
from math import ceil


class Solver(object):
    def __init__(self, model, data, **kwargs):
        """
        Required Arguments:
            - model: caption generating model
            - data: Data object
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - model_path: String; model path for saving
        """
        # model and data
        self.model = model
        self.data = data
        # train related params
        self.n_epochs = kwargs.pop('n_epochs', 100)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.0001)
        self.print_bleu = kwargs.pop('print_bleu', True)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        # model related params
        self.max_words = kwargs.pop('max_words', 30)
        # input shape
        self.L, self.D = kwargs.pop('dim_feature', [28, 2048])
        # number of gpus
        self.num_gpus = kwargs.pop('num_gpus', 8)

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def average_loss(self, tower_loss):
        total_loss = tf.add_n(tower_loss)
        return total_loss / len(tower_loss)

    def average_gradients(self, tower_grad):
        average_grad = []
        for grad_list in zip(*tower_grad):
            expand_grad = []
            for grad in grad_list:
                expand_grad.append(tf.expand_dims(grad, 0))
            average_grad.append(tf.reduce_mean(tf.concat(expand_grad, 0), 0))
        return average_grad

    def train(self):
        # train/val dataset
        train_caps, train_lengths, train_ids = self.data.captions['train'], self.data.lengths['train'], \
                                               self.data.video_ids['train']
        n_examples = len(train_caps)
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))

        tags = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'ROUGE_L']
        # build graphs for training model and sampling captions
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                with tf.variable_scope(tf.get_variable_scope()) as vscope:
                    tower_loss = []
                    tower_grad = []
                    tower_generated_cap = []
                    # create multi gpu train_op, loss_op and generated_captions_op
                    # create placeholder
                    self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
                    self.captions = tf.placeholder(tf.int32, [None, self.max_words + 2])
                    # create train_op, loss_op and generated_captions_op
                    for i in range(self.num_gpus):
                        # on each gpu
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('tower_%d' % i) as scope:
                                # create batch input for each gpu
                                _feat_batch = self.features[
                                              self.batch_size / self.num_gpus * i:
                                              self.batch_size / self.num_gpus * (i + 1), :, :]
                                _cap_batch = self.captions[
                                             self.batch_size / self.num_gpus * i:
                                             self.batch_size / self.num_gpus * (i + 1), :]
                                # compute loss
                                one_loss = self.model.build_model(_feat_batch, _cap_batch)
                                tower_loss.append(one_loss)
                                # reuse variables
                                tf.get_variable_scope().reuse_variables()
                                alphas, betas, generated_cap = self.model.build_sampler(_feat_batch,
                                                                                        max_len=self.max_words)
                                tf.get_variable_scope().reuse_variables()
                                tower_generated_cap.append(generated_cap)
                                # compute grad
                                var_list = tf.trainable_variables()
                                grad = tf.gradients(one_loss, var_list)
                                tower_grad.append(grad)
                # multi gpu loss operation: average loss
                loss_op = self.average_loss(tower_loss)
                # caption operation
                generated_caption_op = tf.concat(tower_generated_cap, 0)
                # average grad
                average_grad = self.average_gradients(tower_grad)
                # initialize optimizer
                optimizer = self.optimizer(learning_rate=self.learning_rate, beta1=0.1, beta2=0.001)
                # train operation: apply gradients
                train_op = optimizer.apply_gradients(zip(average_grad, tf.trainable_variables()))

                # summary op
                tf.summary.scalar('batch_loss', loss_op)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)
                for grad, var in zip(average_grad, tf.trainable_variables()):
                    tf.summary.histogram(var.op.name + '/gradient', grad)

                summary_op = tf.summary.merge_all()

                # create session
                sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
                saver = tf.train.Saver(tf.global_variables())
                # initialized variables
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                for epoch in range(self.n_epochs):
                    # shuffle train data
                    rand_idxs = np.random.permutation(n_examples)
                    train_caps = train_caps[rand_idxs]
                    train_ids = train_ids[rand_idxs]
                    train_lengths = train_lengths[rand_idxs]
                    for it in range(n_iters_per_epoch):
                        captions_batch = train_caps[it * self.batch_size:(it + 1) * self.batch_size]
                        image_idxs_batch = train_ids[it * self.batch_size:(it + 1) * self.batch_size]
                        if len(captions_batch) < self.batch_size:
                            l = len(captions_batch)
                            captions_batch = np.concatenate((captions_batch, train_caps[:self.batch_size - l]), axis=0)
                            image_idxs_batch = np.concatenate((image_idxs_batch, train_ids[:self.batch_size - l]),
                                                              axis=0)
                        features_batch = [self.data.feature(vid) for vid in image_idxs_batch]
                        feed_dict = {self.features: features_batch, self.captions: captions_batch}
                        _, loss, summary_str = sess.run((train_op, loss_op, summary_op), feed_dict=feed_dict)
                        # print epoch, it, loss
                        summary_writer.add_summary(summary_str, epoch * n_iters_per_epoch + it)
                        if (it + 1) % self.print_every == 0:
                            print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (
                                epoch + 1, it + 1, loss)
                            ground_truths = train_caps[train_ids == image_idxs_batch[0]]
                            decoded = decode_captions(ground_truths[:, 1:], self.data.vocab.idx2word)
                            for j, gt in enumerate(decoded):
                                print "Ground truth %d: %s" % (j + 1, gt.encode('utf-8'))
                            gen_caps = sess.run(generated_caption_op, feed_dict)
                            decoded = decode_captions(gen_caps, self.data.vocab.idx2word)
                            print "Generated caption: %s\n" % decoded[0]
                    self.evaluate_on_split(sess=sess, generated_captions=generated_caption_op,
                                           summary_writer=summary_writer,
                                           epoch=epoch, tags=tags, split='train')
                    scores = self.evaluate_on_split(sess=sess, generated_captions=generated_caption_op,
                                                    summary_writer=summary_writer,
                                                    epoch=epoch, tags=tags, split='val')
                    write_bleu(scores=scores, path=self.model_path, epoch=epoch)
                    self.evaluate_on_split(sess=sess, generated_captions=generated_caption_op,
                                           summary_writer=summary_writer,
                                           epoch=epoch, tags=tags, split='test')
                    # save model
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=epoch + 1)
                    print "model-%s saved." % (epoch + 1)

    def evaluate_on_split(self, sess, generated_captions, summary_writer, epoch, tags, split='train'):
        caps = self.data.captions[split]
        ids = self.data.video_ids[split]
        unique_ids = list(set(ids))

        all_gen_cap = np.ndarray((len(unique_ids), self.max_words), dtype=np.int)
        for i in range(int(ceil(len(unique_ids) / float(self.batch_size)))):
            features_batch = [self.data.feature(vid) for vid in
                              unique_ids[i * self.batch_size:(i + 1) * self.batch_size]]
            if len(features_batch) < self.batch_size:
                l = len(features_batch)
                features_batch += [self.data.feature(vid) for vid in unique_ids[:self.batch_size - l]]
            features_batch = np.asarray(features_batch)
            feed_dict = {self.features: features_batch}
            gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
            all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap
        all_decoded = decode_captions(all_gen_cap, self.data.vocab.idx2word)
        # create cand dict
        cand = {}
        for vid, sentence in zip(unique_ids, all_decoded):
            cand[vid] = [sentence]
        # create ref dict
        ref = {}
        for vid in unique_ids:
            ref[vid] = decode_captions(caps[ids == vid][:, 1:], self.data.vocab.idx2word)
        # evaluate
        scores = evaluate(ref=ref, cand=cand, get_scores=True)
        for tag in tags:
            summary = tf.Summary()
            summary.value.add(tag=split + tag, simple_value=scores[tag])
            summary_writer.add_summary(summary, epoch)
        return scores

    def test(self, split='train', save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        caps = self.data.captions[split]
        ids = self.data.video_ids[split]
        unique_ids = list(set(ids))
        n_examples = len(unique_ids)
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(
            max_len=self.max_words)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        all_decoded = []
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            for i in range(n_iters_per_epoch):
                ids_batch = unique_ids[i * self.batch_size: (i + 1) * self.batch_size]
                features_batch = [self.data.feature(vid) for vid in ids_batch]
                features_batch = np.asarray(features_batch)
                feed_dict = {self.model.features: features_batch}
                alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions],
                                              feed_dict)  # (N, max_len, L), (N, max_len)
                decoded = decode_captions(sam_cap, self.data.vocab.idx2word)
                all_decoded.extend(decoded)

        # generate ref and cand
        ref = {}
        cand = {}
        for vid, dec in zip(unique_ids, all_decoded):
            gts = decode_captions(caps[ids == vid][:, 1:], self.data.vocab.idx2word)
            ref[vid] = gts
            cand[vid] = [dec]
        # print ground truths and generated sentences
        for vid in unique_ids:
            print '---' * 10
            for i, gt in enumerate(ref[vid]):
                print i + 1, ':', gt
            print 'generated :', cand[vid][0]
        scores = evaluate(ref, cand, get_scores=True)
        tags = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'ROUGE_L']
        for tag in tags:
            print tag, ':', scores[tag]
        print split, len(unique_ids), len(all_decoded)
