# coding: utf-8
import os
import csv
from data.vocab import Vocabulary
import nltk
from collections import Counter
import numpy as np
import os


class MPIIMD(object):
    def __init__(self, **kwargs):
        self.data_dir = kwargs.pop('data_dir', '/data/mpii-md')
        self.train_anno_file = kwargs.pop('train_anno_file', 'LSMDC16_annos_training.csv')
        self.val_anno_file = kwargs.pop('val_anno_file', 'LSMDC16_annos_val.csv')
        self.test_anno_file = kwargs.pop('test_anno_file', 'LSMDC16_annos_test.csv')
        self.feature_path = kwargs.pop('feature_path', '/data/mpii-md-feature/npy')
        self.max_words = kwargs.pop('max_words', 30)
        self._load_anno()
        self.vocab = self._prepare_vocab(self.sentences['train'] + self.sentences['val'] + self.sentences['test'])
        self.captions, self.lengths = self._prepare_caption(self.vocab, self.max_words)

    def _tag2file(self, tag):
        x = tag.rfind('_')
        return tag[:x] + '/' + tag + '.avi'

    def _load_anno(self):
        self.sentences = {'train': [], 'val': [], 'test': []}
        self.video_ids = {'train': [], 'val': [], 'test': []}
        splits = ['train', 'val', 'test']
        anno_file = {'train': self.train_anno_file, 'val': self.val_anno_file, 'test': self.test_anno_file}
        for split in splits:
            reader = csv.reader(open(os.path.join(self.data_dir, anno_file[split]), 'r'), delimiter='\t')
            for row in reader:
                self.sentences[split].append(row[5])
                self.video_ids[split].append(row[0])

    def _prepare_vocab(self, sentences):
        '''
        根据标注的文本得到词汇表。频数低于threshold的单词将会被略去
        '''
        counter = Counter()
        ncaptions = len(sentences)
        for i, row in enumerate(sentences):
            caption = row
            # 直接按照空格进行单词的切分
            # tokens = caption.lower().split(' ')
            # 使用nltk来进行单词切分
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            if i % 10000 == 0:
                print('[{}/{}] tokenized the captions.'.format(i, ncaptions))
        # 略去一些低频词
        threshold = 3
        words = [w for w, c in counter.items() if c >= threshold]
        # 开始构建词典！
        vocab = Vocabulary()
        for w in words:
            vocab.add_word(w)
        return vocab

    def feature(self, tag):
        file = self._tag2file(tag)
        filepath = os.path.join(self.feature_path, file)
        return np.load(open(filepath, 'r'))

    def _prepare_caption(self, vocab, max_words):
        captions = {'train': [], 'val': [], 'test': []}
        lengths = {'train': [], 'val': [], 'test': []}

        count = 0
        splits = ['train', 'val', 'test']
        for split in splits:
            for vid, sentence in zip(self.video_ids[split], self.sentences[split]):
                caption = sentence.lower()
                video_id = vid
                words = nltk.tokenize.word_tokenize(caption)
                l = len(words) + 1  # 加上一个<end>
                lengths[split].append(l)
                if l > max_words + 1:
                    words = words[:max_words]  # 最后要留一个位置给<end>
                    count += 1
                tokens = []
                for word in words:
                    tokens.append(vocab(word))
                tokens.append(vocab('<end>'))
                while l < max_words + 1:
                    tokens.append(vocab('<pad>'))
                    l += 1
                tokens = [vocab('<start>')] + tokens
                captions[split].append(tokens)
        totallen = len(self.sentences['train']) + len(self.sentences['val']) + len(self.sentences['test'])
        print('There are %.3f%% too long captions' % (100 * float(count) / totallen))
        captions['train'] = np.asarray(captions['train'], dtype=np.int)
        captions['test'] = np.asarray(captions['test'], dtype=np.int)
        captions['val'] = np.asarray(captions['val'], dtype=np.int)
        lengths['train'] = np.asarray(lengths['train'], dtype=np.int)
        lengths['test'] = np.asarray(lengths['test'], dtype=np.int)
        lengths['val'] = np.asarray(lengths['val'], dtype=np.int)
        return captions, lengths


if __name__ == '__main__':
    mpii_md = MPIIMD()
    print len(mpii_md.sentences['train']), len(mpii_md.sentences['val']), len(mpii_md.sentences['test'])
    print len(mpii_md.video_ids['train']), len(mpii_md.video_ids['val']), len(mpii_md.video_ids['test'])
    print mpii_md.vocab
    print mpii_md.captions['train'].shape, mpii_md.captions['val'].shape, mpii_md.captions['test'].shape
    print mpii_md.lengths['train'].shape, mpii_md.lengths['val'].shape, mpii_md.lengths['test'].shape
