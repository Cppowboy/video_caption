# coding: utf-8
import json
#from vocab import Vocabulary
from data.vocab import Vocabulary
import nltk
from collections import Counter
import numpy as np
import os


class MSRVTT(object):
    def __init__(self, json_path, feature_path, max_words):
        self.json_path = json_path
        self.feature_path = feature_path
        self.max_words = max_words
        self._read_json()
        self.vocab = self._prepare_vocab(self.sentences)
        self.split_dict = self._prepare_split()
        self.captions, self.lengths, self.video_ids = self._prepare_caption(self.vocab, self.split_dict, self.sentences,
                                                                            self.max_words)
        print 'msr-vtt dataset loaded'

    def _read_json(self):
        data = json.load(open(self.json_path, 'r'))
        self.info = data['info']
        self.videos = data['videos']
        self.sentences = data['sentences']

    def _prepare_vocab(self, sentences):
        '''
        根据标注的文本得到词汇表。频数低于threshold的单词将会被略去
        '''
        counter = Counter()
        ncaptions = len(sentences)
        for i, row in enumerate(sentences):
            caption = row['caption']
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

    def _prepare_split(self):
        split_dict = {}

        for i in range(0, 6000):
            split_dict['video%d' % i] = 'train'
        for i in range(6000, 7000):
            split_dict['video%d' % i] = 'val'
        for i in range(7000, 10000):
            split_dict['video%d' % i] = 'test'

        # pprint.pprint(split_dict)

        return split_dict

    def _prepare_caption(self, vocab, split_dict, anno_data, max_words):
        '''
        把caption转换成token index表示然后存到picke中
        读取存储文本标注信息的json文件，
        并且将每一条caption以及它对应的video的id保存起来，
        放回caption word_id list和video_id list
        '''
        # 初始化数据存储字典
        captions = {'train': [], 'val': [], 'test': []}
        lengths = {'train': [], 'val': [], 'test': []}
        video_ids = {'train': [], 'val': [], 'test': []}

        count = 0
        for row in anno_data:
            caption = row['caption'].lower()
            video_id = row['video_id']
            if video_id in split_dict:
                split = split_dict[video_id]
            else:
                # 如果video_id不在split_dict中
                # 那么就默认它是test
                # 这样方便我修改split来做一些过拟合训练
                split = 'test'
            words = nltk.tokenize.word_tokenize(caption)
            l = len(words) + 1  # 加上一个<end>
            lengths[split].append(l)
            if l > max_words + 1:
                # 如果caption长度超出了规定的长度，就做截断处理
                words = words[:max_words]  # 最后要留一个位置给<end>
                count += 1
            # 把caption用word id来表示
            tokens = []
            for word in words:
                tokens.append(vocab(word))
            tokens.append(vocab('<end>'))
            while l < max_words + 1:
                # 如果caption的长度少于规定的长度，就用<pad>（0）补齐
                tokens.append(vocab('<pad>'))
                l += 1
            tokens = [vocab('<start>')] + tokens
            captions[split].append(tokens)
            # video_ids[split].append(video_id)
            video_ids[split].append(int(video_id[5:]))

        # 统计一下有多少的caption长度过长
        print('There are %.3f%% too long captions' % (100 * float(count) / len(anno_data)))
        captions['train'] = np.asarray(captions['train'], dtype=np.int)
        captions['test'] = np.asarray(captions['test'], dtype=np.int)
        captions['val'] = np.asarray(captions['val'], dtype=np.int)
        lengths['train'] = np.asarray(lengths['train'], dtype=np.int)
        lengths['test'] = np.asarray(lengths['test'], dtype=np.int)
        lengths['val'] = np.asarray(lengths['val'], dtype=np.int)
        video_ids['train'] = np.asarray(video_ids['train'], dtype=np.int)
        video_ids['test'] = np.asarray(video_ids['test'], dtype=np.int)
        video_ids['val'] = np.asarray(video_ids['val'], dtype=np.int)
        return captions, lengths, video_ids

    def feature(self, vid):
        filepath = os.path.join(self.feature_path, 'video%d.npy' % (vid + 1))
        feat = np.load(open(filepath, 'r'))
        return feat


if __name__ == '__main__':
    msr_vtt_json_path = '/home/sensetime/data/msr-vtt/videodatainfo_2017.json'
    msr_vtt_feature_path = '/home/sensetime/data/msr-vtt/npy'
    max_words = 30
    msrvtt = MSRVTT(msr_vtt_json_path, msr_vtt_feature_path, max_words)
    vid = 67880
    print msrvtt.captions['train'][vid]
    print msrvtt.lengths['train'][vid]
    print msrvtt.video_ids['train'][vid]  # 1003
