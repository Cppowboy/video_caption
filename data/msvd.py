# coding: utf-8
'''
准备文本相关的数据集，包括：
1. 对数据集进行划分
2. 把caption变成tokens
3. 准备ground-truth
'''
from __future__ import print_function
from __future__ import absolute_import
import nltk
import pickle
from collections import Counter
import pandas as pd
import numpy as np
import os


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, w):
        '''
        将新单词加入词汇表中
        '''
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.idx2word.append(w)
            self.nwords += 1

    def __call__(self, w):
        '''
        返回单词对应的id
        '''
        if w not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[w]

    def __len__(self):
        '''
        得到词汇表中词汇的数量
        '''
        return self.nwords


class MSVD(object):
    def __init__(self, msvd_csv_path, msvd_video_name2id_map, msvd_feature_path, max_words):
        self.msvd_csv_path = msvd_csv_path
        self.msvd_video_name2id_map = msvd_video_name2id_map
        self.msvd_feature_path = msvd_feature_path
        self.max_words = max_words

        print('# Build MSVD dataset annotations:')
        self.anno_data = self.build_msvd_annotation(self.msvd_csv_path, self.msvd_video_name2id_map)

        print('\n# Build vocabulary')
        self.vocab = self.prepare_vocab(self.anno_data)

        print('\n# Prepare dataset split')
        self.split_dict = self.prepare_split()

        print('\n# Convert each caption to token index list')
        self.captions, self.lengths, self.video_ids = self.prepare_caption(self.vocab, self.split_dict, self.anno_data,
                                                                           self.max_words)

    def build_msvd_annotation(self, msvd_csv_path, msvd_video_name2id_map):
        '''
        仿照MSR-VTT数据集的格式，为MSVD数据集生成一个包含video信息和caption标注的json文件
        之所以要和MSR-VTT的格式相似，是因为所有的数据集要共用一套prepare_captions的代码
        '''
        # 首先根据MSVD数据集官方提供的CSV文件确定每段视频的名字
        video_data = pd.read_csv(msvd_csv_path, sep=',', encoding='utf8')
        video_data = video_data[video_data['Language'] == 'English']
        # 只使用clean的描述
        # 不行，有的视频没有clean的描述
        # video_data = video_data[video_data['Source'] == 'clean']
        video_data['VideoName'] = video_data.apply(lambda row: row['VideoID'] + '_' +
                                                               str(row['Start']) + '_' +
                                                               str(row['End']), axis=1)

        video_name2id = pickle.load(open(msvd_video_name2id_map, 'r'))
        for name, vid in video_name2id.iteritems():
            video_name2id[name] = int(vid[3:]) - 1
        # 开始准备按照MSR-VTT的结构构造json文件
        sents_anno = []
        not_use_video = []
        for name, desc in zip(video_data['VideoName'], video_data['Description']):
            if name not in video_name2id:
                if name not in not_use_video:
                    print('No use: %s' % name)
                    not_use_video.append(name)
                not_use_video.append(name)
                continue
            # 有个坑，SKhmFSV-XB0这个视频里面有一个caption的内容是NaN
            if type(desc) == float:
                print('Error annotation: %s\t%s' % (name, desc))
                continue
            d = {}
            # 放大招了! 过滤掉所有非ascii字符!
            desc = desc.encode('ascii', 'ignore').decode('ascii')
            # 还有很多新的坑! 有的句子带有一大堆\n或者带有\r\n
            desc = desc.replace('\n', '')
            desc = desc.replace('\r', '')
            # 有的句子有句号结尾,有的没有,甚至有的有多句.把句号以及多于一句的内容去掉
            # MSR-VTT数据集是没有句号结尾的
            desc = desc.split('.')[0]

            d['caption'] = desc
            d['video_id'] = video_name2id[name]
            sents_anno.append(d)

        return sents_anno

    def prepare_vocab(self, sentences):
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

    def prepare_split(self):
        '''
        为数据集生成train，val，test的划分。MSVD数据集可以根据Vsubhashini的划分：
        train:1-1200, val:1201-1300, test:1301-1970
        '''
        split_dict = {}

        for i in range(1, 1201):
            split_dict[i] = 'train'
        for i in range(1201, 1301):
            split_dict[i] = 'val'
        for i in range(1301, 1971):
            split_dict[i] = 'test'

        # pprint.pprint(split_dict)

        return split_dict

    def prepare_caption(self, vocab, split_dict, anno_data, max_words):
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
            captions[split].append(tokens)
            video_ids[split].append(video_id)

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
        filepath = os.path.join(self.msvd_feature_path, 'vid%d.npy' % (vid + 1))
        feat = np.load(open(filepath, 'r'))
        return feat


if __name__ == '__main__':
    msvd_csv_path = '/home/sensetime/data/msvd/MSR_Video_Description_Corpus.csv'
    msvd_video_name2id_map = '/home/sensetime/data/msvd/youtube2text_iccv15/dict_youtube_mapping.pkl'
    msvd = MSVD(msvd_csv_path, msvd_video_name2id_map)
    captions, lengths, video_ids = msvd.captions()
