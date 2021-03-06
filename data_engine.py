from data.msvd import MSVD
from data.msr_vtt import MSRVTT
import cPickle as pickle
import os


class DataEngine(object):
    def __init__(self):
        pass

    def msvd(self):
        msvd_file = './msvd.pkl'
        if os.path.isfile(msvd_file):
            msvd = pickle.load(open(msvd_file, 'r'))
            return msvd
        msvd_csv_path = '/home/sensetime/data/msvd/MSR_Video_Description_Corpus.csv'
        msvd_video_name2id_map = '/home/sensetime/data/msvd/youtube2text_iccv15/dict_youtube_mapping.pkl'
        msvd_feature_path = '/home/sensetime/data/msvd/npy2'
        max_words = 30
        msvd = MSVD(msvd_csv_path, msvd_video_name2id_map, msvd_feature_path, max_words)
        # print len(msvd.captions['train']), len(msvd.captions['test']), len(msvd.captions['val'])
        pickle.dump(msvd, open(msvd_file, 'w'))
        return msvd

    def msr_vtt(self):
        msrvtt_file = './msrvtt.pkl'
        if os.path.isfile(msrvtt_file):
            msvd = pickle.load(open(msrvtt_file, 'r'))
            return msvd
        msr_vtt_json_path = '/home/sensetime/data/msr-vtt/videodatainfo_2017.json'
        msr_vtt_feature_path = '/home/sensetime/data/msr-vtt/npy'
        max_words = 30
        msrvtt = MSRVTT(msr_vtt_json_path, msr_vtt_feature_path, max_words)
        pickle.dump(msrvtt, open(msrvtt_file, 'w'))
        return msrvtt


if __name__ == '__main__':
    engine = DataEngine()
    engine.msvd()
