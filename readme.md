# video caption

## Introduction

This is a python project for video captioning, using hLSTMat model on the msvd or msr-vtt dataset.

## How to use the code?

### Data
1. download msvd dataset
2. download msr-vtt dataset
3. extract video feature using https://github.com/Cppowboy/video_feature_extractor

### Requirements
+ python 2.7
+ tensorflow
+ tensorboard
+ numpy
+ pandas
+ pickle

### Run
1. First, you need to change the data paths in data_engine.py to your own paths.
2. Use `python train.py` to run the train script. 
use `tensorboard --logdir your_log_dir` to visualize the train procedure and show the scores.

## Reference
1. https://github.com/zhaoluffy/hLSTMat
2. https://github.com/yunjey/show-attend-and-tell
3. Song, Jingkuan, et al. "Hierarchical LSTM with Adjusted Temporal Attention for Video Captioning." arXiv preprint arXiv:1706.01231 (2017).