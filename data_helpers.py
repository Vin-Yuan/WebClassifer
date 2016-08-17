# -*- coding:utf-8 -*-
import numpy as np
import re
import itertools
from collections import Counter
from collections import defaultdict
import pandas as pd
import os
import sys
import cPickle
import json
import ipdb

reload(sys)
sys.setdefaultencoding('utf-8')

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec and
    filter select those w2v which word in vocab
    --------------------------------------------------
    Argument:
        fname: the pre-trained word2vec model
        vocab: the vocabulary , dict type {'word':count}
    Return:
        word_vecs: dict for {'word':vector} 
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class Token_Mode(object):
    def __init__(self, W, token_map, vocab, mode, static):
        self.W = W;
        self.token_map = token_map;
        self.vocab = vocab 
        self.mode = mode
        self.static = static

class DataProcessor(object):
    def __init__(self, embedding_size=300, maxTokenLength=50):
        self.train_text = None
        self.dev_text = None
        self.train_x = None
        self.train_y = None
        self.labels_name = None                         #
        self.dev_x = None
        self.dev_y = None
        self.dataInfoDir = ""
        self.embedding_size = embedding_size
        self.maxTokenLength = maxTokenLength
        self.channel_num = 0
        self.ChannelMode = []
    def getInfo(self):
        print 'maxTokenLength : {}'.format(self.maxTokenLength)
        print 'embedding size: {}'.format(self.embedding_size)
        print 'channel_num: {}'.format(self.channel_num)

    def dump(self, filepath):
        dump_data = [self.maxTokenLength, self.embedding_size, self.labels_name, self.channel_num]
        for x in self.ChannelMode:
            dump_data.extend([x.W, x.token_map, x.vocab, x.mode, x.static])
        with open(filepath, 'wb+') as f:
            cPickle.dump(dump_data, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            load_data = cPickle.load(f)
        self.maxTokenLength, self.embedding_size, self.labels_name, self.channel_num \
         = load_data[0], load_data[1], load_data[2], load_data[3]
        for i in range(self.channel_num):
            channel = Token_Mode(load_data[4+i*5], load_data[4+i*5+1],\
                load_data[4+i*5+2], load_data[4+i*5+3], load_data[4+i*5+4])
            self.ChannelMode.append(channel)

    def add_channel(self, mode='char', static=True, embedding_file=''):
        vocab = self.getVocabulary(mode)
        W, token_map = self.add_W(vocab=vocab, word2vecFile=embedding_file)
        channel = Token_Mode(W, token_map, vocab, mode, static)
        self.ChannelMode.append(channel)
        self.channel_num = self.channel_num + 1
        
    def set_dataInfoDir(self, dataInfoDir):
        self.dataInfoDir = dataInfoDir

    # load text and labels from train file:
    def load_train_file(self, filepath):
        self.train_text, labels = self.load_data_and_labels(filepath)
        self.train_y, self.labels_name = self.get_Y(labels)
        self.set_dataInfoDir(os.path.split(filepath)[0])

    def load_dev_file(self, filepath):
        self.dev_text, labels = self.load_data_and_labels(filepath)
        self.dev_y, _ = self.get_Y(labels)

    def load_data_and_labels(self, filepath):
        # expected csv file
        records = []
        with open(filepath, 'r') as f:
            records = f.readlines()
        labels = []
        sentences = []
        for line in records:
            piv = line.find('\t')
            label = line[0:piv]
            label = unicode(label, 'utf8', errors='ignore')
            sentence = line[piv+1:]
            sentence = unicode(sentence, 'utf8', errors='ignore')
            labels.append(label)
            sentences.append(sentence)
            # clean sentences
        return sentences, labels

    def get_Y(self, labels):
        """
            y: a list of list, which every item is a one-hot vector
        """
        if self.labels_name == None:
            class_count = Counter(labels)
            labels_name = class_count.keys()
            labels_name.sort()          # lexicographical
            class_num = len(labels_name)
            labels_map = {}
            for idx, name in enumerate(labels_name):
                temp = [0] * class_num
                temp[idx] = 1
                labels_map[name] = temp
            y = [labels_map[x] for x in labels]
            y = np.array(y, dtype=np.float32)
        else:
            labels_map = dict(enumerate(self.labels_name))
            labels_map = {v:k for k, v in labels_map.items()}
            y = np.zeros((len(labels), len(self.labels_name)), dtype=np.float32)
            for idx, label in enumerate(labels):
                y[idx][labels_map[label]] = 1
            labels_name = self.labels_name
        return y, labels_name
    
    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

    def getLookUpTable(self, word_vecs):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        embedding_size = word_vecs.values()[0].shape[-1]
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, embedding_size), dtype='float32')
        W[0] = np.zeros(embedding_size, dtype='float32')
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def add_W(self, vocab={}, word2vecFile='', name="W"):
        if word2vecFile is '' or word2vecFile is None:
            w2v_map = {}
            self.add_unknown_words(w2v_map, vocab=vocab, k=self.embedding_size)
            W, tokens_map = self.getLookUpTable(w2v_map)
        else:
            processed_w2v_file = os.path.join(self.dataInfoDir, name+'.select')
            if os.path.isfile(processed_w2v_file):
                temp = cPickle.load(open(processed_w2v_file, 'rb'))
                w2v_map = temp[0]
            else:
                w2v_map = load_bin_vec(word2vecFile, vocab)
                cPickle.dump([w2v_map], open(processed_w2v_file, 'wb'))
            # may be adjust the w2v_map to form same embedding_size
            embedding_size = w2v_map.values()[0].shape[-1]
            self.add_unknown_words(w2v_map, vocab=vocab, k=embedding_size)
            W, tokens_map = self.getLookUpTable(w2v_map) 

        return W, tokens_map

    def getVocabulary(self, token_mode, min_count=3, max_vocabulary_size=8000):
        tokens = []
        x_text = self.train_text
        maxTokenLength = 0
        if token_mode == 'char':
            for line in x_text:
                temp = list(line) 
                maxTokenLength += len(temp)
                tokens.extend(temp)
        else:
            for line in x_text:
                temp = line.split(' ')
                maxTokenLength += len(temp)
                tokens.extend(temp)
        # get average nums of sentence length
        maxTokenLength = maxTokenLength / len(x_text) + 5
        tokens_count = Counter(tokens)
        # filter tokens
        vocab = {}
        for k, v in tokens_count.iteritems():
            if(v >= min_count):
                vocab[k] = v
        return vocab

    def text2x(self, x_text):
        collect_x = None
        for TOKEN in self.ChannelMode:
            x = np.zeros((len(x_text), self.maxTokenLength))
            for i, line in enumerate(x_text):
                if TOKEN.mode == 'char':
                    tokens = list(line)
                else:
                    tokens = line.split(' ')
                end = min(self.maxTokenLength, len(tokens))
                for j in range(end):
                    if(TOKEN.token_map.has_key(tokens[j])):
                        x[i][j] = TOKEN.token_map[tokens[j]]
                    else:
                        x[i][j] = 0
            x = np.expand_dims(x, axis=-1)
            if collect_x is None:
                collect_x = x
            else:
                collect_x = np.concatenate((collect_x, x), axis=-1)
        return np.array(collect_x, dtype=np.float32)
    
    def train_text2x(self):
        x = self.text2x(self.train_text)
        self.train_x = np.array(x, dtype=np.float32)
    
    def dev_text2x(self):
        x = self.text2x(self.dev_text)
        self.dev_x = np.array(x, dtype=np.float32)
    def batch_iter(self, x, y,  batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = list(zip(x, y))
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

if __name__ == "__main__":
    data_processor = DataProcessor(embedding_size=300, maxTokenLength=50)
    data_processor.load_train_file('./data/temp/train.label')
    data_processor.load_dev_file('./data/temp/dev.label')
    data_processor.add_channel(mode="char", static=False, embedding_file='')
    data_processor.add_channel(mode="word", static=True, embedding_file='')
    data_processor.getInfo()
    data_processor.dump('./dump.bin')
    data_processor.train_text2x()
    data_processor.dev_text2x()
    del data_processor
    data_processor = DataProcessor()
    data_processor.load('./dump.bin')
    data_processor.getInfo()

