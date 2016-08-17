# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
import os
from data_helpers import DataProcessor
from text_cnn import TextCNN
from tensorflow.contrib import learn
import tensorflow as tf
from tensorflow import app
import cPickle
import ConfigParser
import shutil
import ipdb



class Net(object):
    def __init__(self):
        self.data_processor = DataProcessor()
        self.workspace = ''

    def set_config(self, config_file):
        self.config_file = config_file

    def initNet(self):
        config = ConfigParser.ConfigParser()
        config.read(self.config_file)
        maxTokenLength = int(config.get('input', 'maxTokenLength'))
        embedding_size = int(config.get('input', 'embedding_size'))
        self.data_processor = DataProcessor(embedding_size, maxTokenLength)
        #self.data_processor.channel_num = int(config.get('input', 'channel_num'))
        # create data_processor
        train_file = config.get('data','train_file')
        dev_file = config.get('data', 'dev_file')
        # set workspace for output of model and embedding
        if config.has_option('output', 'workspace'):
            self.workspace = config.get('output', 'workspace')
        else:
            self.workspace = os.path.split(train_file)[0]
        if not os.path.isdir(self.workspace):
            os.mkdir(self.workspace)
        # load data
        self.data_processor.load_train_file(train_file)
        self.data_processor.load_dev_file(dev_file)
        # add channel
        for x in config.sections():
            if x.startswith('channel'):
                static = True if config.get(x, 'static') == 'True' else False
                mode = config.get(x, 'mode') 
                embedding = config.get(x, 'embedding') 
                embedding_file = None if (embedding == 'None' or embedding == '') else embedding
                self.data_processor.add_channel(mode=mode, static=static, embedding_file=embedding_file)
        # get Net config    
        self.filter_size = [int(x) for x in config.get("net", "filter_size").split(',')]
        self.filter_num = int(config.get("net", "filter_num"))

       
    def trainNet(self, epoch_num=20, batch_size=64):
        self.data_processor.train_text2x()
        self.data_processor.dev_text2x()
        # save embedding
        self.embedding = os.path.join(self.workspace, 'embedding.bin')
        self.data_processor.dump(self.embedding)
        # init net.
        graph = tf.get_default_graph()
        with graph.as_default():
            cnn = TextCNN(
                sequence_length = self.data_processor.maxTokenLength,
                num_classes = len(self.data_processor.labels_name),
                filter_sizes = self.filter_size,
                num_filters = self.filter_num,
                ChannelMode = self.data_processor.ChannelMode,
                channel_num = self.data_processor.channel_num,
                l2_reg_lambda = 0,
            )
            # train net
            cnn.set_logdir(os.path.join(self.workspace,'model'))
            accuracy, loss = cnn.Trainer(self.data_processor, epoch_num=epoch_num, batch_size=batch_size)
            # get the model dir 
            self.model_dir = os.path.join(cnn.logdir,'checkpoints')
        return accuracy, loss 
            
    def save(self):
        # save the model and embedding to config
        config = ConfigParser.ConfigParser()
        config.read(self.config_file)
        config.set('output', 'embedding', self.embedding)
        config.set('output', 'model_dir', self.model_dir)
        with open(self.config_file, 'w') as f:
            config.write(f)

    def load(self):
        config = ConfigParser.ConfigParser()
        config.read(self.config_file)
        self.embedding = config.get('output', 'embedding')
        self.model_dir = config.get('output', 'model_dir')

    def testNet(self, text):
        self.data_processor = DataProcessor()
        self.data_processor.load(self.embedding)
        text = [unicode(x, 'utf8', errors='ignore') for x in text]
        x = self.data_processor.text2x(text)
        # init net.
        graph = tf.get_default_graph()
        with graph.as_default():
           # train net
            checkpoint_file = tf.train.latest_checkpoint(self.model_dir)
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            print 'model file {}'.format(checkpoint_file)
            print 'load session and graph'
            with sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
                
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                scores = graph.get_operation_by_name("output/scores").outputs[0]
                x = self.data_processor.text2x(text)
                class_num = scores.get_shape().as_list()[-1]
                print "class num is : {}".format(class_num)
                top_k = 5
                if(class_num < top_k):
                    top_k = class_num
                softmax = tf.nn.softmax(scores, name="softmax")
                values, indices = tf.nn.top_k(softmax, k=top_k)
                label = []
                confidence = []
                label, confidence = sess.run([indices, values],{input_x: x, dropout_keep_prob: 1.0})
        #labels_map = dict(enumerate(self.data_processor.labels_name))
        labels_map = np.array(self.data_processor.labels_name)
        label = label.tolist()
        confidence = confidence.tolist()
        labels = []
        for sample in label:
           labels.append(labels_map[sample].tolist())
        return labels, confidence
               

    def resumeTrain(self, epoch_num=20, batch_size=64):
        config = ConfigParser.ConfigParser()
        config.read(self.config_file)
        # Net config
        self.filter_size = [int(x) for x in config.get("net", "filter_size").split(',')]
        self.filter_num = int(config.get("net", "filter_num"))
        # input config
        self.embedding = config.get('output', 'embedding')
        print "Reload the embedding for resumeTrain..."
        self.data_processor = DataProcessor()
        self.data_processor.load(self.embedding)
        # load data set
        train_file = config.get('data','train_file')
        dev_file = config.get('data', 'dev_file')
        self.data_processor.load_train_file(train_file)
        self.data_processor.load_dev_file(dev_file)
        # convert text to vector
        self.data_processor.train_text2x()
        self.data_processor.dev_text2x()
        # output config
        self.model_dir = config.get('output', 'model_dir')
        self.workspace = config.get('output', 'workspace')
        graph = tf.get_default_graph()
        with graph.as_default():
            cnn = TextCNN(
                sequence_length = self.data_processor.maxTokenLength,
                num_classes = len(self.data_processor.labels_name),
                filter_sizes = self.filter_size,
                num_filters = self.filter_num,
                ChannelMode = self.data_processor.ChannelMode,
                channel_num = self.data_processor.channel_num,
                l2_reg_lambda = 0,
            )
            # train net
            cnn.set_logdir(os.path.join(self.workspace,'model'))
            # get the model dir 
            self.model_dir = os.path.join(cnn.logdir,'checkpoints')
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            print 'Resume Training from {}'.format(checkpoint)
            accuracy, loss = cnn.Trainer(self.data_processor, epoch_num=epoch_num,\
                batch_size=batch_size, checkpoint_file=checkpoint)
        return accuracy, loss 

if __name__ == '__main__':
    config_file = './static/data/config.ini'
    '''
    net = Net()
    net.set_config(config_file)
    net.initNet()
    accuracy, lost = net.trainNet(epoch_num=1)
    #accuracy, lost = net.resumeTrain(epoch_num=1)
    print 'finally accuracy: {} lost: {}'.format(accuracy, lost)
    net.save()
    del net
    '''
    net = Net()
    net.set_config(config_file)
    net.load()
    text = ['转人工']
    label, confidence =  net.testNet(text)
    print label, confidence
