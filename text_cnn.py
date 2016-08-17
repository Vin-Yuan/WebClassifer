import tensorflow as tf
import numpy as np
from data_helpers import DataProcessor
import time
import datetime
import cPickle
import os
import shutil
import ipdb


#tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
#tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
#tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
#FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      filter_sizes, num_filters, ChannelMode, channel_num, l2_reg_lambda=0.0):
        """
            sequence_length: the length of every sentence
            ChannelMode: a list contains the different embeddings for multi channel
                in which element is a dict {name:'w2v', embedding:w2v_matrix, static:True}
                and the num equals to channel_num
            the input_x's channel dimension is corronspandence to each embedding in VocabEmbeddings
        """
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        Embeddings = [] 
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            for idx, channel in enumerate(ChannelMode):
                if channel.static:
                    print 'static , embedding_size: {}'.format(channel.W.shape[-1])
                    W = tf.constant(channel.W, name='static')
                else:
                    print 'non-static , embedding_size: {}'.format(channel.W.shape[-1])
                    W = tf.Variable(channel.W, name='non_static')
                embedded_token = tf.nn.embedding_lookup(W, self.input_x[:,:,idx])
                Embeddings.append(tf.expand_dims(embedded_token, -1))
        # token may be: word or char
        # Create a convolution + maxpool layer for each filter size
        # below take the all the W as same size, so just concat them
        embedding = tf.concat(concat_dim=3, values=Embeddings, name="concat_embeddings")
        # embedding is [None, height, width, channel_num]
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                embedding_size = embedding.get_shape().as_list()[2]     # width
                filter_shape = [filter_size, embedding_size, channel_num, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedding,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # add all channle's maxpooling to [none,1,1,128]
                # which like [None, a+b+c, 128]
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # add l2_norm based on original paper
            W = tf.nn.l2_normalize(W,dim=0) * 3.0
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def Trainer(self, data_processor, epoch_num=20, batch_size=64, checkpoint_file = None):
    # resume_train is checkpoint file path
        graph = tf.get_default_graph()
        #with tf.Graph().as_default():
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=True,
              log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Define Training procedure
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(loss=self.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
                
                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.merge_summary(grad_summaries)
                
                # Output directory for models and summaries
                if not hasattr(self, 'logdir'):
                    timestamp = str(int(time.time()))
                    #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                    out_dir = os.path.join("runs", timestamp)
                    self.logdir = 'runs'
                else:
                    out_dir = self.logdir
                if os.path.isdir(out_dir) and checkpoint_file is None:
                    shutil.rmtree(out_dir)
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.scalar_summary("loss", self.loss)
                acc_summary = tf.scalar_summary("accuracy", self.accuracy)

                # Train Summaries
                self.train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                self.train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

                # Dev summaries
                self.dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                self.dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.join(out_dir, "checkpoints")
                self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                self.saver = tf.train.Saver(tf.all_variables(), max_to_keep = 0)
                if checkpoint_file is None:
                    # Initialize all variables
                    sess.run(tf.initialize_all_variables())
                    dev_accur, dev_loss = \
                        self.batch_train(sess, data_processor, batch_size=batch_size, num_epochs=epoch_num)
                else:
                    print("load checkpoint ......")
                    self.saver.restore(sess, checkpoint_file)
                    print("Model restored.")
                    dev_accur, dev_loss = \
                        self.batch_train(sess, data_processor, batch_size=batch_size, num_epochs=epoch_num)
        return dev_accur, dev_loss

    def set_logdir(self, log_dir):
        self.logdir = log_dir

    def batch_iter(self, x, y,  batch_size, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = list(zip(x, y))
        data = np.array(data)
        data_size = len(data)
        print 'data size : {}'.format(data_size)
        num_batches_per_epoch = int(data_size/batch_size) + 1
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

    def train_step(self, x_batch, y_batch, sess):
        """
        A single training step
        """
        feed_dict = {
          self.input_x: x_batch,
          self.input_y: y_batch,
          self.dropout_keep_prob: 0.5 #FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)
        return loss, accuracy

    def dev_step(self, x_batch, y_batch, sess):
        feed_dict = {
          self.input_x: x_batch,
          self.input_y: y_batch,
          self.dropout_keep_prob: 1.0  
        }
        step, summaries, loss, accuracy = sess.run(
            [self.global_step, self.dev_summary_op, self.loss, self.accuracy],
            feed_dict)
        self.dev_summary_writer.add_summary(summaries, step)
        return loss, accuracy

    def train_epoch(self, sess, train_x, train_y, batch_size):
        # Generate batches
        batches = self.batch_iter(train_x, train_y, batch_size)
        # Training loop. For each batch...
        train_loss = []
        train_accuracy = []
        for batch in batches:
            x, y = zip(*batch)
            loss, accuracy = self.train_step(x, y, sess)
            train_loss.append(loss), train_accuracy.append(accuracy)
        train_loss = sum(train_loss) / len(train_loss)
        train_accuracy = sum(train_accuracy) / len(train_accuracy)
        return train_loss, train_accuracy

    def dev_epoch(self, sess, dev_x, dev_y, batch_size):
        batches = self.batch_iter(dev_x, dev_y, batch_size)
        # dev loop. For each batch...
        dev_loss = []
        dev_accuracy = []
        for batch in batches:
            x, y = zip(*batch)
            loss, accuracy = self.dev_step(x, y, sess)
            dev_loss.append(loss), dev_accuracy.append(accuracy)
        dev_loss = sum(dev_loss) / len(dev_loss)
        dev_accuracy = sum(dev_accuracy) / len(dev_accuracy)
        return dev_loss, dev_accuracy


    def batch_train(self, sess, data_processor, batch_size=64, num_epochs=100):
        # Training loop. For each batch...
        for epoch in range(num_epochs):
            outputStr = []
            outputStr.append("Epoch------{}--------".format(epoch))
            epoch_start = datetime.datetime.now()
            # train epcoh iter 
            train_loss, train_accur = self.train_epoch(sess, data_processor.train_x, data_processor.train_y,batch_size=64)
            time_str = datetime.datetime.now().isoformat()
            current_step = tf.train.global_step(sess, self.global_step)
            outputStr.append("Train:....data size:{}".format(len(data_processor.train_x)))
            outputStr.append("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, train_loss, train_accur))
            # dev epoch iter 
            dev_loss, dev_accur = self.dev_epoch(sess, data_processor.dev_x, data_processor.dev_y,batch_size=64)
            time_str = datetime.datetime.now().isoformat()
            current_step = tf.train.global_step(sess, self.global_step)
            outputStr.append("Evaluation:....data size:{}".format(len(data_processor.dev_x)))
            outputStr.append("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, dev_loss, dev_accur))
            # save model
            path = self.saver.save(sess, self.checkpoint_prefix, global_step=current_step)
            outputStr.append("Saved model checkpoint to {}".format(path))
            epoch_end = datetime.datetime.now()
            delta = (epoch_end - epoch_start).seconds
            if(delta > 60):
                outputStr.append("epoch spend time {} minutes".format(delta / 60.0))
            else:
                outputStr.append("epoch spend time {} seconds".format(delta))
            # write log information
            for x in outputStr:
                print x
            outputStr = [x+'\n' for x in outputStr]
            with open(os.path.join(self.logdir, 'log.txt'),'a+') as f:
                f.writelines(outputStr)
        return dev_accur, dev_loss
        

