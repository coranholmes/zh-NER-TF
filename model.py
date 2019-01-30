import numpy as np
import os, time, sys, pickle
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import *
from utils import get_logger


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.train_path = paths['train_path']
        self.test_path = paths['test_path']
        self.config = config

    def build_graph(self):
        self.add_placeholders()
        self.input_train_op()
        self.input_test_op()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def input_train_op(self):
        filename_queue = tf.train.string_input_producer([self.train_path], shuffle=False)
        reader = tf.TextLineReader()
        key, value = reader.read_up_to(filename_queue, num_records=self.batch_size)
        self.train_data_raw = tf.train.batch([value], batch_size=self.batch_size, capacity=64000, enqueue_many=True,
                                      allow_smaller_final_batch=True)

    def input_test_op(self):
        filename_queue = tf.train.string_input_producer([self.test_path], shuffle=False)
        reader = tf.TextLineReader()
        key, value = reader.read_up_to(filename_queue, num_records=self.batch_size)
        self.test_data_raw = tf.train.batch([value], batch_size=self.batch_size, capacity=64000, enqueue_many=True,
                                      allow_smaller_final_batch=True)


    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self):
        """

        :param train:
        :param dev:
        :return:
        """

        # self.train_size = read_meta(self.train_path)
        # self.test_size = read_meta(self.test_path)
        # print("train data: {}".format(self.train_size))

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            # sess.run(tf.local_variables_initializer())
            self.add_summary(sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, self.tag2label, epoch, saver)
            coord.request_stop()
            coord.join(threads)

    def test(self):
        # self.test_size = read_meta(self.test_path)
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            self.dev_one_epoch(sess)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, tag2label, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """

        # num_batches = self.train_size / self.batch_size if self.train_size > self.batch_size else 1
        num_batches = 78

        # step is the index of batch, seqs is the list of sentences
        try:
            for step in range(num_batches):
                train_data_raw = sess.run([self.train_data_raw])
                train_data = read_corpus(train_data_raw)
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                seqs = []
                labels = []
                for (sent_, tag_) in train_data:
                    sent_ = sentence2id(sent_, self.vocab)
                    label_ = []
                    for tag in tag_:
                        if tag2label.has_key(tag):
                            label_.append(tag2label[tag])
                        else:
                            label_.append(tag2label["O"])
                    seqs.append(sent_)
                    labels.append(label_)

                sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
                step_num = epoch * num_batches + step + 1
                feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)

                _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                             feed_dict=feed_dict)
                # if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                        '{} epoch {}, batch {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                    loss_train, step_num))

                self.file_writer.add_summary(summary, step_num)

                if step + 1 == num_batches:  # store results after last batch
                    saver.save(sess, self.model_path, global_step=step_num)
        except tf.errors.OutOfRangeError:
            pass

        self.logger.info('===========validation / test===========')
        # print("train data: {}".format(self.test_size))
        self.dev_one_epoch(sess, epoch)
        # self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, epoch=None):
        """

        :param sess:
        :param dev:
        :return:
        """
        # num_batches = self.test_size / self.batch_size if self.test_size > self.batch_size else 1
        num_batches = 2  # TODO 15

        # step is the index of batch, seqs is the list of sentences
        pred_label_list, seq_len_list, label_list, seqs_list = [], [], [], []
        try:
            for step in range(num_batches):
                test_data_raw = sess.run([self.test_data_raw])
                test_data = read_corpus(test_data_raw)

                seqs, labels = [], []
                for (sent_, tag_) in test_data:
                    sent_ = sentence2id(sent_, self.vocab)
                    seqs.append(sent_)
                    labels.append(tag_)

                predict_label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
                pred_label_list.extend(predict_label_list_)
                seq_len_list.extend(seq_len_list_)
                label_list.extend(labels)
                seqs_list.extend(seqs)
        except tf.errors.OutOfRangeError:
            pass

        self.evaluate(pred_label_list, seq_len_list, zip(seqs_list, label_list), epoch)
        # return pred_label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """

        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag

        for name in self.tag2label:
            if not name.startswith('B-'):
                continue
            tag_cnt_crt, tag_cnt_pred, tag_cnt = 0, 0, 0
            for label_, (sent, tag) in zip(label_list, data):
                tag_ = [label2tag[label__] for label__ in label_]
                tag_ = tag_[:len(sent)]
                t_ent = set(get_entity(tag, name[2:]))
                _t_ent = set(get_entity(tag_, name[2:]))
                tag_cnt += len(t_ent)
                tag_cnt_pred += len(_t_ent)
                tag_cnt_crt += len(t_ent.intersection(_t_ent))

            pre = tag_cnt_crt * 1.0 / tag_cnt_pred if tag_cnt_pred > 0 else 0
            rec = tag_cnt_crt * 1.0 / tag_cnt if tag_cnt > 0 else 0
            f1 = 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0
            if tag_cnt != 0:
                print('{}:\tprecision:{};\trecall:{};\tF1:{};\ttag_cnt:{}.'.format(name[2:], pre, rec, f1, tag_cnt))
