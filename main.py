import tensorflow as tf
import numpy as np
import sys, pickle, os, random, io, codecs, json, re, logging
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

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
        reader = tf.TableRecordReader()
        key, value = reader.read_up_to(filename_queue, num_records=self.batch_size)
        self.train_data_raw = tf.train.batch([value], batch_size=self.batch_size, capacity=64000, enqueue_many=True,
                                      allow_smaller_final_batch=True)

    def input_test_op(self):
        filename_queue = tf.train.string_input_producer([self.test_path], shuffle=False)
        reader = tf.TableRecordReader()
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

        self.train_size = read_meta(self.train_path)
        self.test_size = read_meta(self.test_path)
        print('train_data_size:', self.train_size)
        print('test_data_size:', self.test_size)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            sess.run(tf.local_variables_initializer())
            self.add_summary(sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, self.tag2label, epoch, saver)
            coord.request_stop()
            coord.join(threads)

    def test(self):
        self.test_size = read_meta(self.test_path)
        print('test_data_size:', self.test_size)

        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            sess.run(self.init_op)
            sess.run(tf.local_variables_initializer())  # todo: what the fuck???
            saver.restore(sess, self.model_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            self.dev_one_epoch(sess)
            coord.request_stop()
            coord.join(threads)

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

        num_batches = self.train_size / self.batch_size if self.train_size > self.batch_size else 1
        print('num_batches:', num_batches)
        # num_batches = 78

        # step is the index of batch, seqs is the list of sentences

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
        num_batches = self.test_size / self.batch_size if self.test_size > self.batch_size else 1
        # num_batches = 2  # TODO 15

        # step is the index of batch, seqs is the list of sentences
        pred_label_list, seq_len_list, label_list, seqs_list = [], [], [], []

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


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def read_meta(table_name):
    """read the size information of the table, only one record is read out

        """
    filename_queue = tf.train.string_input_producer([table_name], num_epochs=1)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        data_size = 0
        try:
            while not coord.should_stop():
                values = sess.run([value])
                data_size += len(values)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)
        return data_size

def read_corpus(lines):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    for line in lines[0]:
        arr = line.split('\001')
        if len(arr) != 3:
            continue
        sent_ = arr[1].split()
        tag_ = json.loads(arr[2])
        tag_ = [str(i) for i in tag_]
        if len(sent_) > 100:
            continue  # delete those too long sentences
        if len(sent_) == len(tag_):
            data.append((sent_, tag_))
    return data


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    for a set of sentences, padding it to the max length of the sentences
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def get_entity(tags, name):
    entities = []
    start = 0
    for i, t in enumerate(tags):
        if t == 'B-'+name or t == 'O':
            if i > start and tags[start] == 'B-'+name:
                entities.append((start, i))  # the entity is from start to i-1
            start = i
    if i > start and tags[start] == 'B-'+name:
        entities.append((start, i+1))
    return entities


def normalize_words(word):
    numal = re.compile(r'^[0-9]+[^0-9]+$')
    alnum = re.compile(r'^[^0-9]+[0-9]+$')
    num = re.compile(r'.?[0-9].?')
    if word.isdigit():
        return "<NUM>"
    elif numal.search(word):
        return "<NUMAL>"
    elif alnum.search(word):
        return "<ALNUM>"
    elif num.search(word):
        return "<MIXED>"
    return word


def dict_build(corpus_path, min_count):

    # check whether the dictionaries are already built
    word_path = os.path.join(corpus_path, 'word2id.pkl')
    tag_path = os.path.join(corpus_path, 'tag2label.pkl')
    print(word_path)
    print(tag_path)
    if os.path.exists(word_path) and os.path.exists(tag_path):
        print("Loading vocabulary and tag dictionary.")
        with open(word_path, 'rb') as fr:
            word2id = pickle.load(fr)

        with open(tag_path, 'rb') as fr:
            tag2label = pickle.load(fr)
    else:
        print("Building vocabulary and tag dictionary.")
        f = open(os.path.join(corpus_path, 'train_data'), 'r')  # TODO CHANGE

        tag_set = set()
        word2id = {}

        for line in f:
            line = line.split('\001')
            if len(line) != 3:
                continue

            product_name = line[1]
            prod_name_list = product_name.split()
            tags = json.loads(line[2].strip())

            if len(prod_name_list) != len(tags):
                # print([prod_name_list, tags])
                continue

            for word in prod_name_list:
                word = normalize_words(word)
                if word not in word2id:
                    word2id[word] = [len(word2id)+1, 1]
                else:
                    word2id[word][1] += 1

            for t in tags:
                tag_set.add(str(t))

        tag2label = dict()
        i = 0
        for tag in tag_set:
            tag2label[tag] = i
            i += 1

        low_freq_words = []
        for word, [word_id, word_freq] in word2id.items():
            if word_freq < min_count and word != '<NUM>' and word != '<NUM_CHAR>' and word != '<CHAR_NUM>':
                low_freq_words.append(word)
        for word in low_freq_words:
            del word2id[word]

        new_id = 1
        for word in word2id.keys():
            word2id[word] = new_id
            new_id += 1
        word2id['<UNK>'] = new_id
        word2id['<PAD>'] = 0

        with open(word_path, 'wb') as fw:
            pickle.dump(word2id, fw)

        with open(tag_path, 'wb') as fw:
            pickle.dump(tag2label, fw)
    print('tag_types:', len(tag2label))
    print('vocab_size:', len(word2id))
    return tag2label, word2id


# set hyperparameters
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_integer("epoch", 2, "#epoch of training")
tf.app.flags.DEFINE_integer("hidden_dim", 300, "#dim of hidden state")
tf.app.flags.DEFINE_string("optimizer", "Adam", "Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD")
tf.app.flags.DEFINE_boolean("CRF", True, "use CRF at the top layer. if False, use Softmax")
tf.app.flags.DEFINE_float("lr", 0.001, "learning rate")
tf.app.flags.DEFINE_float("clip", 5.0, "gradient clipping")
tf.app.flags.DEFINE_float("dropout", 0.5, "dropout keep_prob")
tf.app.flags.DEFINE_boolean("update_embedding", True, "update embedding during training")
tf.app.flags.DEFINE_string("pretrain_embedding", "random", "use pretrained char embedding or init it randomly")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "random init char embedding_dim")
tf.app.flags.DEFINE_boolean("shuffle", True, "shuffle training data before each epoch")
tf.app.flags.DEFINE_integer("min_count", 2, "min_count for vocabulary building")
tf.app.flags.DEFINE_string("action", "test", "train/test")
tf.app.flags.DEFINE_string("tables", "", "tables separated by comma ")
tf.app.flags.DEFINE_string('checkpointDir', '', "Directory where to write event logs and checkpoint.")

FLAGS = tf.app.flags.FLAGS


# Session configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # show only warnings and errors
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# get char embeddings
tag2label, word2id = dict_build(FLAGS.checkpointDir, FLAGS.min_count)
if FLAGS.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, FLAGS.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


# read corpus and get training data
train_path, test_path = FLAGS.tables.split(',')

# paths setting
paths = {}
output_path = FLAGS.checkpointDir
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
paths['train_path'] = train_path
paths['test_path'] = test_path
# get_logger(log_path).info(str(FLAGS))

# training model
if FLAGS.action == 'train':
    model = BiLSTM_CRF(FLAGS, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## train model on the whole training data
    # print("train data: {}".format(train_size))  # no. of sentences
    model.train()  # use test_data as the dev_data to see overfitting phenomena

# testing model
elif FLAGS.action == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(FLAGS, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    # print("test data: {}".format(test_size))
    model.test()
