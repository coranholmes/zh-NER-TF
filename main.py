import tensorflow as tf
from model import BiLSTM_CRF
from utils import get_logger
from data import *


# set hyperparameters
# tf.app.flags.DEFINE_string("train_data", "data_path", "train data source")
# tf.app.flags.DEFINE_string("test_data", "data_path", "test data source")

# TODO change here
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
tf.app.flags.DEFINE_string("tables", "data_path/train_data,data_path/test_data2", "tables separated by comma ")
tf.app.flags.DEFINE_string('bucket', 'data_path_save',"Directory where to write event logs and checkpoint.")

FLAGS = tf.app.flags.FLAGS


# Session configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # show only warnings and errors
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# get char embeddings
tag2label, word2id = dict_build('data_path', FLAGS.min_count)
if FLAGS.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, FLAGS.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


# read corpus and get training data
train_path, test_path = FLAGS.tables.split(',')

# paths setting
paths = {}
output_path = os.path.join('.', FLAGS.bucket)
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
