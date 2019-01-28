import tensorflow as tf
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import *
from params import *


# Session configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # show only warnings and errors
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# get char embeddings
tag2label, word2id = dict_build('data_path', args['MINI_COUNT'])
if args['PRETRAIN_EMBEDDING'] == 'random':
    embeddings = random_embedding(word2id, args['EMBEDDING_DIM'])
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


# read corpus and get training data
train_path = os.path.join('.', DATA_DIR, 'train_data')
test_path = os.path.join('.', DATA_DIR, 'test_data')
train_data = read_corpus(train_path)
test_data = read_corpus(test_path); test_size = len(test_data)

# paths setting
paths = {}
output_path = os.path.join('.', DATA_DIR+"_save")
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
get_logger(log_path).info(str(args))

# training model
if ACTION == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))  # no. of sentences
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

# testing model
elif ACTION == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)
