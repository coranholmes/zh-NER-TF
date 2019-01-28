import sys, pickle, os, random, io, codecs, json
import numpy as np

## tags, BIO


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with codecs.open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    # sent_, tag_ = [], []
    for line in lines:
        # if line != '\n':
        #     [char, label] = line.strip().split()
        #     sent_.append(char)
        #     tag_.append(label)
        # else:
        #     data.append((sent_, tag_))
        #     sent_, tag_ = [], []
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


# def vocab_build(vocab_path, corpus_path, min_count):
#     """
#
#     :param vocab_path:
#     :param corpus_path:
#     :param min_count:
#     :return:
#     """
#     data = read_corpus(corpus_path)
#     word2id = {}
#     for sent_, tag_ in data:
#         for word in sent_:
#             if word.isdigit():
#                 word = '<NUM>'
#             elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
#                 word = '<ENG>'
#             if word not in word2id:
#                 word2id[word] = [len(word2id)+1, 1]
#             else:
#                 word2id[word][1] += 1
#     low_freq_words = []
#     for word, [word_id, word_freq] in word2id.items():
#         if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
#             low_freq_words.append(word)
#     for word in low_freq_words:
#         del word2id[word]
#
#     new_id = 1
#     for word in word2id.keys():
#         word2id[word] = new_id
#         new_id += 1
#     word2id['<UNK>'] = new_id
#     word2id['<PAD>'] = 0
#
#     print(len(word2id))
#     with open(vocab_path, 'wb') as fw:
#         pickle.dump(word2id, fw)


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


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


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


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = []
        for tag in tag_:
            if tag2label.has_key(tag):
                label_.append(tag2label[tag])
            else:
                label_.append(tag2label["O"])

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


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