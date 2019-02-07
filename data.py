import sys, pickle, os, random, io, codecs, json, re
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
    for line in lines:
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
        if word not in word2id:
            word = '<UNK>'
        else:
            word = normalize_words(word)
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
    word_path = os.path.join('.', corpus_path, 'word2id.pkl')
    tag_path = os.path.join('.', corpus_path, 'tag2label.pkl')
    if os.path.exists(word_path) and os.path.exists(tag_path):
        print("Loading vocabulary and tag dictionary.")
        with open(word_path, 'rb') as fr:
            word2id = pickle.load(fr)

        with open(tag_path, 'rb') as fr:
            tag2label = pickle.load(fr)
    else:
        print("Building vocabulary and tag dictionary.")
        f = open(os.path.join('.', corpus_path, 'train_data'), 'r')

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
