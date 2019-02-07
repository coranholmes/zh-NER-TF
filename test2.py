import json
from data import *

lines = open('data_path/test_data')
tag_set = set()
word2id = {}

for line in lines:
    train_data = read_corpus([[line]])
    if train_data == []:
        continue
    prod_name_list = train_data[0][0]
    tags = train_data[0][1]
    if len(prod_name_list) != len(tags):
        continue

    for word in prod_name_list:
        word = normalize_words(word)
        if word not in word2id:
            word2id[word] = [len(word2id) + 1, 1]
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
    if word_freq < 1:
        low_freq_words.append(word)
for word in low_freq_words:
    del word2id[word]

new_id = 1
for word in word2id.keys():
    word2id[word] = new_id
    new_id += 1
word2id['<UNK>'] = new_id
word2id['<PAD>'] = 0
print(tag2label)
print(word2id)