import json, pickle, re


f = open('data_path/train_data', 'r')

tag_set = set()
word2id = {}
min_count = 1


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


for line in f:
    line = line.split('\001')
    if len(line) != 3:
        continue

    product_id = line[0]
    product_name = line[1]
    prod_name_list = product_name.split()
    tags = json.loads(line[2].strip())

    if len(prod_name_list) != len(tags):
        print([prod_name_list, tags])
        continue

    for word in prod_name_list:
        word = normalize_words(word)
        if word not in word2id:
            word2id[word] = [len(word2id)+1, 1]
        else:
            word2id[word][1] += 1

    for t in tags:
        tag_set.add(str(t))

tag_dic = dict()
i = 0
for tag in tag_set:
    tag_dic[tag] = i
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


with open('data_path/word2id.pkl', 'wb') as fw:
    pickle.dump(word2id, fw)

with open('data_path/tag2label.pkl', 'wb') as fw:
    pickle.dump(tag_dic, fw)
