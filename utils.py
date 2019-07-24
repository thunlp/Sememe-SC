import sys
sys.path.append('..')
import tensorflow as tf
import pickle
import random
from numpy import linalg
import numpy as np
import math
import os
import re


def load_data_load(fname):
    with open(fname, 'rb') as fp:
        pic = fp.read()
        obj = pickle.loads(pic)
    return obj


def load_data_load_hownet_simple(filename=False):
    hownet_dict = {}
    hownet_counter = {}
    if not filename:
        # filename = 'data\HowNet_simple_new.txt'
        # filename = 'data/HowNet_simple_new_split.txt'
        print("Loading hownet_93700.txt")
        filename = 'data/hownet/hownet_93700.txt'
    with open(filename, 'r', encoding='utf-8') as hownet:
        while True:
            line = hownet.readline()
            if not line:
                break
            word = line[:-1]
            line = hownet.readline()
            sememes = line.split()
            hownet_dict[word] = sememes
            try:
                if word[-2] not in '12':
                    item = word[:-1]
                else:
                    item = word[:-2]
            except IndexError:
                item = word[:-1]
            if item not in hownet_counter:
                hownet_counter[item] = 1
            else:
                hownet_counter[item] += 1
    return hownet_dict, hownet_counter


def preprocess_build_word_sememe_cooccur(hownet_dict):
    # 根据已有的hownet的词语到义原的dict，建立义原到词语的dict
    sememe_all = list()
    for word, sememes in hownet_dict.items():
        for sememe in sememes:
            if sememe not in sememe_all:
                sememe_all.append(sememe)
    sememe_dict = {item: [] for item in sememe_all}
    for word, sememes in hownet_dict.items():
        for sememe in sememes:
            sememe_dict[sememe].append(word)
    return sememe_dict, sememe_all


class Hownet:

    def __init__(self, hownet_file=None, comp_file=None, num=None, ):
        self.num = num
        self.hownet_file = hownet_file
        self.comp_file = comp_file

        self.hownet = None
        self.predict_weights = None
        self.sememe_count = None
        self.sememe_freq = None
        self.comp = None
        self.sememe2word = None
        self.sememes = None
        self.sem_num = None
        self.word_num = None

        self.comp_train = None
        self.comp_test = None
        self.comp_dev = None

        self.word2id = None
        self.id2word = None
        self.sememe2id = None
        self.id2sememe = None
        # self.token2id()

    def split_dataset(self):
        random.seed(2018101)
        _train_id = list(random.sample(list(range(0, len(self.comp))), int(0.8 * len(self.comp))))
        dev_id = random.sample(_train_id, int(0.2 * len(self.comp)))
        test_id = list(set(range(0, len(self.comp))) - set(_train_id))
        train_id = list(set(range(0, len(self.comp))) - set(dev_id) - set(test_id))
        train_comp = []
        test_comp = []
        dev_comp = []
        for id in train_id:
            train_comp.append(self.comp[id])
        for id in test_id:
            test_comp.append(self.comp[id])
        for id in dev_id:
            dev_comp.append(self.comp[id])
        self.comp_train = train_comp
        self.comp_test = test_comp
        self.comp_dev = dev_comp

    def load_split_dataset(self, train_filename, test_filename, dev_filename):
        with open(train_filename, 'rb') as fp:
            pic = fp.read()
            self.comp_train = pickle.loads(pic)
        with open(test_filename, 'rb') as fp:
            pic = fp.read()
            self.comp_test = pickle.loads(pic)
        with open(dev_filename, 'rb') as fp:
            pic = fp.read()
            self.comp_dev = pickle.loads(pic)
        word2id = {}
        id2word = {}
        for comp_tup in self.comp_train:
            if comp_tup[0] not in word2id:
                word2id[comp_tup[0]] = len(word2id)
                id2word[comp_tup[0]] = len(id2word)
            if comp_tup[2] not in word2id:
                word2id[comp_tup[2]] = len(word2id)
                id2word[comp_tup[2]] = len(id2word)
        for comp_tup in self.comp_dev:
            if comp_tup[0] not in word2id:
                word2id[comp_tup[0]] = len(word2id)
                id2word[comp_tup[0]] = len(id2word)
            if comp_tup[2] not in word2id:
                word2id[comp_tup[2]] = len(word2id)
                id2word[comp_tup[2]] = len(id2word)
        for comp_tup in self.comp_test:
            if comp_tup[0] not in word2id:
                word2id[comp_tup[0]] = len(word2id)
                id2word[comp_tup[0]] = len(id2word)
            if comp_tup[2] not in word2id:
                word2id[comp_tup[2]] = len(word2id)
                id2word[comp_tup[2]] = len(id2word)
        self.word2id = word2id
        self.id2word = id2word

    def token2id(self):
        word2id = {}
        sememe2id = {}
        id2word = {}
        id2sememe = {}
        i = 0
        for word, sememes in self.hownet.items():
            word2id[word] = i
            id2word[i] = word
            i += 1
        i = 0
        for sememe, word in self.sememe2word.items():
            sememe2id[sememe] = i
            id2sememe[i] = sememe
            i += 1
        self.word2id = word2id
        self.id2word = id2word
        self.sememe2id = sememe2id
        self.id2sememe = id2sememe
        weights = []
        for i in range(len(self.sememes)):
            weights.append(self.sememe_freq[self.id2sememe[i]])
        self.predict_weights = weights

    def cut_hownet(self, num):
        hownetword = list(self.hownet.keys())
        hownet = {}
        for i in range(num):
            hownet[hownetword[i]] = self.hownet[hownetword[i]]
        return hownet

    def save(self, hownet_classfile):
        with open(hownet_classfile, 'wb') as fp:
            pic = pickle.dumps(self)
            fp.write(pic)

    def load(self, hownet_classfile):
        with open(hownet_classfile, 'rb') as fp:
            pic = fp.read()
            pic = pickle.loads(pic)
            return pic

    def filter_testset(self):
        train_words = set()
        # train_dict = {}
        for comptup in self.comp_train:
            train_words.add(comptup[0])
            train_words.add(comptup[2])
            # if comptup[0] in train_dict:
            #     train_dict[comptup[0]] += 1
            # else:
            #     train_dict[comptup[0]] = 1
            # if comptup[2] in train_dict:
            #     train_dict[comptup[2]] += 1
            # else:
            #     train_dict[comptup[2]] = 1
        totest_tups = []
        totrain_tups = []
        num = 0
        for comptup in self.comp_test:
            if comptup[4] in train_words:
                num += 1
                totrain_tups.append(comptup)
                change = True
                while change:
                    one_idx= random.sample(list(range(1,len(self.comp_train))), 1)
                    one_sample = self.comp_train[one_idx[0]]
                    if (one_sample[4] not in train_words) and (one_sample not in totest_tups):
                        totest_tups.append(one_sample)
                        change = False
        print('num of test-set comp_tup[4] in traindata %d'%num)
        for tup in totest_tups:
            self.comp_train.remove(tup)
            self.comp_test.append(tup)
        for tup in totrain_tups:
            self.comp_test.remove(tup)
            self.comp_train.append(tup)
        # a = set(totrain_tups).union(set(self.comp_train)-set(totest_tups))
        # self.comp_train = list(set(totrain_tups).union(set(self.comp_train)-set(totest_tups)))
        # self.comp_test = list(set(totest_tups).union(set(self.comp_test)-set(totrain_tups)))

    def filter_devset(self):
        train_words = set()
        for comptup in self.comp_train:
            train_words.add(comptup[0])
            train_words.add(comptup[2])
        todev_tups = []
        totrain_tups = []
        num = 0
        for comptup in self.comp_test:
            if comptup[4] in train_words:
                num += 1
                totrain_tups.append(comptup)
                change = True
                while change:
                    one_idx = random.sample(list(range(1, len(self.comp_train))), 1)
                    one_sample = self.comp_train[one_idx[0]]
                    if (one_sample[4] not in train_words) and (one_sample not in todev_tups):
                        todev_tups.append(one_sample)
                        change = False
        print('num of dev-set comp_tup[4] in traindata %d' % num)
        for tup in todev_tups:
            self.comp_train.remove(tup)
            self.comp_dev.append(tup)
        for tup in totrain_tups:
            self.comp_dev.remove(tup)
            self.comp_train.append(tup)

    def build_hownet(self):
        self.hownet, _ = load_data_load_hownet_simple(self.hownet_file)
        self.comp = load_data_load(self.comp_file)
        if self.num:
            self.hownet = self.cut_hownet(self.num)
        self.sememe2word, self.sememes = preprocess_build_word_sememe_cooccur(self.hownet)
        self.sem_num = len(self.sememes)
        self.word_num = len(self.hownet)

        sememe_count = {word: 0.1 for word in self.sememes}
        total_sememe_count = 0
        for word, sememes in self.hownet.items():
            for s in sememes:
                total_sememe_count += 1
                if s in sememe_count:
                    sememe_count[s] += 1
                else:
                    sememe_count[s] = 1
        sememe_freq = {}
        for s, count in sememe_count.items():
            sememe_freq[s] = float(sememe_count[s]) / total_sememe_count
        self.sememe_count = sememe_count
        self.sememe_freq = sememe_freq


def generate_one_example(hownet, comp_tup):
    word_l = comp_tup[0]
    sememes_l = comp_tup[1]
    word_r = comp_tup[2]
    sememes_r = comp_tup[3]
    word_t = comp_tup[4]

    wl = hownet.word2id[word_l]  # index for word left
    wr = hownet.word2id[word_r]  # index for word right
    sl = [hownet.sememe2id[s] for s in sememes_l]  # indexes for sememes left
    sr = [hownet.sememe2id[s] for s in sememes_r]  # indexes for sememes left
    lb = hownet.word2id[word_t]  # indexes for compound word
    pos = comp_tup[6]

    return {'wl': wl, 'wr': wr, 'sl': sl, 'sr': sr, 'lb': lb, 'pos':pos}


def generate_one_example4sememe_prediction(hownet, comp_tup):
    word_l = comp_tup[0]
    sememes_l = comp_tup[1]
    word_r = comp_tup[2]
    sememes_r = comp_tup[3]

    wl = hownet.word2id[word_l]  # index for word left
    wr = hownet.word2id[word_r]  # index for word right
    sl = [hownet.sememe2id[s] for s in sememes_l]  # indexes for sememes left
    sr = [hownet.sememe2id[s] for s in sememes_r]  # indexes for sememes left
    lb = np.zeros([1, hownet.sem_num], dtype=float)  # multi-hot label
    al = np.array([[hownet.sememe2id[s] for s in comp_tup[5]]], dtype=np.int32)
    for s in comp_tup[5]:
        lb[0][hownet.sememe2id[s]] = 1
    pos = comp_tup[6]

    return {'wl': wl, 'wr': wr, 'sl': sl, 'sr': sr, 'lb': lb, 'al': al, 'pos':pos}


def load_word_embedding(embedding_path, _hownet, scale=True):
    embed = []
    vocab = []
    with open(embedding_path, 'r', encoding='utf-8') as fembed:
        for line in fembed.readlines():
            word = line.split()[0]
            embedding = [float(item) for item in line.split()[1:]]
            embed.append(embedding)
            vocab.append(word)
    assert (len(embed) == len(vocab))
    word2id = {}
    id2word = {}
    for idx, word in enumerate(vocab):
        word2id[word] = idx
        id2word[idx] = word
    _hownet.word2id = word2id
    _hownet.id2word = id2word
    embed = np.array(embed)
    if scale:
        embed = embed / np.sqrt(np.sum(embed * embed, axis=1, keepdims=True))
    return embed, _hownet


def load_sememe_embedding(sem_embed_path, _hownet, scale=False):
    embed_dict = {}
    with open(sem_embed_path, 'r', encoding='utf-8') as fembed:
        for line in fembed.readlines():
            word = line.split()[0]
            embedding = [float(item) for item in line.split()[1:]]
            embed_dict[word] = embedding
    embed = []
    for sememe, idx in _hownet.sememe2id.items():
        embed.append(embed_dict[sememe])
    embed = np.array(embed)
    if scale:
        embed = embed / np.sqrt(np.sum(embed * embed, axis=1, keepdims=True))
    return embed


def cal_map_one(truth, prediction):
    truth_list = truth.tolist()[0]
    prediction_list = list(prediction[0])
    correct = 0
    index = 0
    point = 0
    for prediction_id in prediction_list:
        index += 1
        if prediction_id in truth_list:
            correct += 1
            point += float(correct) / index
        MAP = point / len(truth_list)
    return MAP


def predictlabel2char(id2sem, predict_dict):
    true = predict_dict['truth']
    pred = predict_dict['predict']
    char_dict = {'truth': [], 'predict': []}
    for item in true:
        char_dict['truth'].append(id2sem[item])
    for item in pred:
        char_dict['predict'].append(id2sem[item])
    return char_dict


def hamming_loss(truth, prediction, get_answer=False, predict_num=None):
    truth_list = truth.tolist()[0]
    prediction_list = list(prediction[0])
    trueset = set(truth_list)
    preset = set(prediction_list[:len(truth_list)])
    xor = len(trueset.union(preset)) - len(trueset.intersection(preset))
    return_dict = {}
    if get_answer:
        if not predict_num:
            return_dict['predict'] = prediction_list[:len(truth_list)]
        else:
            return_dict['predict'] = prediction_list[:predict_num]
        return_dict['truth'] = truth_list
    return float(xor) / len(prediction_list), return_dict


def fliter_wordsim(hownet):
    wordsim_file240 = 'wordsim-analogy/filtered_wordsim297.txt'
    wordsim_file297 = 'wordsim-analogy/filtered_wordsim240.txt'
    words = set()
    with open(wordsim_file240, 'r', encoding='utf-8') as f240:
        for line in f240:
            words.add(line.strip().split()[0])
            words.add(line.strip().split()[1])
    with open(wordsim_file297, 'r', encoding='utf-8') as f297:
        for line in f297:
            words.add(line.strip().split()[0])
            words.add(line.strip().split()[1])
    print("number of words in wordsim:{}".format(len(words)))

    new_comptrain = []
    new_compdev = []
    new_comptest = []
    for comp_tuple in hownet.comp_train:
        if comp_tuple[4] not in words:
            new_comptrain.append(comp_tuple)
        else:
            new_comptest.append(comp_tuple)
    num_wordsim_train = len(new_comptest)
    print("number of wordsim words in training set:{}".format(num_wordsim_train))
    for comp_tuple in hownet.comp_dev:
        if comp_tuple[4] not in words:
            new_compdev.append(comp_tuple)
        else:
            new_comptest.append(comp_tuple)
    num_wordsim_dev = len(new_comptest) - num_wordsim_train
    print("number of wordsim words in develop set:{}".format(num_wordsim_dev))
    new_comptrain.extend(hownet.comp_test[:num_wordsim_train])
    new_compdev.extend(hownet.comp_test[num_wordsim_train:len(new_comptest)])
    new_comptest.extend(hownet.comp_test[len(new_comptest):])
    hownet.comp_train = new_comptrain
    hownet.comp_test = new_comptest
    hownet.comp_dev = new_compdev
    return hownet, words


def fliter_wordsim_960(hownet):
    wordsim_file960 = 'wordsim-analogy/wordsim960.txt'
    # wordsim_file960 = 'wordsim-analogy/filtered_wordsim297.txt'
    words = set()
    with open(wordsim_file960, 'r', encoding='utf-8') as f960:
        for line in f960:
            words.add(line.strip().split()[0])
            words.add(line.strip().split()[1])
    print("number of words in wordsim:{}".format(len(words)))

    new_comptrain = []
    new_compdev = []
    new_comptest = []
    for comp_tuple in hownet.comp_train:
        if comp_tuple[4] not in words:
            new_comptrain.append(comp_tuple)
        else:
            new_comptest.append(comp_tuple)
    num_wordsim_train = len(new_comptest)
    print("number of wordsim words in training set:{}".format(num_wordsim_train))
    for comp_tuple in hownet.comp_dev:
        if comp_tuple[4] not in words:
            new_compdev.append(comp_tuple)
        else:
            new_comptest.append(comp_tuple)
    num_wordsim_dev = len(new_comptest) - num_wordsim_train
    print("number of wordsim words in develop set:{}".format(num_wordsim_dev))
    comp_test_rest = []
    for comp_tuple in hownet.comp_test:
        if comp_tuple[4] not in words:
            comp_test_rest.append(comp_tuple)
        else:
            new_comptest.append(comp_tuple)
    num_wordsim_test = len(new_comptest) - num_wordsim_train - num_wordsim_dev
    print("number of wordsim words in test set:{}".format(num_wordsim_test))
    new_comptrain.extend(comp_test_rest[:num_wordsim_train])
    new_compdev.extend(comp_test_rest[num_wordsim_train:num_wordsim_dev+num_wordsim_train])
    new_comptest.extend(comp_test_rest[num_wordsim_dev+num_wordsim_train:])
    hownet.comp_train = new_comptrain
    hownet.comp_test = new_comptest
    hownet.comp_dev = new_compdev
    return hownet, words


def fliter_wordsim_all(hownet):
    wordsim_file960 = 'wordsim/COS960.txt'
    wordsim_file240 = 'wordsim/filtered_wordsim297.txt'
    wordsim_file297 = 'wordsim/filtered_wordsim240.txt'
    words = set()
    with open(wordsim_file960, 'r', encoding='utf-8') as f960:
        for line in f960:
            words.add(line.strip().split()[0])
            words.add(line.strip().split()[1])
    with open(wordsim_file240, 'r', encoding='utf-8') as f960:
        for line in f960:
            words.add(line.strip().split()[0])
            words.add(line.strip().split()[1])
    with open(wordsim_file297, 'r', encoding='utf-8') as f960:
        for line in f960:
            words.add(line.strip().split()[0])
            words.add(line.strip().split()[1])
    print("number of words in wordsim:{}".format(len(words)))

    new_comptrain = []
    new_compdev = []
    new_comptest = []
    for comp_tuple in hownet.comp_train:
        if comp_tuple[4] not in words:
            new_comptrain.append(comp_tuple)
        else:
            new_comptest.append(comp_tuple)
    num_wordsim_train = len(new_comptest)
    print("number of wordsim words in training set:{}".format(num_wordsim_train))
    for comp_tuple in hownet.comp_dev:
        if comp_tuple[4] not in words:
            new_compdev.append(comp_tuple)
        else:
            new_comptest.append(comp_tuple)
    num_wordsim_dev = len(new_comptest) - num_wordsim_train
    print("number of wordsim words in develop set:{}".format(num_wordsim_dev))
    comp_test_rest = []
    for comp_tuple in hownet.comp_test:
        if comp_tuple[4] not in words:
            comp_test_rest.append(comp_tuple)
        else:
            new_comptest.append(comp_tuple)
    num_wordsim_test = len(new_comptest) - num_wordsim_train - num_wordsim_dev
    print("number of wordsim words in test set:{}".format(num_wordsim_test))
    new_comptrain.extend(comp_test_rest[:num_wordsim_train])
    new_compdev.extend(comp_test_rest[num_wordsim_train:num_wordsim_dev+num_wordsim_train])
    new_comptest.extend(comp_test_rest[num_wordsim_dev+num_wordsim_train:])
    hownet.comp_train = new_comptrain
    hownet.comp_test = new_comptest
    hownet.comp_dev = new_compdev
    return hownet, words


# load Part Of Speech tags
def load_hownet_pos():
    pos_dict = {}
    sep = [':', '}', '"']
    filename = './dataset/HowNet_original_new.txt'
    with open(filename, 'r', encoding='utf-8') as hownet:
        word = ''
        re_words = re.compile(u"[\u4e00-\u9fa5]+")
        word_remove = []
        skip = False
        while True:
            line = hownet.readline()
            if not line:
                break
            if line[:3] == 'NO.':
                word = ''
                skip = False
            if line[:3] == 'W_C':
                word = line[4:-1]
                if word in word_remove:
                    skip = True
            if line[:3] == 'G_C':
                if line.strip()[4:] and not skip:
                    pos = line.strip()[4:].split()[0]
                    if word not in pos_dict:
                        pos_dict[word] = pos
                    else:
                        if pos_dict[word] != pos:
                            pos_dict.pop(word)
                            word_remove.append(word)
    print("NUM of words have more than 1 pos:{}".format(len(word_remove)))
    print("NUM of words have only 1 pos:{}".format(len(pos_dict)))
    return pos_dict, word_remove


def divide_data_with_pos(pos_dict, hownet):
    adj_noun = []
    noun_noun = []
    verb_noun = []
    other = []
    skip = []
    comp_train_new = []
    for word_tuple in hownet.comp_train:
        if word_tuple[0] in pos_dict and word_tuple[2] in pos_dict:
            if pos_dict[word_tuple[0]] == 'adj' and pos_dict[word_tuple[2]] == 'noun':
                adj_noun.append(word_tuple)
                comp_train_new.append((word_tuple[0],word_tuple[1],word_tuple[2],word_tuple[3],
                                       word_tuple[4],word_tuple[5], 0))
            elif pos_dict[word_tuple[0]] == 'noun' and pos_dict[word_tuple[2]] == 'noun':
                noun_noun.append(word_tuple)
                comp_train_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                       word_tuple[4], word_tuple[5], 1))
            elif pos_dict[word_tuple[0]] == 'verb' and pos_dict[word_tuple[2]] == 'noun':
                verb_noun.append(word_tuple)
                comp_train_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                       word_tuple[4], word_tuple[5], 2))
            else:
                other.append(word_tuple)
                comp_train_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                       word_tuple[4], word_tuple[5], 3))
        else:
            skip.append(word_tuple)
            comp_train_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                   word_tuple[4], word_tuple[5], 3))
    print("Train set with different POS: adj-n:{}, n-n:{}, v-n:{}, other:{}, skip:{}".format(len(adj_noun), len(noun_noun), len(verb_noun), len(other), len(skip)))

    adj_noun = []
    noun_noun = []
    verb_noun = []
    other = []
    skip = []
    comp_dev_new = []
    for word_tuple in hownet.comp_dev:
        if word_tuple[0] in pos_dict and word_tuple[2] in pos_dict:
            if pos_dict[word_tuple[0]] == 'adj' and pos_dict[word_tuple[2]] == 'noun':
                adj_noun.append(word_tuple)
                comp_dev_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                       word_tuple[4], word_tuple[5], 0))
            elif pos_dict[word_tuple[0]] == 'noun' and pos_dict[word_tuple[2]] == 'noun':
                noun_noun.append(word_tuple)
                comp_dev_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                       word_tuple[4], word_tuple[5], 1))
            elif pos_dict[word_tuple[0]] == 'verb' and pos_dict[word_tuple[2]] == 'noun':
                verb_noun.append(word_tuple)
                comp_dev_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                       word_tuple[4], word_tuple[5], 2))
            else:
                other.append(word_tuple)
                comp_dev_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                       word_tuple[4], word_tuple[5], 3))
        else:
            skip.append(word_tuple)
            comp_dev_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                   word_tuple[4], word_tuple[5], 3))
    print("Dev set with different POS: adj-n:{}, n-n:{}, v-n:{}, other:{}, skip:{}".format(len(adj_noun), len(noun_noun), len(verb_noun), len(other), len(skip)))

    adj_noun = []
    noun_noun = []
    verb_noun = []
    other = []
    skip = []
    comp_test_new = []
    for word_tuple in hownet.comp_test:
        if word_tuple[0] in pos_dict and word_tuple[2] in pos_dict:
            if pos_dict[word_tuple[0]] == 'adj' and pos_dict[word_tuple[2]] == 'noun':
                adj_noun.append(word_tuple)
                comp_test_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                     word_tuple[4], word_tuple[5], 0))
            elif pos_dict[word_tuple[0]] == 'noun' and pos_dict[word_tuple[2]] == 'noun':
                noun_noun.append(word_tuple)
                comp_test_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                     word_tuple[4], word_tuple[5], 1))
            elif pos_dict[word_tuple[0]] == 'verb' and pos_dict[word_tuple[2]] == 'noun':
                verb_noun.append(word_tuple)
                comp_test_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                     word_tuple[4], word_tuple[5], 2))
            else:
                other.append(word_tuple)
                comp_test_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                     word_tuple[4], word_tuple[5], 3))
        else:
            skip.append(word_tuple)
            comp_test_new.append((word_tuple[0], word_tuple[1], word_tuple[2], word_tuple[3],
                                 word_tuple[4], word_tuple[5], 3))
    print("Test set with different POS: adj-n:{}, n-n:{}, v-n:{}, other:{}, skip:{}".format(len(adj_noun), len(noun_noun), len(verb_noun), len(other), len(skip)))
    hownet.comp_train = comp_train_new
    hownet.comp_dev = comp_dev_new
    hownet.comp_test = comp_test_new
    cls_dict = {0:'adj-n',1:'n-n',2:'v-n',3:'other'}
    return hownet, cls_dict


def norm(embed):
    tfnorm = tf.norm(embed, axis=1, keepdims=True)
    return embed / tfnorm
