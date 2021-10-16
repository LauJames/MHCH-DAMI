#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle as pkl
import jieba
import jieba.posseg as pseg
import logging
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
prodir = '..'
sys.path.insert(0, prodir)

from vocab import Vocab
from utility import get_now_time
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from snownlp import SnowNLP
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

pos2id = {
    'a': 1, 'ad': 2, 'ag': 3, 'an': 4, 'b': 5,
    'c': 6, 'd': 7, 'df': 8, 'dg': 9,
    'e': 10, 'f': 11, 'g': 12, 'h': 13,
    'i': 14, 'j': 15, 'k': 16, 'l': 17,
    'm': 18, 'mg': 19, 'mq': 20,
    'n': 21, 'ng': 22, 'nr': 23, 'nrfg': 24, 'nrt': 25,
    'ns': 26, 'nt': 27, 'nz': 28,
    'o': 29, 'p': 30, 'q': 31,
    'r': 32, 'rg': 33, 'rr': 34, 'rz': 35,
    's': 36, 't': 37, 'tg': 38,
    'u': 39, 'ud': 40, 'ug': 41, 'uj': 42, 'ul': 43, 'uv': 44, 'uz': 45,
    'v': 46, 'vd': 47, 'vg': 48, 'vi': 49, 'vn': 50, 'vq': 51,
    'x': 52, 'y': 53, 'z': 54, 'zg': 55, 'pad': 0
}


class DataPrepare(object):
    def __init__(self, mode='train', data_name='normal', log_path=None, use_pre_train=True, embed_size=200,
                 use_senti=True):
        """
        Constant variable declaration and configuration.
        """
        self.use_senti = use_senti
        self.pos_dim = 52

        if data_name == 'clothing':
            dataset_folder_name = '/data' + '/clothing'
            self.raw_dialogue_path = curdir + dataset_folder_name + '/cloth_annotated_3500.shuf.json'
            self.train_raw_path = curdir + dataset_folder_name + '/hmt_cloth_train.json'
            self.val_raw_path = curdir + dataset_folder_name + '/hmt_cloth_eval.json'
            self.test_raw_path = curdir + dataset_folder_name + '/hmt_cloth_test.json'

        elif data_name == 'makeup':
            dataset_folder_name = '/data' + '/makeup'
            self.raw_dialogue_path = curdir + dataset_folder_name + '/makeup_annotated_4000.shuf.json'
            self.train_raw_path = curdir + dataset_folder_name + '/hmt_makeup_train.json'
            self.val_raw_path = curdir + dataset_folder_name + '/hmt_makeup_eval.json'
            self.test_raw_path = curdir + dataset_folder_name + '/hmt_makeup_test.json'

        else:
            raise ValueError("Please confirm the correct data mode you entered.")

        self.vocab_save_path = curdir + dataset_folder_name + '/vocab.pkl'
        self.train_path = curdir + dataset_folder_name + '/train.pkl'
        self.val_path = curdir + dataset_folder_name + '/eval.pkl'
        self.test_path = curdir + dataset_folder_name + '/test.pkl'
        self.predict_path = curdir + dataset_folder_name + '/predict.pkl'

        self.use_pre_train = use_pre_train
        self.pre_train_embeddings_path = curdir + '/data/w2v/cbow.word2vec.200d'
        self.embed_size = embed_size

        self.dialogues_list = []
        self.role_list = []
        self.deal_list = []
        self.contents_list = []
        self.pos_list = []
        self.dialogues_ids_list = []
        self.dialogues_len_list = []
        self.dialogues_sent_len_list = []
        self.label_list = []
        self.session_id_list = []
        self.senti_list = []
        self.tf_list = []

        self.mode = mode
        self._load_raw_data(mode)

    def _load_raw_data(self, mode):
        """
        Check data, create the directories, prepare for the vocabulary and embeddings.
        """
        if mode == 'vocab':
            data_path = self.raw_dialogue_path
        elif mode == 'train':
            data_path = self.train_raw_path
        elif mode == 'eval':
            data_path = self.val_raw_path
        elif mode == 'test':
            data_path = self.test_raw_path
        else:
            raise ValueError("{} mode not exists, please check it.".format(mode))

        if not os.path.exists(data_path):
            now_time = get_now_time()
            raise ValueError("{}: File {} is not exist.".format(now_time, data_path))

        with open(data_path, 'r', encoding='utf-8', errors='ignore', newline='\n') as fin:
            print("Open {} successfully.".format(data_path))

            error_num, count = 0, 0
            while True:
                try:
                    line = fin.readline()
                    count += 1
                except IOError as e:
                    print(e)
                    error_num += 1
                    continue
                if not line:
                    print("Load data successfully!")
                    break

                try:
                    json_obj = json.loads(line)
                    tmp_label_list = []
                    self.dialogues_list.append(json_obj["session"])
                    self.session_id_list.append(json_obj["sessionID"])

                except ValueError as e:
                    error_num += 1
                    print("error line of json format: {}".format(line))

    def _load_vocab(self, vocab_path):
        """
        If we already have preprocessed vocabulary object, load it. Or gen a vocab object using gen_vocab()
        :param vocab_path:
        :return:
        """
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as fin:
                vocab = pkl.load(fin)
                return vocab
        else:
            return self.gen_vocab()

    def word_iter(self):
        """
        Iterates over all the words in dialogue content.
        :return: a generator
        """

        if self.dialogues_list is not None:
            for dialogue in self.dialogues_list:
                for one_turn in dialogue:
                    for token, postag in pseg.cut(one_turn["content"]):
                        yield token, postag
        else:
            raise ValueError("Get a empty dialogues_list.")

    def convert2ids(self):
        """
        Convert the question and passage in the original dataset to ids.
        :return: None
        """
        vocab = self._load_vocab(self.vocab_save_path)
        if self.dialogues_list is not None:
            for dialogue in self.dialogues_list:
                dialogue_ids = []
                dialogue_tfs = []
                role_ids = []
                pos_list = []
                tmp_sent_len_list = []
                label_list = []
                senti_scores_list = []
                for one_turn in dialogue:
                    tmp_ids = vocab.convert2ids(jieba.cut(one_turn["content"]))
                    tmp_tfs = vocab.convert2tfs(jieba.cut(one_turn["content"]))
                    # pos
                    one_pos_list = []
                    for _, pos_flag in pseg.cut(one_turn["content"]):
                        if vocab.pos2id.__contains__(pos_flag):
                            pos_id = vocab.pos2id[pos_flag]
                        else:
                            print("unknow pos tag: {}".format(pos_flag))
                            pos_id = 0
                        one_pos_list.append(pos_id)
                    pos_list.append(one_pos_list)
                    tmp_role = 0 if one_turn["role"] == "c2b" else 1

                    dialogue_ids.append(tmp_ids)
                    dialogue_tfs.append(tmp_tfs)
                    role_ids.append(tmp_role)
                    tmp_sent_len_list.append(len(tmp_ids))
                    if self.use_senti:
                        if len(one_turn["content"]) == 0:
                            tmp_senti = 0.5
                        else:
                            tmp_ss = SnowNLP(one_turn["content"])
                            tmp_senti = tmp_ss.sentiments
                        senti_scores_list.append(tmp_senti)
                    label_list.append(int(one_turn["label"]))
                self.dialogues_sent_len_list.append(tmp_sent_len_list)
                self.dialogues_len_list.append(len(tmp_sent_len_list))
                self.dialogues_ids_list.append(
                    pad_sequences(dialogue_ids, maxlen=50, padding='post', truncating='post'))
                self.tf_list.append(pad_sequences(dialogue_tfs, maxlen=50, padding='post', truncating='post'))
                self.pos_list.append(pad_sequences(pos_list, maxlen=50, padding='post', truncating='post', value=0))
                self.role_list.append(role_ids)
                self.senti_list.append(senti_scores_list)
                self.label_list.append(label_list)

        # scale
        ss_senti = StandardScaler()
        self.senti_list = pad_sequences(self.senti_list, maxlen=30, padding='post', truncating='post', value=0.5)
        self.senti_list = ss_senti.fit_transform(self.senti_list)
        ss_tf = MinMaxScaler()
        self.tf_list = pad_sequences(self.tf_list, maxlen=30, padding='post', truncating='post', dtype='float32')
        self.tf_list_reshape = np.reshape(self.tf_list, (-1, 50))
        self.tf_list_reshape = ss_tf.fit_transform(self.tf_list_reshape)
        self.tf_list = np.reshape(self.tf_list_reshape, (-1, 30, 50))
        print("Transform all data {} to id successfully!".format(len(self.dialogues_len_list)))

    def gen_vocab(self, min_cnt=2):
        """
        Utilizing the corpus to gen vocabulary and save to pickle.
        :return: None
        """
        vocab = Vocab(lower=True)
        for word, postag in self.word_iter():
            vocab.add(word)
            vocab.add_pos2id(postag)

        unfiltered_vocab_size = vocab.size()
        vocab.filter_tokens_by_cnt(min_cnt=min_cnt)
        filtered_num = unfiltered_vocab_size - vocab.size()
        print('After filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.size()))

        print('Assigning embedding ...')
        if self.use_pre_train:
            print('Pre trained')
            vocab.load_pretrained_embeddings(self.pre_train_embeddings_path)
        else:
            print('Random')
            vocab.randomly_init_embeddings(self.embed_size)

        print('Saving vocab ...')
        print('vocab size is: {}'.format(vocab.size()))
        print('pos to id dict size is {}'.format(len(vocab.pos2id)))

        with open(self.vocab_save_path, 'wb') as fout:
            pkl.dump(vocab, fout)
        print('Done with vocab!')
        return vocab

    def desensitization(self):
        with open(self.vocab_save_path, 'rb') as fin:
            vocab = pkl.load(fin)
        vocab.desensitization()
        with open(self.vocab_save_path, 'wb') as fout:
            pkl.dump(vocab, fout)
        print('Done with desensitization!')

    def save_data(self, mode='train'):
        """
        Save the transformed data to pickle.
        :param mode: str  train/val/test
        :return: None
        """
        if mode == 'train':
            load_path = self.train_path
        elif mode == 'eval':
            load_path = self.val_path
        elif mode == 'test':
            load_path = self.test_path
        else:
            raise ValueError("{} mode not exists, please check it.".format(mode))
        self.convert2ids()

        with open(load_path, 'wb') as fout:
            pkl.dump(self.dialogues_ids_list, fout)
            pkl.dump(self.role_list, fout)

            # add term freq, pos tag
            pkl.dump(self.tf_list, fout)
            pkl.dump(self.pos_list, fout)

            pkl.dump(self.senti_list, fout)
            pkl.dump(self.dialogues_sent_len_list, fout)
            pkl.dump(self.dialogues_len_list, fout)
            pkl.dump(self.label_list, fout)

        print("Save variable into {}".format(load_path))

    def load_pkl_data(self, mode='train'):
        if mode == 'train':
            load_path = self.train_path
        elif mode == 'eval':
            load_path = self.val_path
        elif mode == 'test':
            load_path = self.test_path
        else:
            raise ValueError("{} mode not exists, please check it.".format(mode))

        if not os.path.exists(load_path):
            raise ValueError("{} not exists, please generate it firstly.".format(load_path))
        else:
            with open(load_path, 'rb') as fin:
                # X
                self.dialogues_ids_list = pkl.load(fin)
                self.role_list = pkl.load(fin)

                self.tf_list = pkl.load(fin)
                self.pos_list = pkl.load(fin)
                # use sentiment
                self.senti_list = pkl.load(fin)
                self.dialogues_sent_len_list = pkl.load(fin)
                self.dialogues_len_list = pkl.load(fin)
                self.label_list = pkl.load(fin)
            print("Load variable from {} successfully!".format(load_path))

    def load_config(self, config_path):
        with open(config_path, 'r') as fp:
            return json.load(fp)

    def data_generator_sup(self, data_name='makeup', mode='test', batch_size=32, shuffle=True, nb_classes=2):
        print('Using data_generator_sup')
        self.load_pkl_data(mode=mode)
        x1 = self.dialogues_ids_list
        x2 = self.role_list
        x3 = self.senti_list
        label_list = self.label_list
        sent_len = self.dialogues_sent_len_list
        dia_len = self.dialogues_len_list

        print("Total {} dialogues in {} mode".format(len(self.dialogues_ids_list), mode))

        for i in tqdm(range(0, len(label_list), batch_size), desc="Processing:"):
            batch_x1 = pad_sequences(x1[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            # role
            batch_x2 = pad_sequences(x2[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x3 = pad_sequences(x3[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_sent_len = pad_sequences(sent_len[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                           dtype='int32')
            batch_dia_len = dia_len[i: i + batch_size]
            # [B, D_len, nb_classes]
            labels_padded = pad_sequences(label_list[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                          dtype='int32', value=0)
            batch_labels = to_categorical(labels_padded, nb_classes, dtype='int32')

            yield batch_x1, batch_x2, batch_x3, batch_labels, batch_sent_len, batch_dia_len

    def data_generator_crf(self, data_name='makeup', mode='test', batch_size=32, shuffle=True, nb_classes=2):
        print('Using data_generator_crf')
        self.load_pkl_data(mode=mode)
        x1 = self.dialogues_ids_list
        x2 = self.role_list
        x3 = self.senti_list
        label_list = self.label_list
        sent_len = self.dialogues_sent_len_list
        dia_len = self.dialogues_len_list

        print("Total {} dialogues in {} mode".format(len(x1), mode))

        for i in tqdm(range(0, len(label_list), batch_size), desc="Processing:"):
            batch_x1 = pad_sequences(x1[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x2 = pad_sequences(x2[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x3 = pad_sequences(x3[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_sent_len = pad_sequences(sent_len[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                           dtype='int32')
            batch_dia_len = dia_len[i: i + batch_size]
            labels_padded = pad_sequences(label_list[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                          dtype='int32', value=0)

            yield batch_x1, batch_x2, batch_x3, labels_padded, batch_sent_len, batch_dia_len


if __name__ == '__main__':

    mode_list = ['train', 'eval', 'test']
    data_name_list = ['clothing', 'makeup']
    for data_name in data_name_list:
        # gen vocabulary
        data_prepare = DataPrepare(mode='vocab',
                                   data_name=data_name)
        data_prepare.gen_vocab(min_cnt=2)
        for mode in mode_list:
            data_prepare = DataPrepare(mode=mode,
                                       data_name=data_name)
            # save2pkl
            data_prepare.save_data(mode=mode)

        data_prepare.desensitization()


