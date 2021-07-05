import os

import sys
import json
import argparse
import pickle as pkl
import random
import logging
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
prodir = '..'
sys.path.insert(0, prodir)

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class Data_loader(object):
    def __init__(self, mode='test', data_name='makeup', log_path=None):
        """
        Constant variable declaration and configuration.
        """
        if data_name == 'clothing':
            dataset_folder_name = '/data' + '/clothing'
        elif data_name == 'makeup':
            dataset_folder_name = '/data' + '/makeup'
        else:
            raise ValueError("Please confirm the correct data name you entered.")

        self.vocab_save_path = curdir + dataset_folder_name + '/vocab.pkl'
        self.train_path = curdir + dataset_folder_name + '/train.pkl'
        self.val_path = curdir + dataset_folder_name + '/eval.pkl'
        self.test_path = curdir + dataset_folder_name + '/test.pkl'

        self.logger = logging.getLogger("Data Preprocessing")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.role_list = []
        self.pos_list = []
        self.dialogues_ids_list = []
        self.dialogues_len_list = []
        self.dialogues_sent_len_list = []
        self.label_list = []
        self.senti_list = []
        self.tf_list = []

        self.mode = mode

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
            self.logger.info("Load variable from {} successfully!".format(load_path))

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as fp:
            return json.load(fp)

    def data_generator_sup(self, data_name='makeup', mode='test', batch_size=32, shuffle=True, nb_classes=2, epoch=0):
        print('Using data_generator_sup')
        self.load_pkl_data(mode=mode)
        x1 = self.dialogues_ids_list
        x2 = self.role_list
        x3 = self.senti_list
        label_list = self.label_list
        sent_len = self.dialogues_sent_len_list
        dia_len = self.dialogues_len_list

        if shuffle or mode == 'train':
            list_pack = list(zip(x1, x2, x3, label_list, sent_len, dia_len))
            random.seed(epoch + 7)
            random.shuffle(list_pack)
            x1[:], x2[:], x3[:], label_list[:], sent_len[:], dia_len[:] = zip(*list_pack)

        for i in tqdm(range(0, len(dia_len), batch_size), desc="Processing:"):
            batch_x1 = pad_sequences(x1[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x2 = pad_sequences(x2[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x3 = x3[i: i + batch_size]

            batch_sent_len = pad_sequences(sent_len[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                           dtype='int32')
            batch_dia_len = dia_len[i: i + batch_size]
            # [B, D_len, nb_classes]
            labels_padded = pad_sequences(label_list[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                          dtype='int32', value=0)
            batch_labels = to_categorical(labels_padded, nb_classes, dtype='int32')

            yield batch_x1, batch_x2, batch_x3, batch_labels, batch_sent_len, batch_dia_len

    def data_generator_crf(self, data_name='makeup', mode='test', batch_size=32, shuffle=True, nb_classes=2, epoch=0):
        print('Using data_generator_crf')
        self.load_pkl_data(mode=mode)
        x1 = self.dialogues_ids_list
        x2 = self.role_list
        x3 = self.senti_list
        label_list = self.label_list
        sent_len = self.dialogues_sent_len_list
        dia_len = self.dialogues_len_list

        if shuffle or mode == 'train':
            list_pack = list(zip(x1, x2, x3, label_list, sent_len, dia_len))
            random.seed(epoch + 7)
            random.shuffle(list_pack)
            x1[:], x2[:], x3[:], label_list[:], sent_len[:], dia_len[:] = zip(*list_pack)

        for i in tqdm(range(0, len(dia_len), batch_size), desc="Processing:"):
            batch_x1 = pad_sequences(x1[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x2 = pad_sequences(x2[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x3 = x3[i: i + batch_size]
            batch_sent_len = pad_sequences(sent_len[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                           dtype='int32')
            batch_dia_len = dia_len[i: i + batch_size]
            labels_padded = pad_sequences(label_list[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                          dtype='int32', value=0)

            yield batch_x1, batch_x2, batch_x3, labels_padded, batch_sent_len, batch_dia_len

    def data_generator_m(self, data_name='makeup', mode='test', batch_size=32, shuffle=True, nb_classes=2, epoch=0):
        print('Using data_generator_crf')
        self.load_pkl_data(mode=mode)
        x1 = self.dialogues_ids_list
        x2 = self.role_list
        x3 = self.senti_list

        tf_list = self.tf_list
        pos_list = self.pos_list

        label_list = self.label_list
        sent_len = self.dialogues_sent_len_list
        dia_len = self.dialogues_len_list

        if shuffle or mode == 'train':
            list_pack = list(zip(x1, x2, x3, tf_list, pos_list, label_list, sent_len, dia_len))
            random.seed(epoch + 7)
            random.shuffle(list_pack)
            x1[:], x2[:], x3[:], tf_list[:], pos_list[:], label_list[:], sent_len[:], dia_len[:] = zip(*list_pack)

        for i in tqdm(range(0, len(dia_len), batch_size), desc="Processing:"):
            batch_x1 = pad_sequences(x1[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x2 = pad_sequences(x2[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                     dtype='float32')
            batch_x3 = x3[i: i + batch_size]

            batch_tf = tf_list[i: i + batch_size]
            batch_paded_pos = pad_sequences(pos_list[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                            dtype='float32', value=0)
            batch_pos = to_categorical(batch_paded_pos, num_classes=52, dtype='int32')

            batch_sent_len = pad_sequences(sent_len[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                           dtype='int32')
            batch_dia_len = dia_len[i: i + batch_size]
            labels_padded = pad_sequences(label_list[i: i + batch_size], maxlen=30, padding='post', truncating='post',
                                          dtype='int32', value=0)
            batch_labels = to_categorical(labels_padded, nb_classes, dtype='int32')

            yield batch_x1, batch_x2, batch_x3, batch_labels, batch_sent_len, batch_dia_len, batch_tf, batch_pos


if __name__ == "__main__":
    pass
