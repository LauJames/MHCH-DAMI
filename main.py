#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

prodir = '..'
sys.path.insert(0, prodir)

import pickle as pkl
import logging
import argparse
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
from Network import Network
from networks.DAMI import DAMI

from data_loader import Data_loader
from utility import *

CONFIG_ROOT = curdir + '/config'
random_seed = 7
tf.compat.v1.set_random_seed(random_seed)


def main():
    start_t = time.time()
    # Obtain arguments from system
    parser = argparse.ArgumentParser('Tensorflow')
    parser.add_argument('--phase', default='train',
                        help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--data_name', default='makeup',
                        help='Data_Name: The data you will use.')
    parser.add_argument('--model_name', default='dami',
                        help='Model_Name: The model you will use.')
    parser.add_argument('--model_path', default='none',
                        help='Model_Path: The model path you will load.')
    parser.add_argument('--memory', default='0.',
                        help='Memory: The gpu memory you will use.')
    parser.add_argument('--gpu', default='0',
                        help='GPU: Which gpu you will use.')
    parser.add_argument('--log_path', default='./networks/logs/',
                        help='path of the log file. If not set, logs are printed to console.')
    parser.add_argument('--suffix', default='.128',
                        help='suffix for differentiate log.')
    parser.add_argument('--mode', default='train',
                        help='mode fold you will try.')
    parser.add_argument('--ways', default='dami',
                        help='whether to use supervised method or position or deal.')
    parser.add_argument('--use_pretrain', default='1',
                        help='whether to use supervised method or position or deal.')

    args = parser.parse_args()

    logger = logging.getLogger("Tensorflow")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' ')[:3])

    # log directory
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    # log file name setting
    log_path = args.log_path + args.model_name + '.' + args.data_name + '.' + args.phase + args.suffix \
               + '.' + args.mode + '.' + args.ways + '.' + now_time + '.log'

    if os.path.exists(log_path):
        os.remove(log_path)

    if args.log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Random seed: {}'.format(random_seed))
    logger.info('Running with args : {}'.format(args))

    # get object named data_loader
    data_loader = Data_loader(data_name=args.data_name)

    # Get config from file
    logger.info('Load data_set and vocab...')

    data_config_path = CONFIG_ROOT + '/data/config.' + args.data_name + '.json'
    model_config_path = CONFIG_ROOT + '/model/config.' + args.model_name + '.json'
    data_config = data_loader.load_config(data_config_path)
    model_config = data_loader.load_config(model_config_path)

    logger.info('Data config is {}'.format(data_config))
    logger.info('Model config is {}'.format(model_config))

    # Get config param
    model_name = model_config['model_name']
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']
    keep_prob = model_config['keep_prob']

    mode = model_config['mode']
    is_val = model_config['is_val']
    is_test = model_config['is_test']
    save_best = model_config['save_best']
    shuffle = model_config['shuffle']

    data_name = data_config['data_name']
    nb_classes = data_config['nb_classes']

    mode = args.mode

    vocab_path = curdir + '/data/' + data_name + '/vocab.pkl'

    memory = float(args.memory)
    logger.info("Memory in train %s." % memory)

    # Get vocab
    with open(vocab_path, 'rb') as fp:
        vocab = pkl.load(fp)

    # Get Network Framework
    if model_name == 'network':
        network = Network(memory=memory, vocab=vocab)
    elif model_name == 'dami':
        network = DAMI(memory=memory, vocab=vocab, config_dict=model_config)
    else:
        logger.info("We can't find {}: Please check model you want."
                    .format(model_name))
        raise ValueError("We can't find {}: Please check model you want."
                         .format(model_name))

    # Set param for network
    network.set_nb_words(min(vocab.size(), data_config['nb_words']) + 1)
    network.set_data_name(data_name)
    network.set_name(model_name + args.suffix + 'train')
    network.set_from_model_config(model_config)
    network.set_from_data_config(data_config)

    if 'sup' in args.ways:
        print('Using data_generator_sup')
        data_generator = data_loader.data_generator_sup
    elif args.ways == 'crf':
        print('Using data_generate_crf')
        data_generator = data_loader.data_generator_crf
    elif args.ways == 'dami':
        print('Using data_generator_m')
        data_generator = data_loader.data_generator_m
    else:
        raise ValueError("Wrong data generator! Please check the 'ways' you input.")

    network.build_graph()

    logger.info('All values in the Network are {}'.format(network.__dict__))

    if args.phase == 'train':
        train(network, data_generator, keep_prob, epochs, data_name,
              mode=mode, batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle,
              is_val=is_val, is_test=is_test, save_best=save_best, ways=args.ways)
    else:
        logger.info("{}: Please check phase you want, such as 'train' or 'evaluate'.".format(args.phase))
        raise ValueError("{}: Please check phase you want, such as 'train' or 'evaluate'.".format(args.phase))

    logger.info('The whole program spends time: {}h: {}m: {}s'.format(int((int(time.time()) - start_t) / 3600),
                                                                      int((int(time.time()) - start_t) % 3600 / 60),
                                                                      int((int(time.time()) - start_t) % 3600 % 60)))
    print("DONE!")


def train(network, data_generator, keep_prob, epochs, data_name,
          mode='train', batch_size=20, nb_classes=2, shuffle=True,
          is_val=True, is_test=True, save_best=True, ways='crf'):
    if ways == 'crf':
        network.train_crf(data_generator=data_generator, keep_prob=keep_prob, epochs=epochs, data_name=data_name,
                          mode=mode, batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle,
                          is_val=is_val, is_test=is_test, save_best=save_best)
    elif ways == 'sup':
        network.train_sup(data_generator=data_generator, keep_prob=keep_prob, epochs=epochs, data_name=data_name,
                          mode=mode, batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle,
                          is_val=is_val, is_test=is_test, save_best=save_best)
    elif ways == 'dami':
        network.train(data_generator=data_generator, keep_prob=keep_prob, epochs=epochs, data_name=data_name,
                      mode=mode, batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle,
                      is_val=is_val, is_test=is_test, save_best=save_best)
    else:
        raise ValueError("Wrong data generator! Please check the 'ways' you input.")


if __name__ == '__main__':
    main()
