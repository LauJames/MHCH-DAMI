#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import resource
import time
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import numpy as np
import math


def get_now_time():
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' '))  # [:3])
    return now_time


def print_trainable_variables(output_detail, logger):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.compat.v1.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d\n" % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d\n" % (variable.name, str(shape), variable_parameters))

    if logger:
        if output_detail:
            logger.info('\n' + parameters_string)
        logger.info("Total %d variables, %s params" % (len(tf.compat.v1.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print('\n' + parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))


def print_all_variables(output_detail, logger=None):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.all_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d\n" % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d\n" % (variable.name, str(shape), variable_parameters))

    if logger is not None:
        if output_detail:
            logger.info('\n' + parameters_string)
        logger.info("Total %d variables, %s params" % (len(tf.all_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print('\n' + parameters_string)
        print("Total %d variables, %s params" % (len(tf.all_variables()), "{:,}".format(total_parameters)))


def show_layer_info(layer_name, layer_out, logger=None):
    if logger:
        logger.info('[layer]: %s\t[shape]: %s'
                    % (layer_name, str(layer_out.get_shape().as_list())))
    else:
        print('[layer]: %s\t[shape]: %s'
              % (layer_name, str(layer_out.get_shape().as_list())))


def show_layer_info_with_memory(layer_name, layer_out, logger=None):
    if logger:
        logger.info('[layer]: %s\t[shape]: %s \n%s'
                    % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))
    else:
        print('[layer]: %s\t[shape]: %s \n%s'
              % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))


def show_memory_use():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom = rusage_denom * rusage_denom
    ru = resource.getrusage(resource.RUSAGE_SELF)
    total_memory = 1. * (ru.ru_maxrss + ru.ru_ixrss +
                         ru.ru_idrss + ru.ru_isrss) / rusage_denom
    strinfo = "\x1b[33m [Memory] Total Memory Use: %.4f MB \t Resident: %ld Shared: %ld UnshareData: " \
              "%ld UnshareStack: %ld \x1b[0m" % \
              (total_memory, ru.ru_maxrss, ru.ru_ixrss, ru.ru_idrss, ru.ru_isrss)
    return strinfo


def get_a_p_r_f_sara(target, predict, category):
    idx = np.array(range(len(target)))
    _target = set(idx[target == category])
    _predict = set(idx[predict == category])
    _target_0 = set(idx[target == 0])
    _predict_0 = set(idx[predict == 0])
    true = _target & _predict
    true_0 = _target_0 & _predict_0
    accuracy = float(np.sum(np.array(target)==np.array(predict)))/float(len(idx))
    precision = len(true) / float(len(_predict) + 0.0000000001)
    precision_0 = len(true_0) / float(len(_predict_0) + 0.0000000001)
    recall = len(true) / float(len(_target) + 0.0000000001)
    recall_0 = len(true_0) / float(len(_target_0) + 0.0000000001)
    f1_score = precision * recall * 2 / (precision + recall + 0.0000000001)
    macro_f1_score = (precision+precision_0)/2 * (recall+recall_0)/2 * 2 /\
                     ((precision+precision_0)/2 + (recall+recall_0)/2 + 0.0000000001)
    f0_5_score = precision * recall * (1+0.5*0.5) / (0.5*0.5*precision + recall + 0.0000000001)
    f2_score = precision * recall * (1+2*2) / (2*2*precision + recall + 0.0000000001)
    return accuracy, precision, recall, f1_score, macro_f1_score, f0_5_score, f2_score


def golden_switch_within_tolerance_exp(pre_labels, true_labels, t=1, eps=1e-7, lamb=0):
    if t <= 0:
        raise ValueError("Tolerance must be positive!!!")
    if not isinstance(t, int):
        raise TypeError("Tolerance must be Integer!!!")

    gst_score = 0
    # get suggest switch position according to true labels
    suggest_indices = []
    for idx, label in enumerate(true_labels):
        if label == 1:
            suggest_indices.append(idx)
    
    pre_indices = []
    for idx, label in enumerate(pre_labels):
        if label == 1:
            pre_indices.append(idx)

    if len(suggest_indices) == 0:
        if len(pre_indices) == 0:
            gst_score = 1
        else:
            gst_score = 0
    else:
        if len(pre_indices) == 0:
            gst_score = 0
        else:
            GST_score_list = []
            for pre_idx in pre_indices:
                tmp_score_list = []
                for suggest_idx in suggest_indices:
                    # suggest_idx is q_i
                    # pre_idx is p_i
                    pre_bias = pre_idx - suggest_idx
                    adjustment_cofficient = 1. / (1 - lamb * (np.sign(pre_bias)))
                    tmp_score = math.exp(- adjustment_cofficient * math.pow(pre_bias, 2) / (2 * math.pow((t + eps), 2)))
                    tmp_score_list.append(tmp_score)
                GST_score_list.append(np.max(tmp_score_list))
            gst_score = np.mean(GST_score_list)
    return gst_score


def get_gst_score(label_list, pre_list, lamb=0.):
    gst_score_list_1 = []
    gst_score_list_2 = []
    gst_score_list_3 = []
    for pres, labels in zip(pre_list, label_list):
        gst_score_list_1.append(golden_switch_within_tolerance_exp(pres, labels, t=1, lamb=lamb))
        gst_score_list_2.append(golden_switch_within_tolerance_exp(pres, labels, t=2, lamb=lamb))
        gst_score_list_3.append(golden_switch_within_tolerance_exp(pres, labels, t=3, lamb=lamb))

    return np.mean(gst_score_list_1), np.mean(gst_score_list_2), np.mean(gst_score_list_3)

if __name__ == "__main__":
    pass

