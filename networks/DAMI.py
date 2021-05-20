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
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
from utility import *
from keras.utils import to_categorical
from Network import Network
from networks.layers.attention import *
from networks.layers.transformer import *


class DAMI(Network):
    def __init__(self, memory=0, vocab=None, config_dict=None, **kwargs):
        Network.__init__(self, memory=memory, vocab=vocab)
        self.model_name = self.__class__.__name__
        self.logger.info("Model Name: {}".format(self.model_name))

        self.rnn_dim = config_dict["rnn_dim"]
        self.dense_dim = config_dict["dense_dim"]
        self.pos_dim = config_dict["pos2id"]
        
        self.lr = config_dict["learning_rate"]
        self.l2_reg_lambda = config_dict["l2_reg_lambda"]
        self.weight_decay = config_dict["weight_decay"]

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.input_x1 = tf.compat.v1.placeholder(tf.int32, [None, self.dia_max_len, self.sent_max_len], name='input_x1')
        # role
        self.input_x2 = tf.compat.v1.placeholder(tf.float32, [None, self.dia_max_len], name="input_x2")
        # sentiment
        self.input_x3 = tf.compat.v1.placeholder(tf.float32, [None, self.dia_max_len], name="input_x3")

        self.tfs = tf.compat.v1.placeholder(tf.float32, [None, self.dia_max_len, self.sent_max_len], name='tfs')
        self.pos_list = tf.compat.v1.placeholder(tf.int32, [None, self.dia_max_len, self.sent_max_len, self.pos_dim], name='pos_list')
        self.sent_len = tf.compat.v1.placeholder(tf.int32, [None, self.dia_max_len], name='input_x_sent_len')
        self.dia_len = tf.compat.v1.placeholder(tf.int32, [None], name='input_x_len')
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, self.dia_max_len, self.nb_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.logger.info("setup placeholders.")

    def _inference(self):
        """
        encode sentence information
        """
        # [B, D_len, S_len, E_dim]
        self.embedded = tf.nn.embedding_lookup(self.word_embeddings, self.input_x1)
        if self.dropout_keep_prob != 1:
            self.embedded = tf.nn.dropout(self.embedded, rate=1-self.dropout_keep_prob)
            self.tfs = tf.nn.dropout(self.tfs, rate=1-self.dropout_keep_prob)
        # [B, D_len, S_len, pos_list_dim]
        self.pos_list = tf.cast(self.pos_list, dtype=tf.float32)
        show_layer_info_with_memory('embedded', self.embedded, self.logger)
        
        # sentiment score expand dim
        self.senti_score = tf.expand_dims(self.input_x3, axis=-1)

        # [B * D_len, S_len, E_dim]
        self.embedded_reshaped = tf.reshape(self.embedded, shape=[-1, self.sent_max_len, self.embedding_dim])
        self.tfs_reshaped = tf.expand_dims(tf.reshape(self.tfs, shape=[-1, self.sent_max_len]), axis=-1)
        self.pos_list_reshaped = tf.reshape(self.pos_list, shape=[-1, self.sent_max_len, self.pos_dim])
        
        show_layer_info_with_memory('embedded_reshaped', self.embedded_reshaped, self.logger)
        self.sent_len_reshape = tf.reshape(self.sent_len, shape=[-1])

        # word pos embedding
        # [B * D_len, S_len, E_dim]
        self.sent_pos_emb = positional_encoding(self.embedded_reshaped, self.sent_max_len)
        self.embedded_all = tf.concat([self.embedded_reshaped, self.sent_pos_emb, self.pos_list_reshaped], axis=-1)

        with tf.name_scope('sent_encoding'):
            # [B * D_len, S_len, 2 * H]; [B * D_len, 2 * H]
            self.sent_encoder_output, self.sent_encoder_state = self._bidirectional_rnn(self.embedded_all, self.sent_len_reshape, rnn_type='lstm')
            
            with tf.variable_scope('sent_difficulty_atten'):
                # Trainable parameters
                hidden_size = self.embedded_all.get_shape().as_list()[-1]  # E_dim.

                w_omega_c = tf.compat.v1.Variable(tf.random.normal([hidden_size, 2 * self.rnn_dim], stddev=0.1))
                w_omega_a = tf.compat.v1.Variable(tf.random.normal([hidden_size, 2 * self.rnn_dim], stddev=0.1))

                b_omega_c = tf.get_variable("b_omega_c", [2 * self.rnn_dim], initializer=tf.zeros_initializer())
                b_omega_a = tf.get_variable("b_omega_a", [2 * self.rnn_dim], initializer=tf.zeros_initializer())

                u_omega_c = tf.get_variable("u_omega_c", [2 * self.rnn_dim], initializer=tf.ones_initializer())
                u_omega_a = tf.get_variable("u_omega_a", [2 * self.rnn_dim], initializer=tf.ones_initializer())

                # [B * D_len]
                self.agent_tag = tf.reshape(self.input_x2, shape=[-1])
                self.customer_tag = tf.add(tf.negative(self.agent_tag), 1)
                
                # [B * D_len, S_len]
                self.agent_tag_sent = tf.tile(tf.expand_dims(self.agent_tag, axis=-1), multiples=[1, 50])
                self.customer_tag_sent = tf.tile(tf.expand_dims(self.customer_tag, axis=-1), multiples=[1, 50])

                with tf.name_scope('v_agent'):
                    # [B * D_len, S_len, 2H]
                    qk_a = tf.tanh(tf.matmul(self.embedded_all, w_omega_a) + b_omega_a)
                    # [B * D_len, S_len]
                    vu_a = tf.nn.relu(tf.tensordot(qk_a * (1- self.tfs_reshaped), u_omega_a, axes=1, name='vu_a'))

                    # padding mask
                    # [B * D_len, S_len]
                    self.dif_mask = tf.sequence_mask(self.sent_len_reshape, maxlen=self.sent_max_len, dtype=tf.float32)

                    sent_paddings = tf.ones_like(self.dif_mask) * (-2**32+1)
                    self.vu_masked_a = tf.where(tf.equal(self.dif_mask, 0), sent_paddings, vu_a)
                    self.vu_masked_a = tf.where(tf.equal(self.agent_tag_sent, 0), sent_paddings, self.vu_masked_a)
                    self.alphas_a = tf.nn.softmax(self.vu_masked_a)
                    
                with tf.name_scope('v_customer'):
                    qk_c = tf.tanh(tf.matmul(self.embedded_all, w_omega_c) + b_omega_c)
                    vu_c = tf.tensordot(qk_c * (1- self.tfs_reshaped), u_omega_c, axes=1, name='vu_c') 
                    self.vu_masked_c = tf.where(tf.equal(self.dif_mask, 0), sent_paddings, vu_c)
                    self.vu_masked_c = tf.where(tf.equal(self.customer_tag_sent, 0), sent_paddings, self.vu_masked_c)
                    self.alphas_c = tf.nn.softmax(self.vu_masked_c)

            # reduce with attention vector
            self.alphas = self.alphas_a + self.alphas_c
            self.sent_encoder_output_atten = tf.reduce_sum(self.sent_encoder_output * tf.expand_dims(self.alphas, -1), 1)

            self.sent_encoder_attened_reshape = tf.reshape(self.sent_encoder_output_atten, shape=[-1, self.dia_max_len, 2 * self.rnn_dim])
            self.sent_encoder_state_reshape = tf.reshape(self.sent_encoder_state, shape=[-1, self.dia_max_len, 2 * self.rnn_dim])

            self.combine_emb = tf.concat([self.sent_encoder_attened_reshape, self.sent_encoder_state_reshape, self.senti_score], axis=-1)
            if self.dropout_keep_prob != 1:
                # [B * D_len, 2 * H]
                self.combine_emb = tf.nn.dropout(self.combine_emb, rate=1-self.dropout_keep_prob)

        with tf.name_scope('sequence_context_encoding'):
            with tf.compat.v1.variable_scope('local_inference'):
                self.cross_match = tf.matmul(self.combine_emb, tf.transpose(self.combine_emb, [0, 2, 1]))
            
                self.cross_match_upper = tf.matrix_band_part(self.cross_match, num_lower=0, num_upper=-1)
                self.cross_match_sim_one_direct = self.cross_match - self.cross_match_upper

                self.encode_local = tf.concat([self.combine_emb, self.cross_match_sim_one_direct], axis=-1)
                self.encode_local = tf.keras.layers.Dense(units=self.dense_dim, activation='relu', use_bias=True)(self.encode_local)

                if self.dropout_keep_prob != 1:
                    self.encode_local = tf.nn.dropout(self.encode_local, rate=1-self.dropout_keep_prob)
            
            with tf.compat.v1.variable_scope('composition_RNN'):
                # [B, D_len, 2 * H], [B, 2 * H]
                cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.rnn_dim, state_is_tuple=True), output_keep_prob=self.dropout_keep_prob)]
                rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                self.v_dia_encode_output, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.encode_local, sequence_length=self.dia_len, dtype=tf.float32)

        with tf.name_scope('dialogue_mask'):
            self.dia_seq_mask = tf.sequence_mask(self.dia_len, maxlen=self.dia_max_len, dtype=tf.float32)

        with tf.compat.v1.variable_scope('context-aware'):
            w_datt = tf.compat.v1.Variable(tf.random.normal([self.rnn_dim, self.rnn_dim]))
            self.d_att = tf.matmul(tf.matmul(self.v_dia_encode_output, w_datt), tf.transpose(self.v_dia_encode_output, perm=[0, 2, 1]))
            # padding mask
            self.d_att_mask = tf.tile(tf.expand_dims(self.dia_seq_mask, axis=-1), multiples=[1, 1, self.dia_max_len])
            paddings = tf.ones_like(self.d_att) * (-2**32+1)
            self.a_att_masked = tf.where(tf.equal(self.d_att_mask, 0), paddings, self.d_att)
            # sequence mask
            self.d_att_alpha = tf.nn.softmax(tf.matrix_band_part(self.a_att_masked, num_lower=-1, num_upper=0))
            
            self.context_encode = tf.matmul(self.d_att_alpha, self.v_dia_encode_output)
            
            self.concat_d = tf.concat([self.context_encode, self.v_dia_encode_output], axis=-1)
            self.combine_h = tf.keras.layers.Dense(units=self.dense_dim, activation='relu')(self.concat_d)

            if self.dropout_keep_prob != 1:
                self.combine_h = tf.nn.dropout(self.combine_h, rate=1-self.dropout_keep_prob)

        with tf.name_scope("output"):
            # [B, D_len, nb_classes]
            self.logits = tf.keras.layers.Dense(units=self.nb_classes, activation='softmax')(self.combine_h)
            
            # [B, D_len]
            self.output = tf.argmax(self.logits, axis=-1)
            self.proba = tf.nn.softmax(self.logits)

        self.logger.info("network inference.")

    def _compute_loss(self):
        """
        The loss function
        """
        def nll_loss(probs, labels, epsilon=1e-9, scope=None, nb_classes=2):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                losses = - tf.reduce_sum(labels * tf.math.log(probs + epsilon), -1)
            return losses
        # masked loss
        self.cross_entropy = nll_loss(self.logits, self.input_y)
        self.mask_cross_entropy = self.dia_seq_mask * self.cross_entropy
        self.scale_cross_entropy = tf.reduce_sum(self.mask_cross_entropy, -1) / tf.cast(self.dia_len, tf.float32)
        self.loss = tf.reduce_mean(self.scale_cross_entropy)
        self.logger.info("Calculate Loss.")

        self.all_params = tf.compat.v1.trainable_variables()
        if self.l2_reg_lambda > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.l2_reg_lambda * l2_loss
            self.logger.info("Add L2 Loss.")

