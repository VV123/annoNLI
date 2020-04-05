import sys
import os
import keras
import argparse
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils.data_manager import load_data
import utils.vocab as vocab
from utils.vocab import load_vocab_all
from utils.bleu import moses_multi_bleu
from collections import defaultdict
from argparse import ArgumentParser
from tensorflow.python.layers.core import Dense

from tf_utils.attention_wrapper import AttentionWrapper, BahdanauAttention
from tf_utils.beam_search_decoder import BeamSearchDecoder
from tf_utils.decoder import dynamic_decode
from tf_utils.basic_decoder import BasicDecoder

_PAD = vocab._PAD
_GO = vocab._GO
_END = vocab._END


def _decoder(encoder_outputs, encoder_state, mode, beam_width, batch_size,
             args):
    """
    Decoder with BahdanauAttention
    """
    num_units = 2 * args.dim
    memory = encoder_outputs
    if mode == "infer":
        memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        batch_size = batch_size * beam_width
    else:
        batch_size = batch_size

    seq_len = tf.tile(tf.constant([args.maxlen], dtype=tf.int32), [batch_size])
    attention_mechanism = BahdanauAttention(
        num_units=num_units,
        memory=memory,
        normalize=True,
        memory_sequence_length=seq_len)

    cell0 = tf.contrib.rnn.GRUCell(2 * args.dim)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell0,
        input_keep_prob=1 - args.in_drop,
        output_keep_prob=1 - args.out_drop)
    cell = AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=num_units,
        name="attention")

    decoder_initial_state = cell.zero_state(
        batch_size, tf.float32).clone(cell_state=encoder_state)

    return cell, decoder_initial_state


def Decoder(mode, enc_rnn_out, enc_rnn_state, X, emb_Y, emb_out, args):

    with tf.variable_scope("Decoder") as decoder_scope:

        mem_units = 2 * args.dim
        out_layer = Dense(args.output_vocab_size)  # projection W*X+b
        beam_width = args.beam_width
        batch_size = tf.shape(enc_rnn_out)[0]

        cell, initial_state = _decoder(enc_rnn_out, enc_rnn_state, mode,
                                       beam_width, batch_size, args)

        if mode == "train":
            seq_len = tf.tile(
                tf.constant([args.maxlen], dtype=tf.int32), [batch_size])
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=emb_Y, sequence_length=seq_len)

            decoder = BasicDecoder(
                cell=cell,
                helper=helper,
                initial_state=initial_state,
                X=X,
                output_layer=out_layer)
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                maximum_iterations=args.maxlen,
                scope=decoder_scope)
            logits = outputs.rnn_output
            sample_ids = outputs.sample_id
        else:
            start_tokens = tf.tile(
                tf.constant([_GO], dtype=tf.int32), [batch_size])
            end_token = _END
            my_decoder = BeamSearchDecoder(
                cell=cell,
                embedding=emb_out,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=initial_state,
                beam_width=beam_width,
                X=X,
                output_layer=out_layer,
                length_penalty_weight=0.0)

            outputs, t1, t2 = tf.contrib.seq2seq.dynamic_decode(
                my_decoder,
                maximum_iterations=args.maxlen,
                scope=decoder_scope)
            logits = tf.no_op()
            sample_ids = outputs.predicted_ids

    return logits, sample_ids


#----------------------------------------------------------------------------------------------
def construct_graph(mode, env, args):

    _, _, vocab_emb, train_idx = load_vocab_all(args)
    emb_out = tf.get_variable("emb_out", initializer=vocab_emb)
    emb_X = tf.nn.embedding_lookup(emb_out, env.x)
    emb_Y = tf.nn.embedding_lookup(emb_out, env.y)

    with tf.name_scope("Encoder"):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            cell_fw = tf.contrib.rnn.GRUCell(args.dim)
            cell_bw = tf.contrib.rnn.GRUCell(args.dim)
            input_layer = Dense(
                args.dim, dtype=tf.float32, name='input_projection')
            emb_X = input_layer(emb_X)
            enc_rnn_out, enc_rnn_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, emb_X, dtype=tf.float32)
            enc_rnn_out = tf.concat(enc_rnn_out, 2)
        with tf.variable_scope("bidirectional-rnn1"):
            cell_fw1 = tf.contrib.rnn.GRUCell(args.dim)
            cell_bw1 = tf.contrib.rnn.GRUCell(args.dim)
            input_layer1 = Dense(
                args.dim, dtype=tf.float32, name='input_projection1')
            enc_rnn_out = input_layer1(enc_rnn_out)
            enc_rnn_out, enc_rnn_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw1, cell_bw1, enc_rnn_out, dtype=tf.float32)
            enc_rnn_out = tf.concat(enc_rnn_out, 2)
        enc_rnn_state = tf.concat([enc_rnn_state[0], enc_rnn_state[1]], axis=1)

    logits, sample_ids = Decoder(mode, enc_rnn_out, enc_rnn_state, env.x,
                                 emb_Y, emb_out, args)
    if mode == 'train':

        env.pred = tf.concat(
            (env.y[:, 1:], tf.zeros((tf.shape(env.y)[0], 1), dtype=tf.int32)),
            axis=1)
        env.loss = tf.losses.softmax_cross_entropy(
            tf.one_hot(env.pred, args.output_vocab_size), logits)
        optimizer = tf.train.AdamOptimizer(args.lr)
        optimizer.minimize(env.loss)
        gvs = optimizer.compute_gradients(env.loss)

        capped_gvs = [(tf.clip_by_norm(grad, args.clip_grad), var)
                      for grad, var in gvs]
        env.train_op = optimizer.apply_gradients(capped_gvs)
        a = tf.equal(sample_ids, env.pred)
        b = tf.reduce_all(a, axis=1)
        env.acc = tf.reduce_mean(tf.cast(b, dtype=tf.float32))
    else:
        sample_ids = tf.transpose(sample_ids, [0, 2, 1])
        env.acc = None
        env.loss = None
        env.train_op = None

    return env.train_op, env.loss, env.acc, sample_ids, logits
