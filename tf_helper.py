from __future__ import print_function
import sys
import os
import keras
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils.data_manager import load_data
from utils.vocab import load_vocab_all
from utils.bleu import moses_multi_bleu
from collections import defaultdict
from argparse import ArgumentParser
import utils.vocab as vocab
import sys
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_PAD = vocab._PAD
_GO = vocab._GO
_END = vocab._END

def train(sess, env, X_data, y_data, epochs=10, load=False, shuffle=True, batch_size=128,
          name='model', base=0, model2load=''):
    """
    Train TF model by env.train_op
    """
    if load:
        print('\nLoading saved model')
        env.saver.restore(sess, model2load )

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch+1, epochs))
        sys.stdout.flush()
        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
            start = batch * batch_size
            end = min(n_sample, start+batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        
        evaluate(sess, env, X_data, y_data, batch_size=batch_size)

        if (epoch+1) == epochs:
            print('\n Saving model')
            env.saver.save(sess, 'model/{0}-{1}'.format(name, base))
    return 'model/{0}-{1}'.format(name, base) 

def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss,env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return acc

def _decode_data(sess, env, X_data, batch_size, reverse_vocab_dict):
    print('\nDecoding')
    logics_all = []
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        ybar = sess.run(env.pred_ids,
                        feed_dict={env.x: X_data[start:end]})
        ybar = np.asarray(ybar)
        ybar = np.squeeze(ybar[:,0,:])  # pick top prediction
        for seq in ybar:
            seq = np.append(seq, _END)
            seq = seq[:list(seq).index(_END)]
            logic = " ".join([reverse_vocab_dict[idx] for idx in seq])
            logics_all.append(logic)
    return logics_all

def decode_data_recover(sess, env, args, X_data, y_data, s, batch_size=128):
    """
    Inference and calculate EM acc based on recovered SQL
    """
    annotation_path = args.annotation_path
    i, acc = 0, 0
    _, reverse_vocab_dict, _, _ = load_vocab_all(args)
    inf_logics = _decode_data(sess, env, X_data, batch_size, reverse_vocab_dict) 
    xtru, ytru = X_data, y_data
    with gfile.GFile(annotation_path+'%s_infer.txt'%s, mode='w') as output, gfile.GFile(annotation_path+'%s_ground_truth.txt'%s, mode='r') as S_ori_file, \
        gfile.GFile(annotation_path+'%s_sym_pairs.txt'%s, mode='r') as sym_pair_file:

        sym_pairs = sym_pair_file.readlines()  # annotation pairs from question & table files
        S_oris = S_ori_file.readlines()  # SQL files before annotation
        for true_seq, logic, x, sym_pair, S_ori in zip(ytru, inf_logics, xtru, sym_pairs, S_oris):
            sym_pair = sym_pair.replace('<>\n','')
            S_ori = S_ori.replace('\n','')
            Qpairs = []
            for pair in sym_pair.split('<>'):
                Qpairs.append(pair.split('=>'))
            
            true_seq = true_seq[1:]    #delete <bos>
            x = x[1:]   #delete <bos>
  
            true_seq = true_seq[:list(true_seq).index(_END)]
            x = np.append(x, _END)
            x = x[:list(x).index(_END)]

            xseq = " ".join([reverse_vocab_dict[idx] for idx in x])
            true_logic = " ".join([reverse_vocab_dict[idx] for idx in true_seq])

            logic = logic.replace(' (','').replace(' )','')
            true_logic = true_logic.replace(' (','').replace(' )','') 

            logic = _switch_cond(logic, true_logic)

            recover_S = logic
            for sym, word in Qpairs:
                recover_S = recover_S.replace(sym, word) 

            acc += (recover_S==S_ori)
            output.write(recover_S + '\n')
            i += 1
    
    print('EM: %.4f'%(acc*1./len(y_data)))  
    print('number of correct ones:%d'%acc)
    
    return acc*1./len(y_data)

def decode_data(sess, env, args, X_data, y_data, batch_size=128, filename='output.txt'):
    """
    Inference w/o recover annotation symbols
    """
    i, acc = 0, 0
    _, reverse_vocab_dict, _, _ = load_vocab_all(args)
    logics_all = _decode_data(sess, env, X_data, batch_size, reverse_vocab_dict)
    xtru, ytru = X_data, y_data

    for true_seq, logic, x in zip(ytru, logics_all, xtru):
        true_seq = true_seq[1:]
        x = x[1:]
        true_seq = np.append(true_seq, _END) 
        x = np.append(x, _END)
        
        true_seq=true_seq[:list(true_seq).index(_END)]
        x=x[:list(x).index(_END)]
        
        xseq = " ".join([reverse_vocab_dict[idx] for idx in x ])
        true_logic = " ".join([reverse_vocab_dict[idx] for idx in true_seq ])

        logic = logic.replace(' (','').replace(' )','')
        true_logic = true_logic.replace(' (','').replace(' )','') 
        logic = _switch_cond(logic, true_logic)
        
        acc += (logic==true_logic)
        i += 1
    print('EM: %.4f'%(acc*1./len(y_data)))  
    print('number of correct ones:%d'%acc)
    
    return acc*1./len(y_data) 


def _switch_cond(logic, true_logic):
    logic_tokens = logic.split()
    if len(logic_tokens) > 8 and logic_tokens[5] == 'and':
        newlogic = [x for x in logic_tokens]
        newlogic[2], newlogic[6], newlogic[4], newlogic[8] = logic_tokens[6], logic_tokens[2], logic_tokens[8], logic_tokens[4]
        newline = ' '.join(newlogic)
        if newline == true_logic:
            logic = newline
    elif len(logic_tokens) > 9 and logic_tokens[6] == 'and':
        newlogic = [x for x in logic_tokens]
        newlogic[3], newlogic[7], newlogic[5], newlogic[9] = logic_tokens[7], logic_tokens[3], logic_tokens[9], logic_tokens[5]
        newline = ' '.join(newlogic)
        if newline == true_logic:
            logic = newline  
    return logic  
