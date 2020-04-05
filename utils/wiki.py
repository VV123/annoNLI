# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import glove
from collections import defaultdict
import vocab
from vocab import load_vocab_all

'''
0: pad
1: bos
2: eos
'''
_PAD = vocab._PAD
_GO = vocab._GO
_END = vocab._END
_EOF = vocab._EOF
_UNK = vocab._UNK
_TOKEN_NUMBER = vocab._TOKEN_NUMBER
_TOKEN_MODEL = vocab._TOKEN_MODEL
_EOC = vocab._EOC


def load_data_wiki(args, s='train'):
    dir_path = args.data_path
    wiki_path = dir_path + '/DATA/wiki'
    save_path = dir_path
    maxlen = args.maxlen
    load = args.load_data
    if load:
        emb = np.load(os.path.join(save_path,'vocab_emb_all.npy'))
        #print('========embedding shape========')
        #print(emb.shape)
        all_q_tokens = np.load(os.path.join(save_path,'wiki/%s_qu_idx.npy'%(s)))
        all_logic_ids = np.load(os.path.join(save_path,'wiki/%s_lon_idx.npy'%(s)))
        #print(all_q_tokens.shape)
        #print(all_logic_ids.shape)
    else:
        all_q_tokens = []
        all_logic_ids = []
        vocab_dict,_,_,_=load_vocab_all(args)
        vocab_dict = defaultdict(lambda:_UNK, vocab_dict)
        questionFile=os.path.join(wiki_path,'%s.qu'%(s))
        logicFile=os.path.join(wiki_path,'%s.lon'%(s))
        with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
            q_sentences = questions.readlines()
            logics = logics.readlines()
            assert len(q_sentences)==len(logics)
            i = 0
            length = len(logics)
            for q_sentence,logic in zip(q_sentences,logics):
                i+=1
                #print('counting: %d / %d'%(i,length),end='\r')
                sys.stdout.flush()
                token_ids = [_GO]
                token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
                for x in q_sentence.split():
                    if vocab_dict[x]==_UNK:
                        print('ERROR unknow word in question:'+x)
                #token_ids.append(_END)
                logic_ids = [_GO]
                logic_ids.extend([vocab_dict[x] for x in logic.split()])
                for x in logic.split():
                    if vocab_dict[x]==_UNK:
                        print('ERROR unknow word in logic:'+x)
                logic_ids.append(_END)
                if maxlen>len(logic_ids):
                    logic_ids.extend([ _PAD for i in range(len(logic_ids),maxlen)])
                else:
                    logic_ids = logic_ids[:maxlen]
                if maxlen>len(token_ids):
                    token_ids.extend([ _PAD for i in range(len(token_ids),maxlen)])
                else:
                    token_ids = token_ids[:maxlen]
                all_q_tokens.append(token_ids)
                all_logic_ids.append(logic_ids)
            all_logic_ids=np.asarray(all_logic_ids)
            #print('------wiki '+s+' shape------')
            #print(all_logic_ids.shape)
            all_q_tokens=np.asarray(all_q_tokens)
            np.save(os.path.join(save_path,'wiki/%s_lon_idx.npy'%s),all_logic_ids)
            np.save(os.path.join(save_path,'wiki/%s_qu_idx.npy'%s),all_q_tokens)
	
    return all_q_tokens,all_logic_ids

