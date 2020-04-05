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
import argparse

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path).replace('utils', 'data')
wiki_path = dir_path + '/DATA/wiki'
save_path = dir_path
'''
0: pad
1: bos
2: eos
'''
_PAD = 0
_GO = 1
_END = 2
_EOF = 3
_UNK = 4
_TOKEN_NUMBER = 5
_TOKEN_MODEL = 6
_EOC = 7
ori_files1 = [ 'train.lon', 'train.qu', 'test.lon', 'test.qu', 'dev.lon', 'dev.qu']
ori_files2 = [ 'train_vocab.lon', 'train_vocab.qu', 'test_vocab.lon', 'test_vocab.qu']
vocab_files = [ os.path.join(wiki_path, x) for x in ori_files1 ]
for subset in ['basketball', 'calendar', 'housing', 'recipes', 'restaurants']:
    overnight_path = dir_path + '/DATA/overnight/%s'%subset
    vocab_files.extend([ os.path.join(overnight_path, x) for x in ori_files2 ])
annotation = ['<f0>','<f1>','<f2>','<f3>','<v0>','<v1>','<v2>','<v3>']
def build_vocab_all(args, load=True, files=vocab_files):
    embedding_dim = args.embedding_dim    
    if load==False:
        vocab_tokens = ['<pad>','<bos>','<eos>','<eof>','<unk>','<@number>','<@model>','<eoc>']
        vocab_tokens.extend(annotation)
        vocabs = set()
        
        for fname in files:
            #print(fname)
            with gfile.GFile(fname, mode='r') as DATA:
                lines = DATA.readlines()
                for line in lines:
                    for word in line.split():
                        if word not in vocabs and word not in vocab_tokens:
                            vocabs.add(word)
        
        vocab_tokens.extend(list(vocabs))
        np.save(os.path.join(save_path,'vocab_tokens_all.npy'),vocab_tokens)
        #print('build vocab done.')
    else:
        vocab_tokens=np.load(os.path.join(save_path,'vocab_tokens_all.npy'))

    return vocab_tokens

def load_vocab_all(args, load=True):
    embedding_dim = args.embedding_dim
    if load == False:
        vocab_dict = {}
        reverse_vocab_dict = {}
        embedding = glove.Glove()
        vocabs = build_vocab_all(args)
        vocab_tokens = []
        for i,word in enumerate(vocabs):
            vocab_dict[word]=i
            reverse_vocab_dict[i]=word.decode('utf-8')
            vocab_tokens.append([word.decode('utf-8')])
        np.save(os.path.join(save_path,'vocab_dict_all.npy'),vocab_dict)
        np.save(os.path.join(save_path,'reverse_vocab_dict_all.npy'),reverse_vocab_dict)
        vocab_emb, unk_idx = embedding.embedding(vocab_tokens, maxlen=1)
        unk_idx = np.asarray(unk_idx)
        vocab_emb = vocab_emb[:,0] #retrieve embedding
        #print(np.max(vocab_emb))
        #print(np.min(vocab_emb))
        train_idx = unk_idx
        #train_idx = np.concatenate( (np.arange(15), unk_idx) )
        np.save(os.path.join(save_path,'train_idx.npy'),train_idx)
        #print(train_idx)
        #print(len(train_idx))
        i = 0
        emb_dict = {}
        emb_dict['f'] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
        emb_dict['v'] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
        emb_dict['c'] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
        for token,emb in zip(vocab_tokens,vocab_emb):
            token = token[0]
            if len(token)>=4 and token[0]=='<' and token[3]=='>' and token[2].isdigit():
                if token[2] in emb_dict:
                    right = emb_dict[token[2]]
                else:
                    emb_dict[token[2]] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
                    right = emb_dict[token[2]]
                re = np.concatenate((emb_dict[token[1]],right))
                assert re.shape==(300,)
                vocab_emb[i]=re
            elif len(token)>=5 and token[0]=='<' and token[4]=='>' and token[2:4].isdigit():
                if token[2:4] in emb_dict:
                    right = emb_dict[token[2:4]]
                else:
                    emb_dict[token[2:4]] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
                    right = emb_dict[token[2:4]]
                re = np.concatenate((emb_dict[token[1]],right))
                assert re.shape==(300,)
                vocab_emb[i]=re 
            i += 1

        np.save(os.path.join(save_path,'vocab_emb_all.npy'),vocab_emb)
        #print('Vocab shape:')
        #print(vocab_emb.shape)
    else:
        vocab_emb=np.load(os.path.join(save_path,'vocab_emb_all.npy'))
        vocab_dict=np.load(os.path.join(save_path,'vocab_dict_all.npy')).item()
        reverse_vocab_dict=np.load(os.path.join(save_path,'reverse_vocab_dict_all.npy')).item()
        train_idx = np.load(os.path.join(save_path,'train_idx.npy'))
        #print('Vocab shape:')
        #print(vocab_emb.shape)
    return vocab_dict, reverse_vocab_dict, vocab_emb, train_idx




