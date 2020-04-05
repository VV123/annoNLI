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
from wiki import load_data_wiki
from overnight import load_data_overnight
import argparse
from vocab import build_vocab_all, load_vocab_all


def load_data(args):
    maxlen = args.maxlen
    load = args.load_data
    if args.data == 'wikisql':
        X_train, y_train = load_data_wiki(args,s='train')
        print('========Train data shape=======')
        print(X_train.shape)
        print(y_train.shape) 
        X_test, y_test = load_data_wiki(args,s='test')
        print('========Test data shape=======')
        print(X_test.shape)
        print(y_test.shape)
        X_dev, y_dev = load_data_wiki(args,s='dev')
        print('========Dev data shape=======')
        print(X_dev.shape)
        print(y_dev.shape) 
        return X_train, y_train, X_test, y_test, X_dev, y_dev
    elif args.data=='overnight':
        X_all, y_all = None, None
        for subset in ['basketball','calendar','housing','recipes','restaurants']:
            X1, y1 = load_data_overnight(args, subset=subset, s='train')
            X2, y2 = load_data_overnight(args, subset=subset, s='test')
            X = np.concatenate([X1,X2], axis=0)
            if X_all is not None:
                X_all = np.concatenate([X_all, X], axis=0)
            else:
                X_all = X

            y = np.concatenate([y1,y2], axis=0)
            if y_all is not None:
                y_all = np.concatenate([y_all,y], axis=0)
            else:
                y_all = y
        X, y = X_all, y_all
        print('========overnight all shape=======')
        print(X.shape)
        print(y.shape)  
        return X, y
    elif args.data == 'overnight_set':
        lists = []
        for subset in ['basketball', 'calendar', 'housing', 'recipes', 'restaurants']:
            X1, y1 = load_data_overnight(args, subset=subset, s='train')
            X2, y2 = load_data_overnight(args, subset=subset, s='test')
            X = np.concatenate([X1,X2], axis=0)
            y = np.concatenate([y1,y2], axis=0)
            lists.append((X, y))        

        return lists


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--maxlen', default=60, type=int, help='Data record max length.')  
    arg_parser.add_argument('--data', default='overnight', type=str, help='Data.')
    arg_parser.add_argument('--embedding_dim', default=300, type=int, help='Embedding dim.')
    arg_parser.add_argument('--data_path', default=os.path.dirname(os.path.abspath(__file__)).replace('utils', 'data'), type=str, help='Data path.')
    arg_parser.add_argument('--load_data', default=False, type=bool, help='Load data.')
    args = arg_parser.parse_args()

    rebuild = True   
    if rebuild:
        build_vocab_all(args, load=False)
        load_vocab_all(args, load=False)
        load_data(args)


