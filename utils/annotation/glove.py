from __future__ import print_function
from __future__ import division
"""
Preprocess GloVe embedding data.
"""
import os
import json
import numpy as np

from sklearn.externals import joblib
from multiprocessing import current_process
from scipy import spatial
GLOVE_PATH = os.environ['GLOVE_PATH']
class Glove:
    """
    Wrapper for GloVe embedding.
    """
    num_words = 2196018
    embedding_dim = 300
    batch_size = 5
    process_num= 2 
    def __init__(self, glove=GLOVE_PATH, rawfile='glove.840B.300d.txt',
                 rebuild=False):
        row = self.num_words
        dim = self.embedding_dim

        glove = os.path.expanduser(glove)
        rawfile = os.path.join(glove, rawfile)

        if rebuild:
            print('\nReading {}'.format(rawfile))

            with open(rawfile, 'r') as f:
                id2word = [''] * (row + 2)
                word2id = {}
                id2vec = np.empty((row + 2, dim), dtype=np.float32)
                for i, line in enumerate(f):
                    print('{0:8d}/{1}'.format(i+1, row), end='\r')
                    kv = line.split(' ', dim)
                    k = kv[0]
                    v = np.array(kv[1:]).astype(np.float32)
                    id2word[i] = k
                    word2id[k] = i
                    id2vec[i] = v

                id2word[-2] = '<bos>'
                word2id['<bos>'] = row
                id2vec[row] = np.ones(dim)

                id2word[-1] = '<pad>'
                word2id['<pad>'] = row + 1
                id2vec[row + 1] = np.zeros(dim)

            print('\nSaving id2word')
            with open(os.path.join(glove, 'id2word.txt'), 'w') as f:
                f.write('\n'.join(id2word))

            print('\nSaving word2id')
            with open(os.path.join(glove, 'word2id.txt'), 'w') as f:
                f.write(json.dumps(word2id))

            print('\nSaving id2vec')
            np.save(os.path.join(glove, 'id2vec.npy'), id2vec)
        else:
            print('\nLoading id2word')
            with open(os.path.join(glove, 'id2word.txt'), 'r') as f:
                id2word = [line.strip() for line in f]

            print('\nLoading word2id')
            with open(os.path.join(glove, 'word2id.txt'), 'r') as f:
                word2id = json.loads(f.read())

            print('\nLoading id2vec')
            id2vec = np.load(os.path.join(glove, 'id2vec.npy'))

        self._id2word = np.array(id2word, dtype=str)
        self._word2id = word2id
        self._id2vec = id2vec

    def embedding(self, texts, maxlen=0):
        if 0 == maxlen:
            maxlen = len(max(texts, key=len))

        word2id, id2vec = self._word2id, self._id2vec
        dim = id2vec.shape[1]

        print('\nAllocating embedding')
        vec = np.tile(id2vec[word2id['<pad>']], (len(texts) * (maxlen+1), 1))
        vec = np.reshape(vec, (len(texts), maxlen+1, dim))
        vec = vec.astype(np.float32)

        print('\nDo embedding and return unk idx...')
        unk_idx = []
        for i, text in enumerate(texts):
            for j, word in enumerate(text[:(maxlen+1)]):
                assert j < 1
                if word not in word2id or i < 15:
                    #word = 'unk'
                    vec[i,j] = np.random.rand(300)*np.square(3)
                    unk_idx.append(i)
                else:
                    vec[i, j] = id2vec[word2id[word]]
        return vec, unk_idx

    def embed_one(self, word):
        
        word2id, id2vec = self._word2id, self._id2vec
     

        if word in word2id:
            return id2vec[word2id[word]]
        else:
            return np.ones(300)

def _worker(vecs, k, modelpath, num, return_dict):
    cur = current_process()
    print('\n{} loading knn'.format(cur.name))
    #if len(vecs.shape) < 2:
    #    vecs = np.expand_dims(vecs, axis=0)
    knn = joblib.load(modelpath)
    print('\n{} searching...'.format(cur.name))
    res = knn.kneighbors(vecs, n_neighbors=k, return_distance=False) 
    return_dict[num] = res
    return res



