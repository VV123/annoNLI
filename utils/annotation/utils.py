# -*- coding: utf-8 -*-
#!/usr/bin/env python
import json
from argparse import ArgumentParser
from tqdm import tqdm
from lib.query import Query
from lib.common import count_lines
import numpy as np
import re
import collections
from collections import defaultdict
import copy
import editdistance as ed
from scipy import spatial
from glove import Glove
#----------------------------------------------------------
maps = defaultdict(list)
stop_words = ['a', 'of', 'the', 'in']
glove = Glove()

def _backslash(p):
    return p.replace('\\', '\\\\').replace('(', r'\(').replace(')', r'\)').\
            replace('+', r'\+').replace('-', r'\-').\
            replace('*', r'\*').replace('?', r'\?')


def _preclean(Q):
    """
    Clean before annotation.
    """
    Q = re.sub('#([0-9])', r'# \1', Q)
    Q = Q.replace('€', ' €').replace('\'', ' ').replace(',', '').replace('?', '').replace('\"', '').replace('(s)', '').replace('  ', ' ').replace(u'\xa0', u' ')
    return Q.lower()


def _clean(Q):
    """
    Clean after annotation.
    """
    Q = Q.replace('<eof>s', '<eof>').replace('<eof>d', '<eof>')
    Q = re.sub(r'(<f[0-9]>)(s)(\s|$)', r'\1\3', Q)
    return Q

def _equal(name_pairs, candidates):
    """
    name_pairs : ground truth
    candidates : identified (f,v) pairs
    Check whether identified (f,v) pairs are the same as ground truth.
    """
    if len(name_pairs) != len(candidates):
        return False
    return name_pairs == candidates


def _digit(v, Q):
    if v.isdigit() and v not in Q.split():
            v = str(v) + '.0'

    return v


def _value_match(value, v):
    return v + ' ' in value or ' ' + v in value or (v in value and abs(len(v) - len(value)) < 2)


def _strip_stopword(f0):
    tokens = f0.split()
    while tokens[0] in stop_words:
        tokens = tokens[1:]
    while tokens[-1] in stop_words:
        tokens = tokens[:-1]
    return ' '.join(tokens)

def _check_head(head, Q):
    """
    Head should be close to question word.
    """
    question_words = [
        'what', 'what\'s', 'whats', 'whatis', 'how', 'which', 'list', 'who',
        'who\'s', 'whos', 'give', 'tell', 'name', 'where'
    ]
    idx = Q.index(head)
    tokens = Q[:idx].split()
    for i in range(1, 8):
        if len(tokens) - i >= 0 and tokens[-i] in question_words:
            return True
    return False

def _max_span(ids, tokens):
    """
    Search for maximum continuous span.
    """
    if len(ids) < 1:
        return '', -1

    intervals = []
    for i in ids:
        if not intervals:
            intervals.append([i])
        else:
            ADD = False
            for interval in intervals:
                if i == interval[0] - 1:
                    interval.insert(0, i)
                    ADD = True
                elif i == interval[-1] + 1:
                    interval.append(i)
                    ADD = True
            if not ADD:
                intervals.append([i])

    #intervals.sort(key=len, reverse=True)

    max_len = -1
    max_interval = []

    for interval in intervals:
        l = 0
        for idx in interval:
            if tokens[idx] not in stop_words:
                l += 1
        if l > max_len:
            max_len = l
            max_interval = interval

    re = ''
    for i in max_interval[:-1]:
        re += tokens[i]
        re += ' '
    re += tokens[max_interval[-1]]

    return re, max_len


def _abbr_match(a, b):
    """
    Match abbreviation.
    """
    if a[-1] != '.' and b[-1] != '.':
        return False

    if a[-1] == '.':
        idx = a.index('.')
        if len(b) > idx and a[:idx] == b[:idx]:
            return True
    else:
        idx = b.index('.')
        if len(a) > idx and a[:idx] == b[:idx]:
            return True
    return False


def _match_ids(field, Q):
    """
    Imperfect match
    """
    ids = []
    length = 0
    for token in field.replace('/', ' ').split():
        if token == 'no.':
            token = 'number'

        for i, q in enumerate(Q.split()):
            semantic_sim = 1 - spatial.distance.cosine(
                glove.embed_one(q), glove.embed_one(token))
            if q in stop_words or _abbr_match(q, token) or ed.eval(
                    q, token) / len(token) < 0.5 or semantic_sim >= 0.7:
                ids.append(i)
                if q not in stop_words:
                    length += 1
    return ids

def _threshold(f):
    """
    Overlap threshold.
    """
    tokens = []
    for t in f.split():
        if t not in stop_words:
            tokens.append(t)
    return len(tokens) / 2


