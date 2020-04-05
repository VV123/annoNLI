# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
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

import utils
from utils import _preclean, _clean, _equal, _strip_stopword, _check_head, _max_span, _abbr_match, _match_ids, _threshold, _value_match, _digit, _backslash

path = os.path.abspath(__file__)
save_path = os.path.dirname(path).replace('utils/annotation',
                                          'data/DATA/wiki/')
data_path = os.environ['WIKI_PATH']

# set wiki_path to WikiSQL raw data directory
wiki_path = os.environ['WIKI_PATH']
#----------------------------------------------------------
maps = defaultdict(list)
stop_words = ['a', 'of', 'the', 'in']
UNK = 'unk'

def _approx_match_map(items, Q):
    items = [ item for item in items if item not in Q ]
    smap, _ = _approx_match_map_helper(items, Q)
    return smap


def _approx_match_map_helper(items, Q):
    """
    retrieve approximate match map
    """
    smap = defaultdict(lambda: 'unk', {})
    item2len = {} #match len
    
    for item in items:
        ids = _match_ids(item, Q)
        replacement, match_len = _max_span(
                ids, Q.split())
        if match_len >= _threshold(item):
            smap[item] = replacement
            item2len[item] = match_len

    return smap, item2len


def _approx_match(Q, Q_ori, fs, v):
    """
    approximate match fields based on value v
    """

    field2partial  = _approx_match_map(fs, Q)
    fs = list(field2partial.keys())
    fs.sort(key=lambda x: abs(Q_ori.index(field2partial[x])-Q_ori.index(v)))

    return 'unk' if len(fs) == 0 else fs[0]


def _match_pairs (Q, Q_ori, keys, reverse_map):
    """
    Iterate all values and
    match column f for each value
    """
    candidates = []
    cond_fields = []
    # match values first
    for v in keys:
        l = len(v.split())
 
        fs = reverse_map[v]

        v = _digit(v, Q)

        if (l > 1 and v in Q) or (l == 1 and str(v) in Q.split()):
            
            # PASS if current v is a substring of matched v
            if not any(value for _, value in candidates if _value_match(value, v)):
                fs = list(fs)
                # only one possible field, identify (f,v) directly
                if len(fs) == 1:
                    f = fs[0]
                    cond_fields.append(f)
                else:  
                    fs_inQ = [ item for item in fs if item in Q ]
                    if len(fs_inQ) >= 1: 
                        fs_inQ.sort(key=lambda x: abs(Q_ori.index(x) - Q_ori.index(v)))
                        f = fs_inQ[0]
                    else: #approximate match
                        f = _approx_match(Q, Q_ori, fs, v)

                    cond_fields.append(f)

                cond_fields.remove(f) if f == v else candidates.append([_preclean(f), _preclean(v)])

    # sort to compare with ground truth
    candidates.sort(key=lambda x: x[1])
    return candidates, cond_fields


def _match_head_variant(Q, Q_ori, all_fields, cond_fields):
    
    exclude_fields = [f for f in all_fields if f not in cond_fields]
    head2partial, head2matchlen = _approx_match_map_helper(exclude_fields, Q)

    head_cands = list(head2partial.keys())
    head_cands.sort( key=lambda x: head2matchlen[x] )              
    head_cands = [ f for f in head_cands if head2matchlen[f]==head2matchlen[head_cands[-1]] ]
    head_cands.sort( key=lambda x: Q_ori.index(head2partial[x]) )

    head = '' if len(head_cands) == 0 else head_cands[0]

    return head, head2partial

def _match_head(Q, Q_ori, smap, all_fields, cond_fields):
    head_cands = []
    heads = smap.keys()
    heads = sorted(heads, key=len, reverse=True)
    heads = [f for f in heads if f not in cond_fields]
    head2partial = defaultdict(lambda: 'unk', {})
    for f in heads:
        if not any(h for h in head_cands if f in h) and f in Q_ori and _check_head(
                f, Q_ori):
            head_cands.append(f)

    if len(head_cands) >= 1:
        head_cands.sort(key=lambda x: Q_ori.index(x))
        head = head_cands[0]
    elif len(heads) == 1:
        head = heads[0]
    else:
        head, head2partial = _match_head_variant(Q, Q_ori, all_fields, cond_fields)

    return head, head2partial

def annotate_Q(Q, Q_ori, Q_head, candidates, all_fields, head2partial, n, target, PARTIAL_MATCH=True, step2=True, ADD_FIELDS=True):
    """
    annotate Q
    """

    new_candidates = [ ['<f' + str(i+1) + '>', '<v' + str(i+1) + '>'] for i, (field, value) in enumerate(candidates) ]
       
      

    # annotate Q
    Qpairs = []
    for (f, v), (new_f, new_v) in zip(candidates, new_candidates):
        Qpairs.append((f, new_f, 'f'))
        Qpairs.append((v, new_v, 'v'))

    # sort (word,symbol) pairs by word length in descending order
    Qpairs.sort(key=lambda x: 100 - len(x[0]))


    p2partial = defaultdict(lambda: 'unk', {})

    if PARTIAL_MATCH:
        fields = [f for f, _ in candidates]
        p2partial = _approx_match_map(fields, Q)

    Q = _insert_inferred_content(Q, candidates, p2partial, step2)

    Q = _annotate_pairs(Q, Qpairs, p2partial)
    
    Q = _annotate_head(Q, Q_ori, Q_head, head2partial) 

    if ADD_FIELDS:
        Q += ' <eos> '
        for i, f in enumerate(all_fields):
            Q += (' <c' + str(i) + '> ' + f + ' <eoc> ')

    Q = _clean(Q)
    return Q, Qpairs


def _annotate_pairs(Q, Qpairs, p2partial):
    for p, new_p, t in Qpairs:
        cp = _backslash(p)

        if t == 'f':
            p = p2partial[p] if p not in Q else p

            Q0 = re.sub('(\s|^)' + cp + '(\s|$|s|\?|,|.)',
                        ' ' + new_p + ' ' + p + ' <eof> ', Q)
            Q = Q.replace(p, new_p + ' ' + p +
                          ' <eof> ') if Q == Q0 else Q0
        else:
            Q0 = re.sub('(\s|^)' + cp + '(\s|$|s|\?|,|.)',
                        ' ' + new_p + ' ', Q)
            Q = Q.replace(p, new_p + ' ') if Q == Q0 else Q0

    return Q

def _insert_inferred_content(Q, candidates, p2partial, step2 = True):

    # field inference
    if step2:
        for f, v in candidates:
            if f not in Q and p2partial[f] is not UNK:
                Q = Q.replace(v, f + ' ' + v)

    # field covered by value f = street, v = ryan street
    for f, v in candidates:
        if ((f in v and Q.count(f) == 1) or
            (p2partial[f] in v and Q.count(
                p2partial[f]) == 1)) and Q.count(v) == 1:
            Q = Q.replace(v, f + ' ' + v)

    return Q

def _annotate_head(Q, Q_ori, Q_head,head2partial):
    # MATCH HEAD
    f0 = Q_head
    if f0 is not '':
        if f0 in Q:
            Q = Q.replace(f0, '<f0> ' + f0 + ' <eof>')
        else:  
            f0 = head2partial[f0]
            f0 = _strip_stopword(f0)
            Q = Q.replace(f0, '<f0> ' + f0 + ' <eof>')

    return Q


def main():
    parser = ArgumentParser()
    parser.add_argument('--din', default=data_path, help='data directory')
    parser.add_argument('--dout', default='annotated', help='output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    for split in ['dev']:
    #for split in ['train', 'test', 'dev']:
        with open(save_path+'%s.qu'%split, 'w') as qu_file, open(save_path+'%s.lon'%split, 'w') as lon_file, \
            open(save_path+'%s.out'%split, 'w') as out, open(save_path+'%s_sym_pairs.txt'%split, 'w') as sym_file, \
            open(save_path+'%s_ground_truth.txt'%split, 'w') as S_file:

            fsplit = os.path.join(args.din, split) + '.jsonl'
            ftable = os.path.join(args.din, split) + '.tables.jsonl'

            with open(fsplit) as fs, open(ftable) as ft:
                print('loading tables')
                tables = {}
                for line in tqdm(ft, total=count_lines(ftable)):
                    d = json.loads(line)
                    tables[d['id']] = d
                print('loading tables done.')

                print('loading examples')
                n, acc, acc_pair, acc_all, error = 0, 0, 0, 0, 0
                target = -1

                ADD_FIELDS = False
                step2 = True

                for line in tqdm(fs, total=count_lines(fsplit)):
                    ADD_TO_FILE = True
                    d = json.loads(line)
                    Q = d['question']
                    
                    rows = tables[d['table_id']]['rows']
                    rows = np.asarray(rows)
                    fs = tables[d['table_id']]['header']

                    all_fields = []
                    for f in fs:
                        all_fields.append(_preclean(f))
                    # all fields are sorted by length in descending order
                    # for string match purpose
                    all_fields = sorted(all_fields, key=len, reverse=True)

                
                    smap = defaultdict(list)  #f2v
                    reverse_map = defaultdict(list)  #v2f
                    for row in rows:
                        for i in range(len(fs)):
                            cur_f = _preclean(str(fs[i]))
                            cur_row = _preclean(str(row[i]))
                            smap[cur_f].append(cur_row)
                            if cur_f not in reverse_map[cur_row]:
                                reverse_map[cur_row].append(cur_f)

                    #----------------------------------------------------------
                    # all values are sorted by length in descending order
                    # for string match purpose
                    keys = sorted(reverse_map.keys(), key=len, reverse=True)
                
                    Q = _preclean(Q)
                    Q_ori = Q  

                    #####################################
                    ########## Annotate question ########
                    #####################################
                    candidates, cond_fields = _match_pairs(Q, Q_ori, keys, reverse_map)
                    Q_head, head2partial = _match_head(Q, Q_ori, smap, all_fields, cond_fields)

                    
                    Q, Qpairs = annotate_Q(Q, Q_ori, Q_head, candidates, all_fields, head2partial, n, target)
                    qu_file.write(Q + '\n')

                    validation_pairs = copy.copy(Qpairs)
                    validation_pairs.append((Q_head, '<f0>', 'head'))
                    for i, f in enumerate(all_fields):
                        validation_pairs.append((f, '<c' + str(i) + '>', 'c'))


                    #####################################
                    ########## Annotate SQL #############
                    #####################################
                    q_sent = Query.from_dict(d['sql'])
                    S, col_names, val_names = q_sent.to_sentence(
                        tables[d['table_id']]['header'], rows,
                        tables[d['table_id']]['types'])
                    S = _preclean(S)
                    S_ori = S

                    S_noparen = q_sent.to_sentence_noparenthesis(
                        tables[d['table_id']]['header'], rows,
                        tables[d['table_id']]['types'])
                    S_noparen = _preclean(S_noparen)

                    col_names = [ _preclean(col_name) for col_name in col_names ]
                    val_names = [ _preclean(val_name) for val_name in val_names ]


                    HEAD = col_names[-1]
                    S_head = _preclean(HEAD)


                    #annotate for SQL
                    name_pairs = []
                    for col_name, val_name in zip(col_names, val_names):
                        if col_name == val_name:
                            name_pairs.append([_preclean(col_name), 'true'])
                        else:
                            name_pairs.append(
                                [_preclean(col_name),
                                 _preclean(val_name)])

                    # sort to compare with candidates
                    name_pairs.sort(key=lambda x: x[1])
                    new_name_pairs = [  ['<f' + str(i+1) + '>', '<v' + str(i+1) + '>'] for i, (field, value) in enumerate(name_pairs)]
    

                    # only annotate S while identified (f,v) pairs are right
                    if _equal(name_pairs, candidates):
                        pairs = []
                        for (f, v), (new_f, new_v) in zip(
                                name_pairs, new_name_pairs):
                            pairs.append((f, new_f, 'f'))
                            pairs.append((v, new_v, 'v'))
                        # sort (word,symbol) pairs by length in descending order
                        pairs.sort(key=lambda x: 100 - len(x[0]))

                        for p, new_p, t in pairs:
                            cp = _backslash(p)

                            if new_p in Q:
                                if t == 'v':
                                    S = S.replace(p + ' )', new_p + ' )')
                                if t == 'f':
                                    S = re.sub(
                                        '\( ' + cp + ' (equal|less|greater)',
                                        '( ' + new_p + r' \1', S)

                    # only annotate S while identified HEAD is right
                    if S_head == Q_head and '<f0>' in Q:
                        S = S.replace(S_head, '<f0>')

                    # annote unseen fields
                    if ADD_FIELDS:
                        for i, f in enumerate(all_fields):
                            cf = _backslash(f)
                            S = re.sub('(\s|^)' + cf + '(\s|$|s)',
                                       ' <c' + str(i) + '> ', S)

                    S = _clean(S)
                    lon_file.write(S + '\n')

                    ###############################
                    ######### VALIDATION ##########
                    ###############################
                    recover_S = S
                    for word, sym, t in validation_pairs:
                        recover_S = recover_S.replace(sym, word)
                        sym_file.write(sym + '=>' + word + '<>')
                    sym_file.write('\n')

                    S_file.write(S_noparen + '\n')
                    #------------------------------------------------------------------------
                    if _equal(name_pairs, candidates):
                        acc_pair += 1

                    if Q_head == S_head:
                        acc += 1

                    if _equal(name_pairs,
                                    candidates) and Q_head == S_head:
                        acc_all += 1

                    full_anno = True
                    for s in S.split():
                        if s[0] != '<' and s not in [
                                '(', ')', 'where', 'less', 'greater', 'equal',
                                'max', 'min', 'count', 'sum', 'avg', 'and',
                                'true'
                        ]:
                            error += 1
                            full_anno = False
                            break

                    if False and not (_equal(name_pairs, candidates)
                                      and head == col_names[-1]):
                        print('--------' + str(n) + '-----------')
                        print(Q_ori)
                        print(Q)
                        print(S_ori)
                        print(S)
                        print('head:')
                        print(head)
                        print(head2partial)
                        print('true HEAD')
                        print(Q_head)
                        print('fields:')
                        print(candidates)
                        print(p2partial)
                        print('true fields')
                        print(name_pairs)

                    n += 1

                print('total number of examples:' + str(n))
                print('fully snnotated:' + str(1 - error * 1.0 / n))
                print('accurate all percent:' + str(acc_all * 1.0 / n))
                print('accurate HEAD match percent:' + str(acc * 1.0 / n))
                print('accurate fields pair match percent:' +
                      str(acc_pair * 1.0 / n))


if __name__ == '__main__':
    main()
