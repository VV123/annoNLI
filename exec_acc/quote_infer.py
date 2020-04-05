# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import re
import collections
from collections import defaultdict
import copy
import editdistance as ed
from scipy import spatial
from lib.common import count_lines

#insert quote to inferred queries to facilitate parsing process

path = os.path.abspath(__file__)
data_path = os.path.dirname(path).replace('/exec_acc','') + '/data/DATA/wiki/'
#data_path = '/home/wzw0022/DATA/wiki'
save_path = os.path.dirname(path) + '/scratch/'

symbols = ['select','where','and','equal','greater','less','max','min','count','sum','avg']
def _preclean(Q):
    """
    Clean before annotation.
    """
    Q = re.sub('#([0-9])',r'# \1', Q)
    Q = Q.replace('€',' €')
    Q = Q.replace('\'',' ')
    Q = Q.replace(',','')
    Q = Q.replace('?','')
    Q = Q.replace('\"','')
    Q = Q.replace('(s)','')
    Q = Q.replace('  ',' ')
    return Q.lower()


def main():
    for split in ['test', 'dev']:

        fsplit = os.path.join(data_path, split) + '.jsonl'
        ftable = os.path.join(data_path, split) + '.tables.jsonl'
        with open(fsplit) as fs, open(ftable) as ft :
            print('loading tables')
            tables = {}

            for line in tqdm(ft, total=count_lines(ftable)):
                d = json.loads(line)
                tables[d['id']] = d
            print('loading tables done.')

        with open(data_path+'%s_infer.txt'%split, 'r') as in_file, \
            open(save_path+'%s_infer.txt'%split, 'w') as out_file,\
            open(save_path+'%s_ground_truth.txt'%split,'r') as true0,\
            open(save_path+'%s_ground_truth_mark.txt'%split,'r') as true,\
            open(save_path+'%s_SQL2tableid.txt'%split,'r') as id_file,\
            open(save_path+'%s_sym_pairs.txt'%split,'r') as pairs_file:

            lines = in_file.readlines()
            gt_lines = true0.readlines()
            gt_mark_lines = true.readlines()
            ids = id_file.readlines()
            pairs = pairs_file.readlines()

            idx = 1
            for line, gt, gt_mark, t_id, pair in zip(lines,gt_lines,gt_mark_lines,ids,pairs):
        
                t_id = t_id.replace('\n','')
                

                new = ''
            
                inf = line
                    
                rows = tables[t_id]['rows']
                rows = np.asarray(rows)
                fs = tables[t_id]['header']

                all_f = []
                for f in fs:
                    f = _preclean(str(f))
                    all_f.append(f)

                for row in rows:
                    for i in range(len(fs)):
                        f =  _preclean(str(row[i])) 
                        all_f.append(f)

                #all fields (including values) are sorted in dscending order
                all_f.sort(key=len, reverse=True)

                added = []  # collect phrases with SYMBOL word (e.g.,'MAX') in it

                for f in all_f:
                    PASS = False
                    ADD = False #SYMBOL in it or not a valid sub-phrase

                    for a in added:
                        if f in a:
                            PASS = True
                            break

                    if len(f.split())==1 and f not in line.split():
                        PASS = True

                    # whether SYMBOL is in it
                    for s in symbols:
                        if s in f.split():
                            ADD = True
                            break

                    if f in line and PASS == False and ADD and f != 'where':
                        added.append(f)


                # added phrases are replaced away           
                for iden, phrase in enumerate(added):
                    line = line.replace(phrase,'<p'+str(iden)+'>')

                tokens = line.split()
                for i in range(len(tokens)):
                    token = tokens[i]


                    if i == 0:
                        if token not in symbols:
                            new += '\"'

                    elif tokens[i-1] in symbols and token not in symbols:
                        new += '\"'
            

                    new += token

                    if i == len(tokens)-1: 
                        if token not in symbols:
                            new += '\"'
                    elif tokens[i+1] in symbols and token not in symbols:
                        new += '\"'

                    new += ' '

                
                new = new.strip()

                # replace back
                for iden, phrase in enumerate(added):
                    new = new.replace('<p'+str(iden)+'>', phrase)

                gt_mark = gt_mark.replace('\n','')
                if gt == inf and gt_mark != (new):
                
                    print(added)
                    print('inf:'+line)
                    print('gt:'+gt)
                    print('gt_mark:'+gt_mark+'.')
                    print('inf_mark:'+new+'.')
            

                out_file.write(new+'\n')
                idx += 1


if __name__ == '__main__':
    main()  
