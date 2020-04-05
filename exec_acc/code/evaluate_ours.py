#!/usr/bin/env python3

from enum import Enum
import json
from os.path import join

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

from lib.dbengine import DBEngine
from lib.query import Query
from lib.common import count_lines

FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', '', 'root directory for data.')
flags.DEFINE_string('parsed_std_sql_file', '', '')
flags.DEFINE_string('parsed_pred_sql_file', '', '')
flags.DEFINE_string('db_file', '', '')


def main(argv):
    del argv  # Unused.

    db_file = join(FLAGS.data_root, FLAGS.db_file)
    parsed_std_sql_file = join(FLAGS.data_root, FLAGS.parsed_std_sql_file)
    parsed_pred_sql_file = join(FLAGS.data_root, FLAGS.parsed_pred_sql_file)

    engine = DBEngine(db_file)
    exact_match = []

    with open(parsed_std_sql_file) as fs, open(parsed_pred_sql_file) as fp:
        grades = []
        for ls, lp in tqdm(
                zip(fs, fp), total=count_lines(parsed_std_sql_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            
            try:
                qg = Query.from_dict(eg['sql'])
                gold = engine.execute_query(eg['table_id'], qg, lower=True)
            except Exception as e:
                gold = repr(e)
            
            #pred = ep['error']
            qp = None
            #if not ep['error']:
            if True:
                try:
                    qp = Query.from_dict(ep['sql'])
                    pred = engine.execute_query(eg['table_id'], qp, lower=True)
                except Exception as e:
                    pred = repr(e)
            correct = pred == gold
            match = qp == qg
            if pred == gold and qp != qg:
                print(qp)
                print(qg)
            grades.append(correct)
            exact_match.append(match)
        print(
            json.dumps(
                {
                    'ex_accuracy': sum(grades) / len(grades),
                    'lf_accuracy': sum(exact_match) / len(exact_match),
                },
                indent=2))


import pdb, traceback, sys, code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
