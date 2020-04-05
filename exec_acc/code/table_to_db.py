#!/usr/bin/env python3

import json
from os.path import join
from os import system

from absl import app
from absl import flags
from absl import logging
import records
from tqdm import tqdm

from lib import table as table_module

FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', '', 'root directory for data.')
flags.DEFINE_string('table_file', '', '')
flags.DEFINE_string('db_file', '', '')


def main(argv):
    del argv  # Unused.

    # -- Load table
    logging.info('Load table')
    table_list = []
    for raw_line in tqdm(open(join(FLAGS.data_root, FLAGS.table_file))):
        line = raw_line.strip()
        table_list.append(json.loads(line))

    # -- initialize db file
    logging.info('Initialize db file')
    db_filepath = join(FLAGS.data_root, FLAGS.db_file)
    system('rm -rf %s' % db_filepath)
    db = records.Database('sqlite:///%s' % db_filepath)

    # -- insert table to db file
    logging.info('Insert table to db file')
    for table in tqdm(table_list):
        t = table_module.Table(
            table_id=table['id'],
            header=table['header'],
            types=table['types'],
            rows=table['rows'],
        )
        t.create_table(db)


import pdb, traceback, sys, code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
