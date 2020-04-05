#!/usr/bin/env python3

from enum import Enum
import json
from os.path import join

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', '', 'root directory for data.')
flags.DEFINE_string('table_file', '', '')
flags.DEFINE_string('sql_file', '', '')
flags.DEFINE_string('sqltableid_file', '', '')
flags.DEFINE_string('parsed_sql_file', '', '')


class TokenType(Enum):
    KEYWORD = 0
    VALUE = 1
    EOF = 2


agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
agg_ops = [_.lower() for _ in agg_ops]

# cond_ops = ['=', '>', '<', 'OP']
cond_ops = ['equal', 'greater', 'less', 'OP']

agg_ops_to_index = {elem: index for index, elem in enumerate(agg_ops)}
cond_ops_to_index = {elem: index for index, elem in enumerate(cond_ops)}


def tokenize(sql):

    QUOTE = ('"', )
    SPACE = (' ', '\t')

    sql = sql.strip()
    n = len(sql)
    i = 0
    tokens = []
    while i < n:
        if sql[i] == '"':
            j = i + 1
            while j < n and sql[j] not in QUOTE:
                j += 1
            assert sql[j] in QUOTE
            tokens.append((TokenType.VALUE, sql[i + 1:j]))
            i = j + 1
        elif sql[i] in SPACE:
            while i < n and sql[i] in SPACE:
                i += 1
        else:
            j = i
            while j < n and sql[j] not in SPACE:
                j += 1
            tokens.append((TokenType.KEYWORD, sql[i:j]))
            i = j

    tokens.append((TokenType.EOF, None))
    # print(tokens)

    return tokens


def get_ctx(table):
    header_str_to_header_index = {
        header: index
        for index, header in enumerate(table['header'])
    }
    return {
        'header': table['header'],
        'types': table['types'],
        'header_str_to_header_index': header_str_to_header_index,
    }


def maybe_str_to_float(s):
    try:
        result = float(s)
    except Exception:
        result = s
    return result


def parse(sql, table):
    tokens = tokenize(sql)
    ctx = get_ctx(table)

    # -- parse loop
    conds, sel, agg = [], None, None  # results

    n = len(tokens)
    i = 0

    assert tokens[i] == (TokenType.KEYWORD, 'select')
    i += 1

    if tokens[i][0] == TokenType.KEYWORD:
        agg = agg_ops_to_index[tokens[i][1]]
        assert agg != 0
        i += 1

        assert tokens[i][0] == TokenType.VALUE
        sel = ctx['header_str_to_header_index'][tokens[i][1]]
        i += 1
    else:
        agg = 0

        assert tokens[i][0] == TokenType.VALUE
        sel = ctx['header_str_to_header_index'][tokens[i][1]]
        i += 1

    if tokens[i][0] != TokenType.EOF:
        assert tokens[i] == (TokenType.KEYWORD, 'where')
        i += 1

        i_after_where = i

        # -- normal loop
        is_good = True
        while True:
            if tokens[i][0] != TokenType.EOF:
                assert i + 3 < n

                if tokens[i + 1][1] in cond_ops_to_index:
                    operator_index = cond_ops_to_index[tokens[i + 1][1]]
                else:
                    is_good = False
                    break

                if (tokens[i + 0][1] in ctx['header_str_to_header_index']):
                    column_index = ctx['header_str_to_header_index'][tokens[
                        i + 0][1]]
                    condition = maybe_str_to_float(tokens[i + 2][1])
                else:
                    is_good = False
                    break

                conds.append([column_index, operator_index, condition])

                i += 3

                if tokens[i][0] != TokenType.EOF:
                    if tokens[i] == (TokenType.KEYWORD, 'and'):
                        pass
                    else:
                        is_good = False
                        break
                    i += 1
            else:
                break

        # -- special fallback
        if not is_good:
            index_cond_op = [
                index for index in range(i_after_where, n)
                if tokens[index][1] in cond_ops
            ]
            assert len(index_cond_op) == 1
            index_cond_op = index_cond_op[0]

            column_index = ctx['header_str_to_header_index'][' '.join(
                [_[1] for _ in tokens[i_after_where:index_cond_op]])]

            operator_index = cond_ops_to_index[tokens[index_cond_op][1]]
            condition = maybe_str_to_float(' '.join(
                [_[1] for _ in tokens[index_cond_op + 1:n - 1]]))

            conds.append([column_index, operator_index, condition])

    assert agg is not None
    assert sel is not None
    return {'conds': conds, 'sel': sel, 'agg': agg}


def main(argv):
    del argv  # Unused.

    # -- Load table
    logging.info('Load table')
    table_list = []
    for raw_line in tqdm(open(join(FLAGS.data_root, FLAGS.table_file))):
        line = raw_line.strip()
        table_list.append(json.loads(line))
    table_id_to_table_index = {
        table['id']: index
        for index, table in enumerate(table_list)
    }

    # -- Load query id
    query_index_to_table_id = []
    for raw_line in open(join(FLAGS.data_root, FLAGS.sqltableid_file)):
        line = raw_line.strip()
        query_index_to_table_id.append(line)
        assert line in table_id_to_table_index

    # -- Load SQL file
    logging.info('Load SQL')
    sql_list = []
    for raw_line in tqdm(open(join(FLAGS.data_root, FLAGS.sql_file))):
        line = raw_line.strip()
        sql_list.append("select " + line)
    assert len(sql_list) == len(query_index_to_table_id)

    # -- parsing
    logging.info('Parsing')
    count = {'good': 0, 'bad': 0}
    parsed_sql_list = []
    for query_index, sql in tqdm(enumerate(sql_list), desc='parsing'):

        table = table_list[table_id_to_table_index[query_index_to_table_id[
            query_index]]]
        try:
            result = {
                'sql': parse(sql, table),
                'table_id': query_index_to_table_id[query_index]
            }
            parsed_sql_list.append(result)
            count['good'] += 1
        except Exception:
            parsed_sql_list.append([])
            count['bad'] += 1

    print(count['good'], count['bad'])
    if False and count['bad'] > 0:
        print('bad guys')
        for query_index, parsed_sql in enumerate(parsed_sql_list):
            if parsed_sql == []:
                print(query_index + 1, '  ---  ', sql_list[query_index])

    # -- writing
    with open(join(FLAGS.data_root, FLAGS.parsed_sql_file), 'w') as fout:
        for parsed_sql in parsed_sql_list:
            print(json.dumps(parsed_sql), file=fout)


import pdb, traceback, sys, code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
