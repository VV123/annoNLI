#!/usr/bin/env bash

#######################################
# Bash3 Boilerplate Start
# copied from https://kvz.io/blog/2013/11/21/bash-best-practices/

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename ${__file} .sh)"
__root="$(cd "$(dirname "${__dir}")" && pwd)" # <-- change this as it depends on your app

arg1="${1:-}"
# Bash3 Boilerplate End
#######################################

function parse_dev_ground_truth (){
  ./parse_ours.py \
    --data_root ../scratch \
    --table_file dev_cleaned_table.txt \
    --sql_file dev_ground_truth_mark.txt \
    --sqltableid_file dev_SQL2tableid.txt \
    --parsed_sql_file dev_ground_truth_mark.parsed.txt \
    ;
}

function parse_dev_infer (){
  ./parse_ours.py \
    --data_root ../scratch \
    --table_file dev_cleaned_table.txt \
    --sql_file dev_infer.txt \
    --sqltableid_file dev_SQL2tableid.txt \
    --parsed_sql_file dev_infer.parsed.txt \
    ;
}

function parse_test_ground_truth (){
  ./parse_ours.py \
    --data_root ../scratch \
    --table_file test_cleaned_table.txt \
    --sql_file test_ground_truth_mark.txt \
    --sqltableid_file test_SQL2tableid.txt \
    --parsed_sql_file test_ground_truth_mark.parsed.txt \
    ;
}

function parse_test_infer (){
  ./parse_ours.py \
    --data_root ../scratch \
    --table_file test_cleaned_table.txt \
    --sql_file test_infer.txt \
    --sqltableid_file test_SQL2tableid.txt \
    --parsed_sql_file test_infer.parsed.txt \
    ;
}

mkdir -p "${__dir}/../scratch/output"

# excute all functions if instructed
if [ "$1" = "all" ] ; then
  for func_name in `compgen -A function ` ; do
    echo "running \"$func_name\""
    ( cd "${__dir}/../code" && $func_name )
  done

  exit 0
fi

( cd "${__dir}/../code" && $1 )
