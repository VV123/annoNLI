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

function evaluate_dev (){
  ./evaluate_ours.py \
    --data_root ../scratch \
    --parsed_std_sql_file dev_ground_truth_mark.parsed.txt \
    --parsed_pred_sql_file dev_infer.parsed.txt \
    --db_file dev_cleaned_table.db \
    ;
}


function evaluate_dev_transformer (){
  ./evaluate_ours.py \
    --data_root ../scratch \
    --parsed_std_sql_file dev_ground_truth_mark.parsed.txt \
    --parsed_pred_sql_file dev_infer_transformer.parsed.txt \
    --db_file dev_cleaned_table.db \
    ;
}

function evaluate_test (){
  ./evaluate_ours.py \
    --data_root ../scratch \
    --parsed_std_sql_file test_ground_truth_mark.parsed.txt \
    --parsed_pred_sql_file test_infer.parsed.txt \
    --db_file test_cleaned_table.db \
    ;
}

function evaluate_test_transformer (){
  ./evaluate_ours.py \
    --data_root ../scratch \
    --parsed_std_sql_file test_ground_truth_mark.parsed.txt \
    --parsed_pred_sql_file test_infer_transformer.parsed.txt \
    --db_file test_cleaned_table.db \
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
