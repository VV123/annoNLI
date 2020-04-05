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

function convert_dev (){
  ./table_to_db.py \
    --data_root ../scratch \
    --table_file dev_cleaned_table.txt \
    --db_file dev_clean_table.db \
    ;
}

function convert_test (){
  ./table_to_db.py \
    --data_root ../scratch \
    --table_file test_cleaned_table.txt \
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
