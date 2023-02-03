#!/bin/bash

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)

BATCH_PLAN=${WORKING_DIR}'/batch_plan.txt'
BATCH_HANDLER=${WORKING_DIR}'/../dev/batch_run.py'
LOG_FILE=${WORKING_DIR}'/batch_log.txt'

python ${BATCH_HANDLER} local_start_tasks \
    ${BATCH_PLAN} > ${LOG_FILE} 2>&1 &
