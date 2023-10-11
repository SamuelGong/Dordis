#!/bin/bash

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)

EC2_LAUNCH_RESULT=${WORKING_DIR}'/ec2_launch_result.json'
LOCAL_PRIVATE_KEY=${HOME}'/.ssh/MyKeyPair.pem'
#BATCH_PLAN=${WORKING_DIR}'/batch_plan.txt'
BATCH_PLAN=${WORKING_DIR}/"$1"
BATCH_HANDLER=${WORKING_DIR}'/../dev/batch_run.py'
LOG_FILE=${WORKING_DIR}'/batch_log.txt'

python ${BATCH_HANDLER} start_tasks ${EC2_LAUNCH_RESULT} \
    ${LOCAL_PRIVATE_KEY} ${BATCH_PLAN} > ${LOG_FILE} 2>&1 &
