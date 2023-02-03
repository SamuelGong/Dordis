#!/bin/bash

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)

APP_HANDLER=${WORKING_DIR}'/../dev/app_related.py'
STAT_HANDLER=${WORKING_DIR}'/../analysis/stat.py'
BATCH_STAT_HANDLER=${WORKING_DIR}'/../analysis/batch_stat.py'
LAST_RESPONSE=${WORKING_DIR}'/last_response.json'


case "$1" in
    start_a_task)
      CONFIG_PATH=${WORKING_DIR}"/$2"
      START_TIME=$(date "+%Y%m%d-%H%M%S")
      python ${APP_HANDLER} local_start ${WORKING_DIR} \
      ${LAST_RESPONSE} ${START_TIME} ${CONFIG_PATH}
      ;;
    kill_a_task)
      TASK_FOLDER=${WORKING_DIR}"/$2"
      python ${APP_HANDLER} local_kill ${TASK_FOLDER} \
      ${LAST_RESPONSE}
      ;;
    analyze_a_task)
      TASK_FOLDER=${WORKING_DIR}"/$2"
      python ${STAT_HANDLER} ${TASK_FOLDER}
      ;;
    analyze_tasks)
        TASK_FOLDER=${WORKING_DIR}"/$2"
        python ${BATCH_STAT_HANDLER} ${TASK_FOLDER} $3
        ;;
    *)
      echo "Unknown command!"
      ;;
esac

cd ${ORIGINAL_DIR}