#!/bin/bash

LOCAL_PRIVATE_KEY=${HOME}'/.ssh/MyKeyPair.pem'
ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)

APP_HANDLER=${WORKING_DIR}'/../dev/app_related.py'
STAT_HANDLER=${WORKING_DIR}'/../analysis/stat.py'
BATCH_STAT_HANDLER=${WORKING_DIR}'/../analysis/batch_stat.py'
SETUP_SCRIPT=${WORKING_DIR}'/../dev/remote_setup.sh'
EC2_LAUNCH_RESULT=${WORKING_DIR}'/ec2_launch_result.json'
LAST_RESPONSE=${WORKING_DIR}'/last_response.json'

case "$1" in
    start_a_task)
        CONFIG_PATH=${WORKING_DIR}"/$2"
        START_TIME=$(date "+%Y%m%d-%H%M%S")
        bash ${SETUP_SCRIPT} start_cluster ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        python ${APP_HANDLER} remote_start ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} \
          ${START_TIME} ${CONFIG_PATH}
        ;;
    kill_a_task)
        TASK_FOLDER=${WORKING_DIR}"/$2"
        python ${APP_HANDLER} remote_kill ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${TASK_FOLDER}
        ;;
    conclude_a_task)
        TASK_FOLDER=${WORKING_DIR}"/$2"
        python ${APP_HANDLER} collect_result ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${TASK_FOLDER}
        ;;
    analyze_a_task)
        TASK_FOLDER=${WORKING_DIR}"/$2"
        python ${STAT_HANDLER} ${TASK_FOLDER}
#        echo python ../../${PLOT_HANDLER} --config=./${CONFIG_FILE_REL} \
#          > "${TASK_FOLDER}/${PLOT_REL}"
#        echo python ../../${TIME_SEQ_PLOT_HANDLER} --config=./${CONFIG_FILE_REL} \
#          > "${TASK_FOLDER}/${TIME_SEQ_PLOT_REL}"
#        echo python ../../${SCORES_PLOT_HANDLER} --config=./${CONFIG_FILE_REL} \
#          > "${TASK_FOLDER}/${SCORES_PLOT_REL}"
#        cd ${TASK_FOLDER}
#        bash ${PLOT_REL}
#        bash ${TIME_SEQ_PLOT_REL}
#        bash ${SCORES_PLOT_REL}
#        cd ${CWD}
        ;;
    analyze_tasks)
        TASK_FOLDER=${WORKING_DIR}"/$2"
        python ${BATCH_STAT_HANDLER} ${TASK_FOLDER}
        ;;
    *)
      echo "Unknown command!"
      ;;
esac

cd ${ORIGINAL_DIR}

