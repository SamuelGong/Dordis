#!/bin/bash

pip install -q boto3
pip install -q paramiko
ORIGINAL_DIR=$(pwd)
LOCAL_PRIVATE_KEY=${HOME}'/.ssh/MyKeyPair.pem'

cd `dirname $0`
WORKING_DIR=$(pwd)
DEV_PATH=${WORKING_DIR}'/../dev'

EC2_NODE_TEMPLATE=${DEV_PATH}'/ec2_node_template.yml'
EC2_HANDLER=${DEV_PATH}'/ec2.py'
EC2_CONFIG=${WORKING_DIR}'/ec2_cluster_config.yml'
EC2_LAUNCH_RESULT=${WORKING_DIR}'/ec2_launch_result.json'
LAST_RESPONSE=${WORKING_DIR}'/last_response.json'

case "$1" in
    launch)
        python ${EC2_HANDLER} launch ${EC2_CONFIG} \
          ${EC2_NODE_TEMPLATE} ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    scale)
        python ${EC2_HANDLER} scale ${EC2_CONFIG} \
          ${EC2_NODE_TEMPLATE} ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    start)
        python ${EC2_HANDLER} start ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT} $2
        ;;
    stop)
        python ${EC2_HANDLER} stop ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    terminate)
        python ${EC2_HANDLER} terminate ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    reboot)
        python ${EC2_HANDLER} reboot ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    show)
        python ${EC2_HANDLER} show ${EC2_LAUNCH_RESULT}
        ;;
    limit_bandwidth)
        python ${EC2_HANDLER} limit_bandwidth ${LAST_RESPONSE} \
          ${EC2_LAUNCH_RESULT} ${LOCAL_PRIVATE_KEY} ${DEV_PATH} ${EC2_CONFIG}
        ;;
    free_bandwidth)
        python ${EC2_HANDLER} free_bandwidth ${LAST_RESPONSE} \
          ${EC2_LAUNCH_RESULT} ${LOCAL_PRIVATE_KEY} ${DEV_PATH}
        ;;
    *)
        echo "Unknown command!"
        ;;
esac

cd ${ORIGINAL_DIR}