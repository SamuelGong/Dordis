#!/bin/bash

GITHUB_REPO='git@github.com:SamuelGong/Hyades.git'
REPO_BRANCH='main'
pip install -q paramiko

LOCAL_PRIVATE_KEY=${HOME}'/.ssh/MyKeyPair.pem'
NODE_PRIVATE_KEY=${HOME}'/.ssh/id_rsa'
ORIGINAL_DIR=$(pwd)
LAST_RESPONSE="$2"
EC2_LAUNCH_RESULT="$3"

cd `dirname $0`
WORKING_DIR=$(pwd)

GITHUB_HANDLER=${WORKING_DIR}'/github_related.py'
APP_HANDLER=${WORKING_DIR}'/app_related.py'

case "$1" in
    install)
        python ${GITHUB_HANDLER} initialize ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${NODE_PRIVATE_KEY} \
          ${GITHUB_REPO}
        python ${GITHUB_HANDLER} checkout ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${GITHUB_REPO} \
          ${REPO_BRANCH}
        python ${APP_HANDLER} standalone ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE}
        ;;
    update)
        python ${GITHUB_HANDLER} pull ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${GITHUB_REPO}
        ;;
    deploy_cluster)
        python ${APP_HANDLER} deploy_cluster ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE}
        ;;
    start_cluster)
        python ${APP_HANDLER} start_cluster ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE}
        ;;
    add_pip_dependency)
        python ${APP_HANDLER} add_pip_dependency ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} "$4"
        ;;
    add_apt_dependency)
        python ${APP_HANDLER} add_apt_dependency ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} "$4"
        ;;
    clean_memory)
        python ${APP_HANDLER} clean_memory ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE}
        ;;
    *)
        echo "Unknown command!"
        ;;
esac

cd ${ORIGINAL_DIR}
