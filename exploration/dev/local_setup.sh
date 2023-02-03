#!/bin/bash

pip install -q paramiko  # important, otherwise import error
ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)

LAST_RESPONSE="$2"
APP_HANDLER=${WORKING_DIR}'/app_related.py'

case "$1" in
    install)
        python ${APP_HANDLER} local_standalone ${LAST_RESPONSE}
        ;;
    *)
        echo "Unknown command!"
        ;;
esac

cd ${ORIGINAL_DIR}
