  #!/bin/bash

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)
SETUP_HANDLER=${WORKING_DIR}"/../dev/remote_setup.sh"
LAST_RESPONSE=${WORKING_DIR}"/last_response.json"
EC2_LAUNCH_RESULT=${WORKING_DIR}"/ec2_launch_result.json"

bash ${SETUP_HANDLER} "$1" ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT} "$2"

cd ${ORIGINAL_DIR}