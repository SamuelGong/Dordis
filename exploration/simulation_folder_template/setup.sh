  #!/bin/bash

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)
SETUP_HANDLER=${WORKING_DIR}"/../dev/local_setup.sh"
LAST_RESPONSE=${WORKING_DIR}"/last_response.json"

bash ${SETUP_HANDLER} "$1" ${LAST_RESPONSE} "$2"

cd ${ORIGINAL_DIR}