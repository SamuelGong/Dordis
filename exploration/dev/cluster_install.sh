#!/bin/bash
# should only be executed at the coorindator

ORIGINAL_DIR=$(pwd)
NGINX_CONFIG_REL='exploration/dev/server_nginx_config.txt'

cd `dirname $0`
WORKING_DIR=$(pwd)

# cannot even proceed if not having sudo privilege
if ! groups | grep "\<sudo\>" &> /dev/null; then
   echo "[FAILED] You need to have sudo privilege."
   exit -1
fi

# need to change according to the relative location of this script
PROJECT_DIR=${WORKING_DIR}/../..
cd ${PROJECT_DIR}

if ! which nginx > /dev/null 2>&1; then
    sudo apt update
    sudo apt install nginx -y
fi

sudo mv /etc/nginx/sites-available/default /etc/nginx/sites-available/default.bak
sudo cp ${NGINX_CONFIG_REL} /etc/nginx/sites-available/default
sudo mv /etc/nginx/sites-enabled/default .
sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl restart nginx

cd ${ORIGINAL_DIR}