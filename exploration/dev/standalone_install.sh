#!/bin/bash

# To avoid error "Font family 'Times New Roman' not found."
sudo apt install msttcorefonts -qq
rm ~/.cache/matplotlib -rf

# make it suitable for your time zone
sudo timedatectl set-timezone Asia/Hong_Kong

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)

# need to change according to the relative location of this script
PROJECT_DIR=${WORKING_DIR}/../..
echo $PROJECT_DIR
CONDA_ENV_NAME='dordis'

# install anaconda if necessary
CONDA_DIR=${HOME}/anaconda3
if [ ! -d ${CONDA_DIR} ]; then
  echo "[INFO] Install Anaconda Package Manager..."
  cd ~
  wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  bash Anaconda3-2020.11-Linux-x86_64.sh -b -p ${CONDA_DIR}
  export PATH=${CONDA_DIR}/bin:$PATH
  rm Anaconda3-2020.11-Linux-x86_64.sh
  conda init bash
else
  echo "[INFO] Anaconda already installed."
fi

source ~/anaconda3/etc/profile.d/conda.sh
ENVS=$( conda env list | awk '{print $1}' )

if [[ $ENVS = *"${CONDA_ENV_NAME}"* ]]; then
  echo "[INFO] Environment ${CONDA_ENV_NAME} already exists."
else
  echo "[INFO] Create environment "${CONDA_ENV_NAME}"..."
  conda create -n ${CONDA_ENV_NAME} python=3.8 -y
  conda activate ${CONDA_ENV_NAME}

  echo "[INFO] Installing dependencies..."
  cd ${PROJECT_DIR}
  chmod u+x ./run
  pip install -r requirements.txt --upgrade
  pip install yapf mypy pylint

  conda deactivate
fi

if ! which redis-server > /dev/null 2>&1; then
    sudo apt update
    sudo apt install redis-server -y
fi

# used by dev/utils
pip install -q paramiko  # should be outside the env dordis
cd ${ORIGINAL_DIR}
