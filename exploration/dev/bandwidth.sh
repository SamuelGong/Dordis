#!/bin/bash

ORIGINAL_DIR=$(pwd)
ADAPTER="ens5"

check_and_install() {
  if ! which wondershaper > /dev/null 2>&1; then
    cd ${HOME}
    sudo apt install make -y
    sudo apt install iperf3 -y
    git clone https://github.com/magnific0/wondershaper.git
    cd wondershaper
    sudo make install
  fi
}

# cannot even proceed if not having sudo privilege
if ! groups | grep "\<sudo\>" &> /dev/null; then
   echo "[FAILED] You need to have sudo privilege."
   exit -1
fi

case "$1" in
    limit_both)
      check_and_install
      sudo wondershaper -a "${ADAPTER}" -u "$2" -d "$3"
      ;;
    limit_download)
      check_and_install
      sudo wondershaper -a "${ADAPTER}" -d "$2"
      ;;
    limit_upload)
      check_and_install
      sudo wondershaper -a "${ADAPTER}" -u "$2"
      ;;
    clean_limit)
      check_and_install
      sudo wondershaper -c -a "${ADAPTER}"
      ;;
    *)
      echo "Unknown command!"
      ;;
esac

cd ${ORIGINAL_DIR}