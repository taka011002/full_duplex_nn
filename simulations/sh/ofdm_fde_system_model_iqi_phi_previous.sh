#!/bin/sh
SIMULATIONS_NAME=`basename $0 .sh`
SCRIPT_DIR=$(cd $(dirname $0); pwd)

sh $SCRIPT_DIR/base.sh $SIMULATIONS_NAME