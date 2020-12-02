#!/bin/sh
SIMULATIONS_NAME="snr_ber_average_ibo"
CONFIG=`cat $PYTHONPATH/simulations/configs/$SIMULATIONS_NAME.json | jq .  -c`
NOW=`date "+%Y/%m/%d/%H_%M_%S"`
OUT="$PYTHONPATH/results/$SIMULATIONS_NAME/$NOW"
mkdir -p $OUT
echo "out: $OUT"

nohup python $PYTHONPATH/simulations/$SIMULATIONS_NAME.py -c $CONFIG -o $OUT > "$OUT/nohup" &