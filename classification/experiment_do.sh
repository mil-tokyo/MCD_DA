#!/usr/bin/bash

echo "\$1（Source）: $1"
echo "\$2（Target）: $2"
echo "\$3（Epoch）: $3"
echo "\$4（number of k): $4"
echo "\$5（all_use(option for usps)): $5"
echo "\$6（gpuid): $6"
SOURCE=$1
TARGET=$2
EPOCH=$3
NUMK=$4
ALLU=$5
GPUID=$6
for i in `seq 1 5`
do
CUDA_VISIBLE_DEVICES=${GPUID} python main.py --source ${SOURCE} --target ${TARGET} --num_k ${NUMK} --max_epoch ${EPOCH} --all_use ${ALLU}
done