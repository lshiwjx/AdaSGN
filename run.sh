#!/usr/bin/env bash
#printf $0
#printf $(dirname $0)
cd $(dirname $0)

rm -rf code
hadoop fs -get hdfs://haruna/home/byte_labcv_default/user/shilei.stone/ImageToVideo/code ./
cd code

printf $1
python3 train.py -config $1

echo $(ls ../experiments/)
hadoop fs -put -f ../experiments hdfs://haruna/home/byte_labcv_default/user/shilei.stone/ImageToVideo/
# -config config/flow/adam_hdfs.yaml

