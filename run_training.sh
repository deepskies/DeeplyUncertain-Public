#!/bin/bash

SCRIPT=train_network.py
TIME=$(date +%s)
LOGFILE=${TIME}.log

for MODEL_TYPE in cd bnn de
do
  for T_MAX in 0.05 0.1 0.2
  do
    if [ ${MODEL_TYPE} == de ]
    then
      N_EPOCHS=40
    else
      N_EPOCHS=200
    fi
    ARGS="--model-type ${MODEL_TYPE} --t-spread-max ${T_MAX} --n-epochs ${N_EPOCHS}"
    time python ${SCRIPT} ${ARGS} >> ${LOGFILE}
  done
done
