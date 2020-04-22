#!/bin/bash

SCRIPT=test_network.py

for MODEL_TYPE in cd bnn de
do
  if [ ${MODEL_TYPE} == de ]
  then
    N_EPOCHS=40
  else
    N_EPOCHS=200
  fi
  for ell in {2..15}
  do
    let ell_max=$ell+1
    ARGS="--model-type ${MODEL_TYPE} --ell-range-min ${ell} --ell-range-max ${ell_max} --n-epochs ${N_EPOCHS} --t-spread-max 0.2"
    time python ${SCRIPT} ${ARGS}
  done
  ARGS="--model-type ${MODEL_TYPE} --ell-range-min 8 --ell-range-max 12 --n-epochs ${N_EPOCHS} --t-spread-max 0.2"
  time python ${SCRIPT} ${ARGS}
  ARGS="--model-type ${MODEL_TYPE} --ell-range-min 12 --ell-range-max 16 --n-epochs ${N_EPOCHS} --t-spread-max 0.2"
  time python ${SCRIPT} ${ARGS}
  ARGS="--model-type ${MODEL_TYPE} --g-range-min 15 --g-range-max 20 --n-epochs ${N_EPOCHS} --t-spread-max 0.2"
  time python ${SCRIPT} ${ARGS}
  for g in {10..24}
  do
    let g_max=$g+1
    ARGS="--model-type ${MODEL_TYPE} --g-range-min ${g} --g-range-max ${g_max} --n-epochs ${N_EPOCHS} --t-spread-max 0.2"
    time python ${SCRIPT} ${ARGS}
  done
done
