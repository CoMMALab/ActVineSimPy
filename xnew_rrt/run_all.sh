#!/bin/bash
# Every env, one at a time (so the runtimes are not sharing the GPU with each other), with one
# checkpoint and a large budget. Per env: the console log in <out>/logs/<env>.run.log and a row of
# runtimes.tsv (process wall time, including startup and CUDA graph capture). gpu.csv samples the
# GPU every 10 s, with the number of other processes on it, to show whether a runtime was contended.
#
#   bash xnew_rrt/run_all.sh checkpoints/wrinkle_mu03.json
#   MAX_EDGES=400000 TIME_LIMIT=120 MAX_ATTEMPTS=200 bash xnew_rrt/run_all.sh <checkpoint> [envs...]
cd "$(dirname "$0")/.."
CK=${1:?checkpoint}; shift
ENVS=${@:-envs/*.json}
OUT=${OUT:-out/solutions}
MAX_EDGES=${MAX_EDGES:-400000}; TIME_LIMIT=${TIME_LIMIT:-120}; MAX_ATTEMPTS=${MAX_ATTEMPTS:-200}
PY="xnew_rrt/py.sh"
mkdir -p $OUT/logs
TSV=$OUT/logs/runtimes.tsv
echo -e "env\tstart\twall_s\texit\tresult" > $TSV
(while true; do
   echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits),$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
   sleep 10
 done) > $OUT/logs/gpu.csv &
MON=$!
trap "kill $MON" EXIT
for env in $ENVS; do
  name=$(basename $env .json)
  log=$OUT/logs/$name.run.log
  t0=$(date +%s.%N)
  if [[ $name == 5* ]]; then
    $PY xnew_rrt/block_puzzle.py $env --checkpoint $CK --max-attempts $MAX_ATTEMPTS --out $OUT 2>&1 \
      | grep -v -i "warn\|OpenCV" > $log
  else
    $PY xnew_rrt/run_envs.py $env --checkpoint $CK --max-edges $MAX_EDGES --time-limit $TIME_LIMIT --out $OUT 2>&1 \
      | grep -v -i "warn\|OpenCV" | tr '\r' '\n' | grep -v "^ rrt iter" > $log
  fi
  code=${PIPESTATUS[0]}
  wall=$(echo "$(date +%s.%N) - $t0" | bc)
  echo -e "$name\t$(date -d @${t0%.*} +%FT%T)\t$wall\t$code\t$(tail -1 $log)" >> $TSV
  echo "$name ${wall}s: $(tail -1 $log)"
done
git checkout -q -- '*.pyc'
