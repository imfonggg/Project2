#!/bin/bash
# Project 3 benchmark: N = 100k and 500k, w = 2000, 5000, 10000, block size 256,
# 3 runs each.  Run from the folder that holds SDH_proj3.cu:  bash bench.sh
# Writes one log per run to logs/ and prints a summary at the end.
mkdir -p logs
for N in 100000 500000; do
  for w in 2000 5000 10000; do
    for r in 1 2 3; do
      echo "== N=$N w=$w run $r =="
      /apps/GPU_course/runScript.sh SDH_proj3.cu $N $w 256 > logs/run_${N}_${w}_${r}.txt 2>&1
      grep -E "RESULT|Total Kernel" logs/run_${N}_${w}_${r}.txt
    done
  done
done
echo; echo "==== SUMMARY (paste this back) ===="
for f in logs/run_*.txt; do
  echo "$f: $(grep -E 'RESULT|Total Kernel|Octree|error' "$f" | sed 's/\*//g; s/Total Kernel Execution Time//' | tr -s ' ' | tr '\n' ';')"
done
