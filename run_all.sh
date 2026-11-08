#!/usr/bin/env bash
# run_all.sh - runs every experiment and makes the graphs.
#   bash run_all.sh
# Plug in the charger and close other programs first.

make || exit 1

for size in 1024 2048 4096 8192 16384; do
    ./serial $size all
    for threads in 1 2 4 8 16; do
        ./parallel $size all $threads static
    done
done

# loop schedule comparison on one size
for threads in 2 4 8; do
    ./parallel 4096 all $threads dynamic
    ./parallel 4096 all $threads guided
done

python3 plot.py
