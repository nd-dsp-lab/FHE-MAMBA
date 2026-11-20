#!/bin/sh
# Bootstrap sweep: which (levels kept, level budget, iterations) fits ring 2^16, at what precision.
# One configuration at a time: ring 2^17 key sets take ~55 GB.
cd "$(dirname "$0")"
out=logs/boot_sweep.log
: > "$out"
for cfg in "19 4 4 2" "12 3 3 2" "10 3 3 2" "10 4 4 2" "10 3 3 1" "8 3 3 2" "8 2 2 2" "6 3 3 2"; do
    set -- $cfg
    echo "== levels $1 budget $2,$3 iters $4" >> "$out"
    ./build/bench --boot --levels-after "$1" --budget "$2" "$3" --boot-iters "$4" >> "$out" 2>&1 || echo "FAILED levels $1 budget $2,$3 iters $4" >> "$out"
done
echo SWEEPDONE >> "$out"
