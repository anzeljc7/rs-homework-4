#!/bin/bash
# Izlusci metrike za histogram naive in opt pri razlicnih CU konfiguracijah.
# stats.txt ima DVA razdelka: 1. med izvajanjem kernela, 2. po koncu.
# Koristimo PRVI razdelek.

exec > >(tee results_task1.txt)

printf "%-8s | %-4s | %-16s | %-12s | %-14s | %-12s | %-8s\n" \
    "Kernel" "CU" "loadLatency" "vALUInsts" "ldsBankAccess" "totalCycles" "vpc"
printf '%s\n' "---------------------------------------------------------------------------------"

for KERNEL in naive opt; do
    for CU in 2 4 8; do
        FILE=results_task1/${KERNEL}/cu_${CU}/stats.txt

        if [ ! -f "$FILE" ]; then
            printf "%-8s | %-4s | ni podatkov\n" "$KERNEL" "$CU"
            continue
        fi

        LAT=$(awk  '/Begin Simulation/{n++} n==1 && /loadLatencyDist::mean/{print $2; exit}' "$FILE")
        VALU=$(awk '/Begin Simulation/{n++} n==1 && /vALUInsts /{print $2; exit}' "$FILE")
        LDS=$(awk  '/Begin Simulation/{n++} n==1 && /ldsBankAccess/{print $2; exit}' "$FILE")
        CYCS=$(awk '/Begin Simulation/{n++} n==1 && /totalCycles /{print $2; exit}' "$FILE")
        VPC=$(awk  '/Begin Simulation/{n++} n==1 && /vpc /{print $2; exit}' "$FILE")

        printf "%-8s | %-4s | %-16s | %-12s | %-14s | %-12s | %-8s\n" \
            "$KERNEL" "$CU" "$LAT" "$VALU" "$LDS" "$CYCS" "$VPC"
    done
done
