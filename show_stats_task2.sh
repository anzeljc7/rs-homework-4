#!/bin/bash
# Izlusci metrike za naive in shared memory matrix transpose pri razlicnih CU.
# stats.txt ima TRI razdelke: 1. naive kernel, 2. shared kernel, 3. po koncu.
# Koristimo PRVI (naive) in DRUGI (shared) razdelek.

printf "%-8s | %-4s | %-16s | %-12s | %-14s | %-12s | %-8s\n" \
    "Kernel" "CU" "loadLatency" "vALUInsts" "ldsBankAccess" "totalCycles" "vpc"
printf '%s\n' "---------------------------------------------------------------------------------"

for CU in 2 4 8; do
    FILE=results_task2/cu_${CU}/stats.txt

    if [ ! -f "$FILE" ]; then
        printf "%-8s | %-4s | ni podatkov\n" "(vsi)" "$CU"
        continue
    fi

    for IDX in 1 2; do
        case $IDX in
            1) LABEL="naive" ;;
            2) LABEL="shared" ;;
        esac

        LAT=$(awk  "/Begin Simulation/{n++} n==$IDX && /loadLatencyDist::mean/{print \$2; exit}" "$FILE")
        VALU=$(awk "/Begin Simulation/{n++} n==$IDX && /vALUInsts /{print \$2; exit}" "$FILE")
        LDS=$(awk  "/Begin Simulation/{n++} n==$IDX && /ldsBankAccess/{print \$2; exit}" "$FILE")
        CYCS=$(awk "/Begin Simulation/{n++} n==$IDX && /totalCycles /{print \$2; exit}" "$FILE")
        VPC=$(awk  "/Begin Simulation/{n++} n==$IDX && /vpc /{print \$2; exit}" "$FILE")

        printf "%-8s | %-4s | %-16s | %-12s | %-14s | %-12s | %-8s\n" \
            "$LABEL" "$CU" "$LAT" "$VALU" "$LDS" "$CYCS" "$VPC"
    done
done
