#!/bin/bash
# Izlusci metrike za divergent in sorted SpMV pri razlicnih CU konfiguracijah.
# stats.txt ima TRI razdelke: 1. divergent kernel, 2. sorted kernel, 3. po koncu.
# Koristimo PRVI (divergent) in DRUGI (sorted) razdelek.

exec > >(tee results_task3.txt)

printf "%-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n" \
    "Kernel" "CU" "CFDiv::mean" "CFDiv::stdev" "vALUInsts" "globalReads" "globalWrites" "coalsrLines"
printf '%s\n' "------------------------------------------------------------------------------------------------------------"

for CU in 2 4 8; do
    FILE=results_task3/cu_${CU}/stats.txt

    if [ ! -f "$FILE" ]; then
        printf "%-10s | %-4s | ni podatkov\n" "(vsi)" "$CU"
        continue
    fi

    for IDX in 1 2; do
        case $IDX in
            1) LABEL="divergent" ;;
            2) LABEL="sorted" ;;
        esac

        CFM=$(awk   "/Begin Simulation/{n++} n==$IDX && /controlFlowDivergenceDist::mean/{print \$2; exit}" "$FILE")
        CFS=$(awk   "/Begin Simulation/{n++} n==$IDX && /controlFlowDivergenceDist::stdev/{print \$2; exit}" "$FILE")
        VALU=$(awk  "/Begin Simulation/{n++} n==$IDX && /vALUInsts /{print \$2; exit}" "$FILE")
        GR=$(awk    "/Begin Simulation/{n++} n==$IDX && /globalReads /{print \$2; exit}" "$FILE")
        GW=$(awk    "/Begin Simulation/{n++} n==$IDX && /globalWrites /{print \$2; exit}" "$FILE")
        COALS=$(awk "/Begin Simulation/{n++} n==$IDX && /coalsrLineAddresses::total/{print \$2; exit}" "$FILE")

        printf "%-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n" \
            "$LABEL" "$CU" "$CFM" "$CFS" "$VALU" "$GR" "$GW" "$COALS"
    done
done
