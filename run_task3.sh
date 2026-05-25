#!/bin/sh
#SBATCH --job-name=gem5_task3
#SBATCH --output=gem5_task3_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --reservation=fri

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM5_BIN=$GEM5_ROOT/build/VEGA_X86/gem5.opt
APPTAINER_IMG=$GEM5_WORKSPACE/gcn-gpu_v24-0.sif

srun apptainer exec $APPTAINER_IMG make -C spmv
if [ $? -ne 0 ]; then
    echo "Build failed! Exiting."
    exit 1
fi

# Ena binarka vsebuje oba kernela (divergent in sorted); stats.txt ima tri razdelke.
for CU in 2 4 8; do
    OUTDIR=results_task3/cu_${CU}
    echo "=== CU=$CU ==="

    srun apptainer exec $APPTAINER_IMG \
        $GEM5_BIN --outdir=$OUTDIR \
        $GEM5_ROOT/configs/example/apu_se.py \
        -n 3 --num-compute-units $CU --gfx-version="gfx902" \
        -c ./spmv/bin/spmv.bin
done

echo "Vse simulacije za nalogo 3 zakljucene."
