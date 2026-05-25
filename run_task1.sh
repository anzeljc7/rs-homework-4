#!/bin/sh
#SBATCH --job-name=gem5_task1
#SBATCH --output=gem5_task1_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --reservation=fri

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM5_BIN=$GEM5_ROOT/build/VEGA_X86/gem5.opt
APPTAINER_IMG=$GEM5_WORKSPACE/gcn-gpu_v24-0.sif

srun apptainer exec $APPTAINER_IMG make -C histogram
if [ $? -ne 0 ]; then
    echo "Build failed! Exiting."
    exit 1
fi

for KERNEL in naive opt; do
    for CU in 2 4 8; do
        OUTDIR=results_task1/${KERNEL}/cu_${CU}
        echo "=== Kernel=$KERNEL, CU=$CU ==="

        srun apptainer exec $APPTAINER_IMG \
            $GEM5_BIN --outdir=$OUTDIR \
            $GEM5_ROOT/configs/example/apu_se.py \
            -n 3 --num-compute-units $CU --gfx-version="gfx902" \
            -c ./histogram/bin/histogram_${KERNEL}.bin
    done
done

echo "Vse simulacije za nalogo 1 zakljucene."
