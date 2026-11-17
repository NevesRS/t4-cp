#!/bin/bash
# Submete o benchmark na atlantica (4 nos exclusivos):
#   sbatch -N 1 -t 02:00:00 job.sh seq      # referencia sequencial em 1 core
#   sbatch job.sh forte hibrida             # MPI puro e hibrida em 4 nos
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -J t4-fases
#SBATCH -o job-%j.log

export PATH=/LADAPPs/OpenMPI/openmpi-4.1.1/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
echo "nos: $SLURM_JOB_NODELIST"
python3 -u benchmark.py "$@"
gi