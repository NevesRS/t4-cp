#!/bin/bash
# Diagnostico da anomalia da hibrida 4x8: com 8 threads por no o calc por
# rank nao cai (fica igual ao 4x4), mas com 4 e 16 ele escala normalmente.
# Hipotese principal: mapeamento errado de cores pelo binding do srun/OpenMP.
#
#   sbatch --exclusive -N 4 -t 00:10:00 diagnostico.sh
#
# O log (diag-<jobid>.log) traz, para cada variante:
#   - o mapa de CPUs de cada task MPI (srun --cpu-bind=verbose);
#   - onde cada thread OpenMP pousou (OMP_DISPLAY_ENV=verbose);
#   - o calc por rank (# rank ...) para comparar com o benchmark.
# Variante 2 re-roda o 4x8 com posicionamento explicito de threads:
# OMP_PLACES=cores + OMP_PROC_BIND=spread (um core fisico por thread,
# espalhado entre os sockets). Se o 4x8 "spread" dobrar o ganho do 4x4,
# o binding era o problema.

export PATH=/LADAPPs/OpenMPI/openmpi-4.1.1/bin:$PATH
cd "$SLURM_SUBMIT_DIR"

H=./conway_fases_hibrido
[ -x $H ] || { echo "binario ausente: rode make antes"; exit 1; }

A=(--altura 4000 --largura 4000 --geracoes 75 --no-print --estatisticas)

echo "== topologia do no"
lscpu | grep -E "^(CPU\(s\)|Thread|Core|Socket|NUMA)"

echo; echo "== 1) 4x8 default (reproduz a anomalia)"
srun -N 4 -n 4 --cpus-per-task=8 --cpu-bind=verbose --exclusive \
    env OMP_NUM_THREADS=8 OMP_DISPLAY_ENV=verbose \
    $H --threads 8 "${A[@]}"

echo; echo "== 2) 4x8 com OMP_PLACES=cores OMP_PROC_BIND=spread"
srun -N 4 -n 4 --cpus-per-task=8 --cpu-bind=verbose --exclusive \
    env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=spread \
    OMP_DISPLAY_ENV=verbose \
    $H --threads 8 "${A[@]}"

echo; echo "== 3) 4x4 controle"
srun -N 4 -n 4 --cpus-per-task=4 --cpu-bind=verbose --exclusive \
    env OMP_NUM_THREADS=4 OMP_DISPLAY_ENV=verbose \
    $H --threads 4 "${A[@]}"

echo; echo "== 4) 4x16 controle (HT)"
srun -N 4 -n 4 --cpus-per-task=16 --cpu-bind=verbose --exclusive \
    env OMP_NUM_THREADS=16 OMP_DISPLAY_ENV=verbose \
    $H --threads 16 "${A[@]}"
