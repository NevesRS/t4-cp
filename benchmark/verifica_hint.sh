#!/bin/bash
# Verifica se o --hint corrige o mapeamento de CPUs da hibrida.
#
#   sbatch --exclusive -N 4 -t 00:10:00 verifica_hint.sh
#
# Esperado no log (W ~= 45.7 core·s: calc = W / cores efetivos):
#   4x4  (nomultithread, cpus-per-task=4):  mask com 4 CPUs de cores
#        distintos; calc ~= 2.8s
#   4x8  (nomultithread, cpus-per-task=8):  mask com 8 CPUs = 1 por core
#        fisico (pares E impares, os dois sockets); calc ~= 1.4s  <-- se
#        sair ~2.8s, o socket inteiro continua so com pares: sem correcao
#   4x16 (multithread,   cpus-per-task=16): mask 0xffff; calc ~= 1.4s

export PATH=/LADAPPs/OpenMPI/openmpi-4.1.1/bin:$PATH
cd "$SLURM_SUBMIT_DIR"

H=./conway_fases_hibrido
[ -x $H ] || { echo "binario ausente: rode make antes"; exit 1; }

A=(--altura 4000 --largura 4000 --geracoes 75 --no-print --estatisticas)

echo "== topologia do no"
lscpu | grep -E "^(CPU\(s\)|Thread|Core|Socket|NUMA)"

echo; echo "== 4x4 com --hint=nomultithread"
srun -N 4 -n 4 --hint=nomultithread --cpus-per-task=4 \
    --cpu-bind=verbose --exclusive \
    env OMP_NUM_THREADS=4 OMP_DISPLAY_ENV=verbose \
    $H --threads 4 "${A[@]}"

echo; echo "== 4x8 com --hint=nomultithread (era a anomalia)"
srun -N 4 -n 4 --hint=nomultithread --cpus-per-task=8 \
    --cpu-bind=verbose --exclusive \
    env OMP_NUM_THREADS=8 OMP_DISPLAY_ENV=verbose \
    $H --threads 8 "${A[@]}"

echo; echo "== 4x16 com --hint=multithread"
srun -N 4 -n 4 --hint=multithread --cpus-per-task=16 \
    --cpu-bind=verbose --exclusive \
    env OMP_NUM_THREADS=16 OMP_DISPLAY_ENV=verbose \
    $H --threads 16 "${A[@]}"
