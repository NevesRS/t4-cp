#!/bin/bash
# Confere que as versoes paralelas produzem o mesmo tabuleiro final da versao
# sequencial (mesmo checksum), para varios tamanhos (incluindo altura nao
# divisivel) e numeros de processos/threads (ate mais processos que linhas).
# Rodar dentro de uma alocacao:
#   salloc -N 4 --exclusive bash corretude.sh
set -u
cd "$(dirname "$0")"

ALT=37; LAR=41; GER=9
esperado=$(srun -N 1 -n 1 ./conway_seq --altura $ALT --largura $LAR \
           --geracoes $GER --no-print | cut -d' ' -f2)
falhas=0

testar_mpi() {  # testar_mpi <processos>
    local np=$1 saida h
    saida=$(srun -N 4 -n "$np" ./conway_fases_mpi --altura $ALT --largura $LAR \
            --geracoes $GER --no-print 2>&1 | head -1)
    h=$(echo "$saida" | cut -d' ' -f2)
    if [[ "$h" == "$esperado" ]]; then
        echo "OK    mpi np=$np ${ALT}x${LAR}"
    else
        echo "FALHA mpi np=$np: esperado $esperado, obtido '$saida'"
        falhas=$((falhas + 1))
    fi
}

testar_hib() {  # testar_hib <threads>
    local th=$1 saida h
    saida=$(srun -N 4 -n 4 --cpus-per-task="$th" ./conway_fases_hibrido \
            --threads "$th" --altura $ALT --largura $LAR --geracoes $GER \
            --no-print 2>&1 | head -1)
    h=$(echo "$saida" | cut -d' ' -f2)
    if [[ "$h" == "$esperado" ]]; then
        echo "OK    hibrido thr=$th ${ALT}x${LAR}"
    else
        echo "FALHA hibrido thr=$th: esperado $esperado, obtido '$saida'"
        falhas=$((falhas + 1))
    fi
}

for np in 1 2 3 4 5 7 8 16 31 32 37 64; do
    testar_mpi "$np"
done
for th in 1 2 4 8 16; do
    testar_hib "$th"
done

# Tamanho real do trabalho (so tempo e checksum, sem comparacao)
echo "1500x1500 x75 mpi np=32:      $(srun -N 4 -n 32 ./conway_fases_mpi --no-print)"
echo "1500x1500 x75 hibrido 4x8thr: $(srun -N 4 -n 4 --cpus-per-task=8 ./conway_fases_hibrido --threads 8 --no-print)"
echo "falhas: $falhas"
exit $falhas
