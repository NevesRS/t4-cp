#!/bin/bash
# Confere que as versoes paralelas produzem o mesmo tabuleiro final da versao
# sequencial (mesmo checksum), para varios tamanhos (incluindo altura nao
# divisivel) e numeros de processos/threads (ate mais processos que linhas).
# Rodar dentro de uma alocacao:
#   salloc -N 4 --exclusive bash corretude.sh
set -u
cd "$(dirname "$0")"

ALT=37; LAR=41; GER=9
esperado=$(srun --quiet -N 1 -n 1 ./conway_seq --altura $ALT --largura $LAR \
           --geracoes $GER --no-print | cut -d' ' -f2)
falhas=0

testar_mpi() {  # testar_mpi <processos> <altura>
    local np=$1 alt=$2 saida h ref
    # Referencia por altura: o esperado de $ALT ja foi calculado acima.
    ref=$esperado
    if [[ "$alt" != "$ALT" ]]; then
        ref=$(srun --quiet -N 1 -n 1 ./conway_seq --altura "$alt" \
              --largura $LAR --geracoes $GER --no-print | cut -d' ' -f2)
    fi
    saida=$(srun --quiet -N 4 -n "$np" ./conway_fases_mpi --altura "$alt" \
            --largura $LAR --geracoes $GER --no-print 2>&1 |
            grep -E '^[0-9.]+ [0-9a-f]{16}$' | head -1)
    h=$(echo "$saida" | cut -d' ' -f2)
    if [[ "$h" == "$ref" ]]; then
        echo "OK    mpi np=$np ${alt}x${LAR}"
    else
        echo "FALHA mpi np=$np ${alt}x${LAR}: esperado $ref, obtido '$saida'"
        falhas=$((falhas + 1))
    fi
}

testar_hib() {  # testar_hib <threads> <altura>
    local th=$1 alt=$2 saida h
    saida=$(srun --quiet -N 4 -n 4 --cpus-per-task="$th" \
            ./conway_fases_hibrido --threads "$th" --altura "$alt" \
            --largura $LAR --geracoes $GER --no-print 2>&1 |
            grep -E '^[0-9.]+ [0-9a-f]{16}$' | head -1)
    h=$(echo "$saida" | cut -d' ' -f2)
    if [[ "$h" == "$esperado" ]]; then
        echo "OK    hibrido thr=$th ${alt}x${LAR}"
    else
        echo "FALHA hibrido thr=$th ${alt}x${LAR}: esperado $esperado, obtido '$saida'"
        falhas=$((falhas + 1))
    fi
}

# O anel de troca de linhas fantasma exige np <= altura: com mais processos
# que linhas, sobram ranks com 0 linhas e o vizinho manda memoria nao
# inicializada. Por isso os testes com 64 processos usam altura 100.
for np in 1 2 3 4 5 7 8 16 31 32 37; do
    testar_mpi "$np" $ALT
done
testar_mpi 64 100
for th in 1 2 4 8 16; do
    testar_hib "$th" $ALT
done

# Tamanho real do trabalho (so tempo e checksum, sem comparacao)
echo "1500x1500 x75 mpi np=32:      $(srun -N 4 -n 32 ./conway_fases_mpi --no-print)"
echo "1500x1500 x75 hibrido 4x8thr: $(srun -N 4 -n 4 --cpus-per-task=8 ./conway_fases_hibrido --threads 8 --no-print)"
echo "falhas: $falhas"
exit $falhas
