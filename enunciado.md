# T4 (TPP4) — MPI Fases Paralelas

> Transcrito do Moodle em 17/11/2026.
>
> - **Prazo/atualização: quinta-feira, 12 nov. 2026, 20:04** — Grupo 12
> - Rubrica: verificar os itens de avaliação no Moodle ("Verificar os itens de
>   avaliação na rubrica para este trabalho")

## Condições de conclusão (texto do Moodle)

O objetivo do trabalho é implementar, usando a biblioteca MPI, uma versão
paralela seguindo o modelo **fases paralelas**, de um programa escolhido pelo
grupo que seja adequado a este modelo (**uma versão MPI puro e uma versão
híbrida MPI com OpenMP**). Após implementado, o programa deve ser executado no
cluster atlântica com **4 nós (32/64 processos, no segundo caso com HT)** para
realização das medições de desempenho. Cada grupo deve entregar um **relatório
em .pdf de uma página** com a análise dos resultados e o **código em anexo**
(seguir modelo proposto).

## Modelo fases paralelas

Cada um dos processos é responsável por **1/np do trabalho**, que já pode ser
gerado internamente em cada processo, ou gerado em um único processo e depois
distribuído em partes iguais entre os restantes. Depois, o processamento segue
as seguintes fases, que são executadas **em paralelo por todos os processos**
até a conclusão (por isso o nome do modelo):

1. **Processamento local:** cada processo realiza a sua parte do trabalho global;
2. **Verificação distribuída** para identificar se o trabalho está pronto
   (**comunicação coletiva**);
3. Se não, cada processo **troca valores com seus vizinhos** em busca de
   convergência (padrão depende da lógica do problema).

O foco deste trabalho é a **comparação da versão em MPI puro (np 32/64) com a
versão híbrida MPI/OpenMP (np 4, 8 e 16 threads dentro do nó)**.

## Decisões do grupo

- **Programa:** Jogo da Vida de Conway (problema do T1/T2), 1500x1500, 75
  gerações, densidade inicial 20% (`srand(42)`).
- **Condição de término:** máximo de gerações (como no T1/T2); a verificação
  distribuída (fase 2) conta as células que mudaram (`MPI_Allreduce`) e encerra
  antes se o tabuleiro estabilizar.
- **Partição:** cada processo gera internamente sua fatia de linhas (1/np) e
  troca as linhas fantasma com os vizinhos do toro a cada geração
  (`MPI_Sendrecv`).
- **Híbrida:** 4 processos MPI (1 por nó), com 4, 8 e 16 threads OpenMP dentro
  do nó (16 = HT). Comparação direta com MPI puro np = 16, 32 e 64.

## Execução no LAD

```bash
mpicc file.c -o file_exec               # compilar (na hospedeira)
salloc -N 4 --exclusive                 # alocar 4 nos
srun -N 4 -n 32 ./conway_fases_mpi      # fisico
srun -N 4 -n 64 ./conway_fases_mpi      # com HT
srun -N 4 -n 4 --cpus-per-task=8 ./conway_fases_hibrido   # 4 ranks x 8 threads
```
