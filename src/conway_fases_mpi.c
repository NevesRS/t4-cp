/*
 * Jogo da Vida de Conway -- versao paralela MPI, modelo fases paralelas
 *
 * MODELAGEM
 *
 * O tabuleiro (toro altura x largura) e dividido em np fatias de linhas
 * consecutivas, uma por processo, de modo que cada processo fica com 1/np do
 * trabalho. Ninguem monta o tabuleiro inteiro: cada processo o gera
 * internamente, percorrendo a mesma sequencia de rand() da versao sequencial
 * (mesmo srand(42)) e gravando apenas as linhas da sua fatia. Assim o estado
 * inicial global e identico ao da linha de base, sem comunicacao.
 *
 * Cada geracao e um ciclo das fases do modelo, executado por todos os
 * processos em paralelo:
 *
 *   Fase 0 (preparacao do ciclo): troca de linhas fantasma -- cada processo
 *     envia sua primeira linha ao vizinho de cima e sua ultima ao vizinho de
 *     baixo (MPI_Sendrecv, duas chamadas). Como o tabuleiro e um toro, o
 *     vizinho de cima do rank 0 e o rank np-1. As linhas fantasma sao apenas
 *     lidas; a fatia de cada processo nunca e calculada por outro processo.
 *
 *   Fase 1 (processamento local): cada processo calcula a proxima geracao das
 *     suas linhas. De quebra, conta quantas celulas mudaram de estado.
 *
 *   Fase 2 (verificacao distribuida, comunicacao coletiva): MPI_Allreduce
 *     soma as mudancas de todos. Se ninguem mudou, o tabuleiro e estavel e o
 *     trabalho acabou (a geracao seguinte seria identica). O limite de
 *     geracoes (argumento --geracoes, como no T1/T2) continua valendo.
 *
 *   Fase 3 (troca com vizinhos em busca de convergencia): e a troca de linhas
 *     fantasma da Fase 0 do ciclo seguinte -- as novas geracoes dos vizinhos
 *     so chegam depois dela, e o ciclo recomeca.
 *
 * O nucleo de calculo por celula e identico ao da versao sequencial, menos
 * os modulos verticais: com as linhas fantasma, o vizinho de cima/baixo de
 * qualquer linha local esta dentro do proprio buffer. O modulo horizontal
 * permanece, pois a fatia contem linhas inteiras.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_LARGURA 1500
#define DEFAULT_ALTURA 1500
#define DEFAULT_GERACOES 75
#define DENSIDADE_INICIAL 20 /* % de celulas vivas no tabuleiro inicial */

/* A mensagem da Fase 3 (linhas fantasma) circula no toro em dois sentidos:
   cada sentido usa uma tag propria para as duas trocas coexistirem. */
#define TAG_SENTIDO_CIMA 1  /* primeira linha local viaja para cima   */
#define TAG_SENTIDO_BAIXO 2 /* ultima linha local viaja para baixo */

static int largura = DEFAULT_LARGURA;
static int altura = DEFAULT_ALTURA;
static int geracoes = DEFAULT_GERACOES;
static int silent = 0;
static int estatisticas = 0;

static unsigned char *alocar_tabuleiro(int linhas) {
    unsigned char *tabuleiro = malloc((size_t)linhas * largura);
    if (tabuleiro == NULL) {
        fprintf(stderr, "Erro ao alocar o tabuleiro.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return tabuleiro;
}

/*
 * Gera a fatia do processo dentro do tabuleiro inicial. A sequencia de
 * rand() e global (linha a linha, da esquerda para a direita), entao o
 * processo percorre o tabuleiro inteiro e grava so as suas linhas
 * [inicio, fim). Custo extra de O(tabuleiro) por processo, uma unica vez,
 * fora da regiao cronometrada; em troca, zero comunicacao na inicializacao
 * e tabuleiro inicial identico ao da versao sequencial.
 */
static void gerar_fatia(unsigned char *fatia, int inicio, int fim) {
    for (int i = 0; i < inicio; i++) {
        for (int j = 0; j < largura; j++) {
            (void)rand();
        }
    }
    for (int i = inicio; i < fim; i++) {
        for (int j = 0; j < largura; j++) {
            fatia[(size_t)(i - inicio + 1) * largura + j] =
                (rand() % 100 < DENSIDADE_INICIAL) ? 1 : 0;
        }
    }
    for (int i = fim; i < altura; i++) {
        for (int j = 0; j < largura; j++) {
            (void)rand();
        }
    }
}

/*
 * Conta as celulas vivas nas 8 posicoes adjacentes a (x, y), com x em
 * coordenadas do buffer local (as linhas fantasma garantem x-1 >= 0 e
 * x+1 <= linhas). O modulo horizontal implementa o toro nas colunas.
 */
static int contar_vizinhos(const unsigned char *tabuleiro, int x, int y) {
    int vizinhos = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue; /* a propria celula nao e vizinha de si mesma */
            }

            int vizinho_y = (y + j + largura) % largura;

            vizinhos += tabuleiro[(size_t)(x + i) * largura + vizinho_y];
        }
    }

    return vizinhos;
}

/*
 * FASE 1: calcula a proxima geracao das linhas [1, 1 + k) do buffer local e
 * devolve em *mudancas quantas celulas mudaram de estado (insumo da Fase 2).
 * O buffer tem k + 2 linhas: fantasma em cima (linha 0) e embaixo (k + 1).
 */
static void calcular_linhas_locais(const unsigned char *atual,
                                   unsigned char *proximo, int k,
                                   long *mudancas) {
    long m = 0;

    for (int i = 1; i <= k; i++) {
        for (int j = 0; j < largura; j++) {
            size_t pos = (size_t)i * largura + j;
            int vizinhos = contar_vizinhos(atual, i, j);
            int viva = atual[pos];

            /* Regras de Conway: uma celula viva sobrevive com 2 ou 3
               vizinhos vivos; uma celula morta nasce com exatamente 3. */
            proximo[pos] = (viva && (vizinhos == 2 || vizinhos == 3)) ||
                           (!viva && vizinhos == 3);

            m += proximo[pos] != viva;
        }
    }

    *mudancas = m;
}

/*
 * FASE 3: troca as linhas fantasma com os vizinhos do toro. Cada processo
 * envia a primeira linha ao vizinho de cima e a ultima ao de baixo e, no
 * mesmo ato, recebe as deles. Duas chamadas MPI_Sendrecv (um sentido cada);
 * como o padrao de mensagens e o mesmo em todos os processos, ninguem fica
 * preso esperando.
 */
static void trocar_fantasma(unsigned char *atual, int k, int rank, int proc_n) {
    int cima = (rank - 1 + proc_n) % proc_n;
    int baixo = (rank + 1) % proc_n;

    MPI_Sendrecv(&atual[1 * largura], largura, MPI_UNSIGNED_CHAR, cima,
                 TAG_SENTIDO_CIMA, &atual[(size_t)(k + 1) * largura], largura,
                 MPI_UNSIGNED_CHAR, baixo, TAG_SENTIDO_CIMA, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(&atual[(size_t)k * largura], largura, MPI_UNSIGNED_CHAR,
                 baixo, TAG_SENTIDO_BAIXO, &atual[0], largura,
                 MPI_UNSIGNED_CHAR, cima, TAG_SENTIDO_BAIXO, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
}

/* Hash FNV-1a de 64 bits: compara o resultado final com o da versao
   sequencial sem gravar o tabuleiro inteiro em disco. Como o hash depende da
   ordem dos bytes, o rank 0 recebe as fatias em ordem (MPI_Gatherv) e calcula
   o hash do tabuleiro inteiro. */
static unsigned long long checksum_global(const unsigned char *local, int k,
                                          int rank, int proc_n) {
    int *contagens = NULL;
    int *deslocamentos = NULL;
    unsigned char *tabuleiro = NULL;
    unsigned long long hash = 0;

    if (rank == 0) {
        contagens = malloc(proc_n * sizeof(int));
        deslocamentos = malloc(proc_n * sizeof(int));
        tabuleiro = malloc((size_t)altura * largura);
        if (contagens == NULL || deslocamentos == NULL || tabuleiro == NULL) {
            fprintf(stderr, "Erro ao alocar o checksum.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < proc_n; r++) {
            int ini = (int)((long)altura * r / proc_n);
            int fim = (int)((long)altura * (r + 1) / proc_n);
            contagens[r] = (fim - ini) * largura;
            deslocamentos[r] = ini * largura;
        }
    }

    MPI_Gatherv(local + largura, k * largura, MPI_UNSIGNED_CHAR, tabuleiro,
                contagens, deslocamentos, MPI_UNSIGNED_CHAR, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        hash = 1469598103934665603ULL;
        for (size_t i = 0; i < (size_t)altura * largura; i++) {
            hash ^= tabuleiro[i];
            hash *= 1099511628211ULL;
        }
        free(contagens);
        free(deslocamentos);
        free(tabuleiro);
    }

    return hash;
}

int main(int argc, char *argv[]) {
    int my_rank, proc_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_n);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--altura") == 0 && i + 1 < argc) {
            altura = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--largura") == 0 && i + 1 < argc) {
            largura = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--geracoes") == 0 && i + 1 < argc) {
            geracoes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-print") == 0) {
            silent = 1;
        } else if (strcmp(argv[i], "--estatisticas") == 0) {
            estatisticas = 1;
        }
    }

    /* O anel de troca de linhas fantasma precisa de ao menos 1 linha por
       processo: com np > altura sobram ranks com 0 linhas e o vizinho
       acabaria lendo memoria nao inicializada. */
    if (proc_n > altura) {
        if (my_rank == 0) {
            fprintf(stderr, "Exige np <= altura (1 linha por processo).\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* Fatia do processo: [inicio, fim) do toro, quase 1/np de linhas (o resto
       da divisao inteira sobra para os primeiros processos). */
    int inicio = (int)((long)altura * my_rank / proc_n);
    int fim = (int)((long)altura * (my_rank + 1) / proc_n);
    int k = fim - inicio; /* linhas locais */

    /* Buffer local: k linhas de trabalho + 1 fantasma em cima + 1 embaixo. */
    unsigned char *atual = alocar_tabuleiro(k + 2);
    unsigned char *proximo = alocar_tabuleiro(k + 2);

    srand(42);
    /* Fora da regiao cronometrada, mas medido a parte: cada processo percorre
       o tabuleiro INTEIRO (O(altura x largura) chamadas de rand) para gerar
       a propria fatia -- custo de inicializacao constante, independente de np. */
    double tempo_init = MPI_Wtime();
    gerar_fatia(atual, inicio, fim);
    tempo_init = MPI_Wtime() - tempo_init;

    MPI_Barrier(MPI_COMM_WORLD); /* todos prontos: comeca a medicao */
    double starttime = MPI_Wtime();

    long mudancas_locais = 0;
    long mudancas_globais = 1;
    double tempo_troca = 0.0, tempo_calculo = 0.0, tempo_coletiva = 0.0;
    int geracoes_executadas = 0;

    for (int geracao = 0; geracao < geracoes; geracao++) {
        double t;

        /* FASE 3 (do ciclo anterior) / FASE 0 (deste): linhas fantasma. */
        t = MPI_Wtime();
        trocar_fantasma(atual, k, my_rank, proc_n);
        tempo_troca += MPI_Wtime() - t;

        /* FASE 1: processamento local. */
        t = MPI_Wtime();
        calcular_linhas_locais(atual, proximo, k, &mudancas_locais);
        tempo_calculo += MPI_Wtime() - t;

        /* FASE 2: verificacao distribuida (comunicacao coletiva). */
        t = MPI_Wtime();
        MPI_Allreduce(&mudancas_locais, &mudancas_globais, 1, MPI_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        tempo_coletiva += MPI_Wtime() - t;

        unsigned char *temporario = atual;
        atual = proximo;
        proximo = temporario;
        geracoes_executadas = geracao + 1;

        if (mudancas_globais == 0) {
            break; /* tabuleiro estavel: o trabalho esta pronto */
        }
    }

    double stoptime = MPI_Wtime();

    /* Fora da medicao: tempo de cada fase de cada processo, para analisar o
       balanceamento de carga. O rank 0 recebe tudo e imprime em ordem. */
    double meus_tempos[4] = {tempo_calculo, tempo_troca, tempo_coletiva,
                             tempo_init};
    double *tempos = NULL;
    if (my_rank == 0) {
        tempos = malloc(4 * proc_n * sizeof(double));
        if (tempos == NULL) {
            fprintf(stderr, "Erro ao alocar as estatisticas.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Gather(meus_tempos, 4, MPI_DOUBLE, tempos, 4, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    unsigned long long hash = checksum_global(atual, k, my_rank, proc_n);

    if (my_rank == 0) {
        if (silent) {
            printf("%.6f %016llx\n", stoptime - starttime, hash);
        } else {
            printf("Tempo de execucao: %3.2f segundos\n",
                   stoptime - starttime);
            printf("Checksum do tabuleiro final: %016llx\n", hash);
            printf("Geracoes executadas: %d de %d", geracoes_executadas,
                   geracoes);
            if (mudancas_globais == 0 && geracoes_executadas < geracoes) {
                printf(" (estavel)");
            }
            printf("\n");
        }
        if (estatisticas) {
            for (int r = 0; r < proc_n; r++) {
                printf("# rank %d calc %.6f troca %.6f coletiva %.6f "
                       "init %.6f\n", r,
                       tempos[4 * r], tempos[4 * r + 1], tempos[4 * r + 2],
                       tempos[4 * r + 3]);
            }
        }
        free(tempos);
    }

    free(atual);
    free(proximo);

    MPI_Finalize();
    return 0;
}
