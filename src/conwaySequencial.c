/*
 * Jogo da Vida de Conway -- versao sequencial (linha de base do T2)
 *
 * Mesmo problema do T1, com duas mudancas de representacao que a versao MPI
 * exige e que, por justica na comparacao, a linha de base tambem adota:
 *
 *  - celulas em unsigned char (1 byte) em vez de int (4 bytes), porque na
 *    versao MPI todo o tabuleiro trafega pela rede e o volume de dados
 *    enviado e proporcional ao tamanho da celula;
 *  - tabuleiro em um unico vetor contiguo (linha a linha) em vez de int**,
 *    para que um bloco de linhas consecutivas seja um trecho contiguo de
 *    memoria e possa ser enviado em uma unica mensagem, sem copia.
 *
 * O nucleo de calculo (contar_vizinhos e calcular_linhas) e identico, linha a
 * linha, ao usado pelos trabalhadores da versao MPI. Assim a diferenca de tempo
 * entre as duas versoes vem apenas do paralelismo e da comunicacao.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_LARGURA 1500
#define DEFAULT_ALTURA 1500
#define DEFAULT_GERACOES 75
#define DENSIDADE_INICIAL 20 /* % de celulas vivas no tabuleiro inicial */

static int largura = DEFAULT_LARGURA;
static int altura = DEFAULT_ALTURA;
static int geracoes = DEFAULT_GERACOES;
static int silent = 0;

static unsigned char *alocar_tabuleiro(int linhas) {
    unsigned char *tabuleiro = malloc((size_t)linhas * largura);
    if (tabuleiro == NULL) {
        fprintf(stderr, "Erro ao alocar o tabuleiro.\n");
        exit(1);
    }
    return tabuleiro;
}

/* Mesma sequencia de rand() do T1: o tabuleiro inicial e o mesmo. */
static void inicializar_tabuleiro(unsigned char *tabuleiro) {
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < largura; j++) {
            tabuleiro[(size_t)i * largura + j] =
                (rand() % 100 < DENSIDADE_INICIAL) ? 1 : 0;
        }
    }
}

/*
 * Conta as celulas vivas nas 8 posicoes adjacentes a (x, y) em um tabuleiro
 * de "linhas" linhas. O modulo nas duas dimensoes trata o tabuleiro como um
 * toro, exatamente como no T1 (16 operacoes de modulo por celula).
 */
static int contar_vizinhos(const unsigned char *tabuleiro, int linhas, int x,
                           int y) {
    int vizinhos = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue; /* a propria celula nao e vizinha de si mesma */
            }

            int vizinho_x = (x + i + linhas) % linhas;
            int vizinho_y = (y + j + largura) % largura;

            vizinhos += tabuleiro[(size_t)vizinho_x * largura + vizinho_y];
        }
    }

    return vizinhos;
}

/* Calcula a proxima geracao das linhas [inicio, fim) de "atual". */
static void calcular_linhas(const unsigned char *atual, unsigned char *proximo,
                            int linhas, int inicio, int fim) {
    for (int i = inicio; i < fim; i++) {
        for (int j = 0; j < largura; j++) {
            size_t pos = (size_t)i * largura + j;
            int vizinhos = contar_vizinhos(atual, linhas, i, j);
            int viva = atual[pos];

            /* Regras de Conway: uma celula viva sobrevive com 2 ou 3 vizinhos
               vivos; uma celula morta nasce com exatamente 3. */
            proximo[pos] = (viva && (vizinhos == 2 || vizinhos == 3)) ||
                           (!viva && vizinhos == 3);
        }
    }
}

/* Hash FNV-1a de 64 bits do tabuleiro: compara o resultado final com o da
   versao MPI sem precisar gravar o tabuleiro inteiro em disco. */
static unsigned long long checksum(const unsigned char *tabuleiro) {
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < (size_t)altura * largura; i++) {
        h ^= tabuleiro[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--altura") == 0 && i + 1 < argc) {
            altura = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--largura") == 0 && i + 1 < argc) {
            largura = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--geracoes") == 0 && i + 1 < argc) {
            geracoes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-print") == 0) {
            silent = 1;
        }
    }

    srand(42);

    unsigned char *atual = alocar_tabuleiro(altura);
    unsigned char *proximo = alocar_tabuleiro(altura);

    /* A inicializacao fica fora do cronometro, como no T1. */
    inicializar_tabuleiro(atual);

    double starttime = get_time();

    for (int geracao = 0; geracao < geracoes; geracao++) {
        calcular_linhas(atual, proximo, altura, 0, altura);

        /* Troca os papeis dos dois buffers em vez de copiar o tabuleiro. */
        unsigned char *temporario = atual;
        atual = proximo;
        proximo = temporario;
    }

    double stoptime = get_time();

    unsigned long long hash = checksum(atual);
    free(atual);
    free(proximo);

    if (silent) {
        printf("%.6f %016llx\n", stoptime - starttime, hash);
    } else {
        printf("Tempo de execucao: %3.2f segundos\n", stoptime - starttime);
        printf("Checksum do tabuleiro final: %016llx\n", hash);
    }

    return 0;
}
