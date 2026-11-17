#!/usr/bin/env python3
"""
Benchmark do T4: Jogo da Vida com MPI no modelo fases paralelas.

Cada execucao vira uma linha em ../resultados/brutos-<job>.csv, gravada assim
que termina, para que um job interrompido nao perca o que ja mediu. Os tempos
das fases (calculo, troca de linhas fantasma, coletiva) de cada processo na
primeira repeticao de cada configuracao vao para
../resultados/processos-<job>.csv. A analise e feita fora do cluster.

Experimentos:
  seq      versao sequencial em 1 core (linha de base)
  forte    MPI puro em 4 nos com 4, 8, 16, 32 e 64 processos (64 = HT)
  hibrida  MPI + OpenMP: 4 processos (1 por no), com 4, 8 e 16 threads
           por no (16 = HT) -- comparacao direta com MPI puro de mesmo
           numero total de nucleos (16/32/64)
  fraca    escalabilidade fraca: fatia constante de FATIA linhas x 1500 por
           unidade (processo ou thread), tabuleiro cresce com o numero de
           unidades. Sequencial medido no tabuleiro grande de cada ponto
           (referencia). Sem HT: nem 64 processos, nem 16 threads.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

ALTURA = 1500
LARGURA = 1500
GERACOES = 75
TIMEOUT = 2 * 3600

AQUI = os.path.dirname(os.path.abspath(__file__))
RESULTADOS = os.path.join(AQUI, "..", "resultados")
JOB = os.environ.get("SLURM_JOB_ID", "local")
BRUTOS = os.path.join(RESULTADOS, f"brutos-{JOB}.csv")
PROCESSOS = os.path.join(RESULTADOS, f"processos-{JOB}.csv")

CAMPOS_BRUTOS = ["experimento", "versao", "processos", "threads", "nos",
                 "altura", "largura", "geracoes", "repeticao", "tempo_s", "ok"]
CAMPOS_PROC = ["experimento", "versao", "processos", "threads", "rank",
               "calc_s", "troca_s", "coletiva_s"]


def anexar(caminho, campos, linhas):
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        if novo:
            w.writeheader()
        w.writerows(linhas)


def executar(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                       cwd=AQUI)
    linhas = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or not linhas:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    tempo, ok = linhas[0].split()[:2]
    procs = []
    for l in linhas[1:]:
        # "# rank R calc X troca Y coletiva Z"
        p = l.split()
        if p[0] == "#" and p[1] == "rank":
            procs.append(dict(rank=p[2], calc_s=p[4], troca_s=p[6],
                              coletiva_s=p[8]))
    return float(tempo), ok, procs


def medir(experimento, versao, cmd, processos, threads, reps):
    tempos = []
    for rep in range(reps):
        tempo, ok, procs = executar(cmd)
        tempos.append(tempo)
        anexar(BRUTOS, CAMPOS_BRUTOS, [dict(
            experimento=experimento, versao=versao, processos=processos,
            threads=threads, nos=NOS, altura=ALTURA, largura=LARGURA,
            geracoes=GERACOES, repeticao=rep, tempo_s=f"{tempo:.6f}", ok=ok)])
        if rep == 0 and procs:
            anexar(PROCESSOS, CAMPOS_PROC, [dict(
                experimento=experimento, versao=versao, processos=processos,
                threads=threads, **p) for p in procs])
    print(f"  {experimento} {versao} p={processos} t={threads}: "
          f"{sum(tempos) / len(tempos):.3f}s (n={len(tempos)}) "
          f"[{', '.join(f'{t:.2f}' for t in tempos)}]", flush=True)


NOS = 4
GEOM = ["--altura", str(ALTURA), "--largura", str(LARGURA), "--geracoes",
        str(GERACOES), "--no-print"]

# Escalabilidade fraca: cada unidade (processo ou thread) fica sempre com
# FATIA x LARGURA celulas; o tabuleiro cresce com o numero de unidades.
FATIA = 47  # 1500 / 32 arredondado: com 32 unidades o tabuleiro e 1504x1500


def cmd_fraca(unidades):
    altura = unidades * FATIA
    return ["--altura", str(altura), "--largura", str(LARGURA), "--geracoes",
            str(GERACOES), "--no-print"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experimentos", nargs="+",
                    choices=["seq", "forte", "hibrida", "fraca"])
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(RESULTADOS, exist_ok=True)
    inicio = time.time()

    for exp in a.experimentos:
        print(f"== {exp}", flush=True)
        if exp == "seq":
            medir("seq", "seq",
                  ["srun", "-N", "1", "-n", "1", "./conway_seq", *GEOM],
                  1, 1, a.reps)
        elif exp == "forte":
            for p in [4, 8, 16, 32, 64]:
                medir("forte", "mpi",
                      ["srun", "-N", str(NOS), "-n", str(p),
                       "./conway_fases_mpi", "--estatisticas", *GEOM],
                      p, 1, a.reps)
        elif exp == "hibrida":
            for t in [4, 8, 16]:
                medir("hibrida", "mpi_omp",
                      ["srun", "-N", str(NOS), "-n", "4",
                       "--cpus-per-task", str(t),
                       "./conway_fases_hibrido", "--threads", str(t),
                       "--estatisticas", *GEOM],
                      4, t, a.reps)
        elif exp == "fraca":
            # Sem HT: nem 64 processos, nem 16 threads por no.
            for u in [1, 4, 8, 16, 32]:
                medir("fraca_seq", "seq",
                      ["srun", "-N", "1", "-n", "1", "./conway_seq",
                       *cmd_fraca(u)], u, 1, a.reps)
            for u in [1, 4, 8, 16, 32]:
                medir("fraca", "mpi",
                      ["srun", "-N", str(NOS), "-n", str(u),
                       "./conway_fases_mpi", "--estatisticas", *cmd_fraca(u)],
                      u, 1, a.reps)
            for t in [4, 8]:
                u = 4 * t
                medir("fraca", "mpi_omp",
                      ["srun", "-N", str(NOS), "-n", "4",
                       "--cpus-per-task", str(t),
                       "./conway_fases_hibrido", "--threads", str(t),
                       "--estatisticas", *cmd_fraca(u)],
                      u, t, a.reps)

    print(f"== fim em {(time.time() - inicio) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
