# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""
Analise dos resultados do T4: le ../resultados/brutos-*.csv (gerado por
benchmark.py no LAD) e produz

  ../relatorio/fig_forte.pdf        barras do speed-up forte (MPI puro x
  ../slides/fig_forte.png           hibrida) com o ideal S = p
  ../relatorio/fig_fraca.pdf        o mesmo para a escalabilidade fraca
  ../slides/fig_fraca.png           (sem HT)
  ../relatorio/fig_eficiencia.pdf   eficiencia (%): painel forte | painel fraca
  ../slides/fig_eficiencia.png
  ../relatorio/tabelas.tex          tabelas da pagina 2 do relatorio
  ../resultados/resumo.md           todos os numeros, para conferencia

Definicoes (unidade = 1 processo MPI na versao pura, 1 thread OpenMP na
hibrida; 4 processos x t threads = 4t unidades):

  forte   tabuleiro fixo 4000x4000; S = T_seq(4000^2) / T_p, E = 100 S / p.
  fraca   fatia constante de 125 x 4000 celulas por unidade (o tabuleiro
          cresce com p); S = T_seq(tabuleiro grande) / T_p, E = 100 S / p.
          O sequencial do tabuleiro grande e medido (experimento fraca_seq),
          nao extrapolado; como a referencia e o sequencial do MESMO trabalho
          total, o ideal de S vale p nas duas escalas.
          SEM HT: nem 64 processos, nem 16 threads/no -- com HT o trabalho
          por nucleo fisico dobra e a fraca deixa de comparar tamanhos
          equivalentes.

Uso: uv run analise.py          (dados reais)
     uv run analise.py --mock   (numeros inventados, para aprovar os graficos;
                                 salva em ../slides/fig_*_mock.png)
"""

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "resultados")
REL = os.path.join(AQUI, "..", "relatorio")
SLI = os.path.join(AQUI, "..", "slides")

# Paleta do T3: a cor segue a versao.
COR_MPI, COR_HIB = "#2a78d6", "#eb6834"
ROTULO_MPI, ROTULO_HIB = "MPI puro", "híbrida MPI/OpenMP"
TEXTO, TEXTO_2, GRADE, IDEAL = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": TEXTO_2, "axes.labelcolor": TEXTO,
    "axes.linewidth": 0.6, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 200,
})

FATIA = 125  # linhas por unidade na fraca (igual ao benchmark.py)
UNIDADES = [4, 8, 16, 32, 64]  # 64 so existe na forte (HT)
SEM_HT = [u for u in UNIDADES if u != 64]

SERIES_FORTE = [("forte_mpi", COR_MPI, ROTULO_MPI), ("hib", COR_HIB, ROTULO_HIB)]
SERIES_FRACA = [("fraca_mpi", COR_MPI, ROTULO_MPI),
                ("fraca_hib", COR_HIB, ROTULO_HIB)]

TITULO_BASE = (" — Jogo da Vida de Conway, 4 nós da atlântica "
               "(4000×4000×75 forte; 125×4000 por unidade fraca)")


def ler(padrao):
    linhas = []
    for caminho in sorted(glob.glob(os.path.join(RES, padrao))):
        with open(caminho) as f:
            linhas += list(csv.DictReader(f))
    return linhas


def br(x, casas=2):
    return "--" if x is None else f"{x:.{casas}f}".replace(".", ",")


# ----------------------------------------------------------------- dados
def dados_reais():
    tempos = defaultdict(list)
    for r in ler("brutos-*.csv"):
        chave = (r["experimento"], r["versao"], int(r["processos"]),
                 int(r["threads"]), int(r["altura"]))
        tempos[chave].append(float(r["tempo_s"]))

    def medida(exp, versao, p, threads, altura):
        t = tempos.get((exp, versao, p, threads, altura))
        return statistics.mean(t) if t else None

    d = {"seq": medida("seq", "seq", 1, 1, 4000)}
    for p in UNIDADES:
        d["forte_mpi", p] = medida("forte", "mpi", p, 1, 4000)
    for p in SEM_HT:
        # 64 nunca e medido na fraca (sem HT).
        d["fraca_seq", p] = medida("fraca_seq", "seq", p, 1, p * FATIA)
        d["fraca_mpi", p] = medida("fraca", "mpi", p, 1, p * FATIA)
    for t in (4, 8, 16):
        d["hib", 4 * t] = medida("hibrida", "mpi_omp", 4, t, 4000)
    for t in (4, 8):
        d["fraca_hib", 4 * t] = medida("fraca", "mpi_omp", 4, t,
                                       4 * t * FATIA)
    return d


def dados_mock():
    """Numeros plausiveis, so para aprovar o desenho dos graficos.

    Perfil inventado: ganho saturando por comunicacao (troca de halo +
    Allreduce por geracao), queda clara com HT (64 unidades em 32 nucleos
    fisicos) e eficiencia da fraca melhor que a da forte nos primeiros
    pontos (na forte o tabuleiro per-process encolhe e a comunicacao pesa).
    """
    d = {"seq": 58.2}  # 8.19 s em 1500^2 escalado por 4000^2 / 1500^2
    for p, t in ((4, 12.2), (8, 6.2), (16, 3.25), (32, 1.80), (64, 1.75)):
        d["forte_mpi", p] = t
    for u, t in ((16, 3.40), (32, 1.90), (64, 1.85)):
        d["hib", u] = t
    for p in SEM_HT:
        d["fraca_seq", p] = p * 58.2 / 32  # mesmo tamanho por unidade
        d["fraca_mpi", p] = {4: 1.95, 8: 1.98, 16: 2.02, 32: 2.10}[p]
    for u, t in ((16, 2.10), (32, 2.20)):
        d["fraca_hib", u] = t
    return d


# ------------------------------------------------------------- metricas
# A chave metrica e sempre (serie, unidades): a hibrida com t threads/no
# entra como 4t unidades, igual ao MPI puro com 4t processos.
def metricas(d):
    m = {}

    def adiciona(serie, u, ref):
        t = d.get((serie, u))
        if t and ref:
            m[serie, u] = dict(t=t, s=ref / t, e=100 * ref / t / u)

    for u in UNIDADES:
        adiciona("forte_mpi", u, d["seq"])
        adiciona("fraca_mpi", u, d.get(("fraca_seq", u)))
    for u in (16, 32, 64):
        adiciona("hib", u, d["seq"])
    for u in (16, 32):
        adiciona("fraca_hib", u, d.get(("fraca_seq", u)))
    return m


# -------------------------------------------------------------- graficos
def barras_series(ax, m, series, unidades, campo):
    """Barras pareadas por grupo de unidades (MPI puro a esquerda, hibrida a
    direita); grupo sem a hibrida fica so com a barra do MPI puro."""
    larg = 0.38
    for i, (chave, cor, rotulo) in enumerate(series):
        desloc = (i - len(series) / 2 + 0.5) * larg
        vals = [(g, m[chave, u][campo]) for g, u in enumerate(unidades)
                if (chave, u) in m]
        ax.bar([g + desloc for g, _ in vals], [v for _, v in vals],
               width=larg * 0.92, color=cor, label=rotulo, zorder=3)


def sombrear_ht(ax, unidades):
    if 64 in unidades:
        ax.axvspan(len(unidades) - 1.5, len(unidades) - 0.5,
                   color="#f3f2ef", zorder=0)
        y = ax.get_ylim()[1]
        ax.text(len(unidades) - 1, y * 0.95, "HT", ha="right", color=TEXTO_2,
                fontsize=7)


def fig_speedup(m, series, unidades, maximo, titulo):
    """Barras de speed-up de uma escala, com a linha ideal S = p."""
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    barras_series(ax, m, series, unidades, "s")
    ax.plot(range(len(unidades)), unidades, color=IDEAL, ls="--", lw=1.2,
            zorder=4)
    ax.set_ylim(0, maximo)
    sombrear_ht(ax, unidades)
    ax.set_xticks(range(len(unidades)), [str(u) for u in unidades])
    ax.set_xlabel("Unidades de execução (processo MPI na pura, thread "
                  "OpenMP na híbrida)")
    ax.set_ylabel("Speed-up")
    ax.yaxis.grid(True, color=GRADE, lw=0.6, zorder=0)
    ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(titulo, fontsize=9, color=TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_eficiencia(m):
    """Eficiencia (%): painel da forte | painel da fraca, com o ideal 100%."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.2), sharey=True)
    for ax, series, unidades, rot in ((a1, SERIES_FORTE, UNIDADES, "forte"),
                                      (a2, SERIES_FRACA, SEM_HT, "fraca")):
        barras_series(ax, m, series, unidades, "e")
        ax.axhline(100, color=IDEAL, ls="--", lw=1.2, zorder=4)
        ax.set_ylim(0, 108)
        sombrear_ht(ax, unidades)
        ax.set_xticks(range(len(unidades)), [str(u) for u in unidades])
        ax.set_xlabel("Unidades")
        ax.set_title(f"Escalabilidade {rot}" +
                     (" (sem HT)" if rot == "fraca" else ""),
                     fontsize=8, loc="left", color=TEXTO)
        ax.yaxis.grid(True, color=GRADE, lw=0.6, zorder=0)
    a1.set_ylabel("Eficiência (%)")
    alc, rot = a1.get_legend_handles_labels()
    fig.legend(alc, rot, loc="upper right", bbox_to_anchor=(0.99, 1.02),
               ncol=2, fontsize=7)
    fig.suptitle("Eficiência — Jogo da Vida de Conway, 4 nós da atlântica",
                 fontsize=9, color=TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def salvar(fig, nome):
    fig.savefig(os.path.join(REL, f"{nome}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(SLI, f"{nome}.png"), bbox_inches="tight",
                dpi=220)
    plt.close(fig)


# --------------------------------------------------------------- tabelas
def tabela(nome, linhas, cabecalho):
    l = [r"\begin{tabular}{@{}" + "c" * len(cabecalho) + "@{}}", r"\toprule",
         " & ".join(cabecalho) + r" \\", r"\midrule", *linhas, r"\bottomrule",
         r"\end{tabular}"]
    return f"\\newcommand{{\\{nome}}}{{%\n" + "\n".join(l) + "}\n"


def linha_tabela(m, serie, u, nome_u):
    v = m.get((serie, u))
    if not v:
        return None
    return (f"{nome_u} & {br(v['t'], 3)} & {br(v['s'], 1)} & {br(v['e'], 0)}"
            r" \\")


def gerar_tabelas(m):
    forte = [l for u in UNIDADES
             if (l := linha_tabela(m, "forte_mpi", u, str(u)))]
    forte += [l for u in (16, 32, 64)
              if (l := linha_tabela(m, "hib", u, f"4×{u // 4}"))]
    fraca = [l for u in SEM_HT
             if (l := linha_tabela(m, "fraca_mpi", u, str(u)))]
    fraca += [l for u in (16, 32)
              if (l := linha_tabela(m, "fraca_hib", u, f"4×{u // 4}"))]
    conteudo = "% Gerado por benchmark/analise.py -- nao editar a mao.\n"
    conteudo += tabela("TabForte", forte,
                       ["Unidades", "Tempo (s)", "S", "E (%)"])
    conteudo += tabela("TabFraca", fraca,
                       ["Unidades", "Tempo (s)", "S", "E (%)"])
    with open(os.path.join(REL, "tabelas.tex"), "w") as f:
        f.write(conteudo)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="numeros inventados, so para aprovar os graficos")
    a = ap.parse_args()

    d = dados_mock() if a.mock else dados_reais()
    m = metricas(d)

    if a.mock:
        os.makedirs(SLI, exist_ok=True)
        fig_speedup(m, SERIES_FORTE, UNIDADES, 70,
                    "Escalabilidade forte" + TITULO_BASE).savefig(
            os.path.join(SLI, "fig_forte_mock.png"), bbox_inches="tight")
        fig_speedup(m, SERIES_FRACA, SEM_HT, 40,
                    "Escalabilidade fraca (sem HT)" +
                    TITULO_BASE).savefig(
            os.path.join(SLI, "fig_fraca_mock.png"), bbox_inches="tight")
        fig_eficiencia(m).savefig(
            os.path.join(SLI, "fig_eficiencia_mock.png"), bbox_inches="tight")
        print("Graficos mockados em ../slides/fig_{forte,fraca,"
              "eficiencia}_mock.png")
    else:
        faltando = [k for k, v in d.items() if v is None]
        assert not faltando, f"faltam medicoes: {faltando}"
        os.makedirs(SLI, exist_ok=True)
        salvar(fig_speedup(m, SERIES_FORTE, UNIDADES, 70,
                           "Escalabilidade forte" + TITULO_BASE), "fig_forte")
        salvar(fig_speedup(m, SERIES_FRACA, SEM_HT, 40,
                           "Escalabilidade fraca (sem HT)" + TITULO_BASE),
               "fig_fraca")
        salvar(fig_eficiencia(m), "fig_eficiencia")
        gerar_tabelas(m)
        resumo(m)


def resumo(m):
    with open(os.path.join(RES, "resumo.md"), "w") as f:
        f.write("# Resumo (gerado por analise.py)\n\n")
        f.write("| série | unidades | tempo (s) | S | E% |\n"
                "|---|---|---|---|---|\n")
        for (serie, u), v in sorted(m.items()):
            f.write(f"| {serie} | {u} | {v['t']:.3f} | {v['s']:.1f} | "
                    f"{v['e']:.0f} |\n")


if __name__ == "__main__":
    main()
