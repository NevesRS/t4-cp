#!/usr/bin/env python3
"""
Visualizacao Interativa — Jogo da Vida de Conway Paralelizado
Modelo Fases Paralelas: MPI puro e MPI+OpenMP hibrido

Uso: python conway_vis.py
Requisitos: pip install pygame

Teclas:
  SPACE    — play / pausar
  ENTER    — avancar uma fase
  R        — reiniciar
  H        — alternar modo MPI puro / MPI+OpenMP
  +  /  -  — aumentar / diminuir velocidade
"""

import pygame
import sys
import math
import random

pygame.init()

# ═══════════════════════════════════════════════════════
#  PARAMETROS
# ═══════════════════════════════════════════════════════
WIN_W, WIN_H = 1400, 820
GRID_COLS  = 80
GRID_ROWS  = 80
CELL       = min(690/GRID_COLS, 590/GRID_ROWS)
#CELL       = 23 #690 x 460
N_RANKS    = 4
RPR        = GRID_ROWS // N_RANKS   # = 5 linhas por rank
MAX_GEN    = 30
DENSITY    = 0.20
SEED       = 42

GX   = 95
GY   = 92
RSEP = 6

GRID_W = GRID_COLS * CELL
GRID_H = GRID_ROWS * CELL + (N_RANKS - 1) * RSEP

SIDE_X = GX + GRID_W + 48
SIDE_W = WIN_W - SIDE_X - 10

CTRL_Y = WIN_H - 108

# ═══════════════════════════════════════════════════════
#  CORES
# ═══════════════════════════════════════════════════════
COL_BG    = ( 12,  12,  18)
COL_PBG   = ( 20,  20,  30)
COL_TEXT  = (210, 215, 230)
COL_DIM   = ( 90, 100, 120)
COL_WHITE = (255, 255, 255)
COL_BORD  = ( 40,  45,  65)

RANK_ALIVE = [(55, 115, 215), (55, 185,  85), (215, 125, 40), (160,  75, 215)]
RANK_DEAD  = [(16,  30,  55), (16,  50,  22), ( 58,  34, 10), ( 44,  18,  58)]
RANK_GHOST = [(100,155,255),  (100,220, 135), (255, 165, 80), (210, 120, 255)]

PHASE_COL = {
    'ghost':     (255, 200,  50),
    'compute':   ( 40, 210, 255),
    'allreduce': (255,  80,  80),
    'swap':      ( 90, 255, 150),
}
PHASE_LABEL = {
    'ghost':     'Fase 0 — Troca de Linhas Fantasma (MPI_Sendrecv)',
    'compute':   'Fase 1 — Processamento Local',
    'allreduce': 'Fase 2 — Verificacao Distribuida (MPI_Allreduce)',
    'swap':      'Troca de Buffers  (atual <-> proximo)',
}
PHASE_DESC = {
    'ghost': [
        'Cada rank envia sua 1a linha ao vizinho de cima',
        'e sua ultima linha ao vizinho de baixo.',
        'MPI_Sendrecv() — comunicacao bidirecional simultanea.',
        'O tabuleiro e um toro: rank 0 <-> rank N-1.',
    ],
    'compute': [
        'Cada rank calcula a proxima geracao das suas linhas.',
        'Linhas fantasma permitem tratar bordas sem modulo.',
        'Conta celulas que mudaram de estado.',
        '[Hibrido] Threads OpenMP dividem as linhas entre si.',
    ],
    'allreduce': [
        'Cada rank conta mudancas locais (mudancas_locais).',
        'MPI_Allreduce(SUM): reducao distribuida coletiva.',
        'Se Soma = 0 -> tabuleiro estavel -> encerra.',
        'Todos os ranks sabem o resultado simultaneamente.',
    ],
    'swap': [
        'Ponteiros atual<->proximo trocados em O(1).',
        'Sem copia de memoria: so troca de ponteiros.',
        'A geracao esta completa.',
        'Proximo ciclo comeca pela troca de fantasmas.',
    ],
}

# ═══════════════════════════════════════════════════════
#  FONTES E TELA
# ═══════════════════════════════════════════════════════
F_TITLE = pygame.font.SysFont('Segoe UI', 24, bold=True)
F_BIG   = pygame.font.SysFont('Segoe UI', 20, bold=True)
F_MED   = pygame.font.SysFont('Segoe UI', 16)
F_SML   = pygame.font.SysFont('Segoe UI', 14)
F_XSM   = pygame.font.SysFont('Segoe UI', 12)
F_MONO  = pygame.font.SysFont('Consolas', 14)

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption('Conway MPI/OpenMP')
clock  = pygame.time.Clock()

# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════
def lerp(a, b, t):
    return a + (b - a) * t

def lerp_color(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ease_out(t):
    return 1 - (1 - t) ** 2

def rank_range(r):
    return GRID_ROWS * r // N_RANKS, GRID_ROWS * (r + 1) // N_RANKS

def rank_of(row):
    for r in range(N_RANKS):
        a, b = rank_range(r)
        if a <= row < b:
            return r
    return N_RANKS - 1

def rank_top(r):
    return GY + r * (RPR * CELL + RSEP)

def draw_txt(surf, text, font, color, x, y, align='left'):
    img = font.render(str(text), True, color[:3] if len(color) > 3 else color)
    if   align == 'center': x -= img.get_width() // 2
    elif align == 'right':  x -= img.get_width()
    surf.blit(img, (x, y))
    return img.get_size()

def draw_rrect(surf, color, rect, radius=6, bw=0, bc=None):
    pygame.draw.rect(surf, color[:3] if len(color) > 3 else color,
                     rect, border_radius=radius)
    if bw and bc:
        pygame.draw.rect(surf, bc, rect, bw, border_radius=radius)

def alpha_rect(surf, color, rect, alpha, radius=0):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    r3 = color[:3]
    if radius:
        pygame.draw.rect(s, (*r3, alpha), (0, 0, rect[2], rect[3]),
                         border_radius=radius)
    else:
        s.fill((*r3, alpha))
    surf.blit(s, (rect[0], rect[1]))

# ═══════════════════════════════════════════════════════
#  SIMULACAO
# ═══════════════════════════════════════════════════════
def generate_grid():
    random.seed(SEED)
    return [[1 if random.random() < DENSITY else 0
             for _ in range(GRID_COLS)]
            for _ in range(GRID_ROWS)]

def compute_next(grid):
    ng      = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    changes = [0] * N_RANKS
    for i in range(GRID_ROWS):
        r = rank_of(i)
        for j in range(GRID_COLS):
            n = sum(
                grid[(i + di) % GRID_ROWS][(j + dj) % GRID_COLS]
                for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)
            )
            alive      = grid[i][j]
            new_val    = int((alive and n in (2, 3)) or (not alive and n == 3))
            ng[i][j]   = new_val
            changes[r] += (new_val != alive)
    return ng, changes

# ═══════════════════════════════════════════════════════
#  ESTADO DA SIMULACAO
# ═══════════════════════════════════════════════════════
class SimState:
    PHASES = ['ghost', 'compute', 'allreduce', 'swap']
    PH_DUR = {'ghost': 2.2, 'compute': 2.8, 'allreduce': 2.4, 'swap': 0.9}

    def __init__(self):
        self.reset()

    def reset(self):
        self.grid      = generate_grid()
        self.next_grid = None
        self.gen       = 0
        self.phase     = 'ghost'
        self.ph_timer  = 0.0
        self.ph_prog   = 0.0
        self.changes   = [0] * N_RANKS
        self.total_ch  = 0
        self.stable    = False
        self.running   = False
        self.speed     = 1.0
        self.hybrid    = False
        self.n_threads = 4
        self.changed   = set()
        self.anim      = {}
        self._enter('ghost')

    def _enter(self, ph):
        self.phase    = ph
        self.ph_timer = 0.0
        self.ph_prog  = 0.0
        self.anim     = {}

        if ph == 'ghost':
            arrows = []
            for r in range(N_RANKS):
                ini, fim = rank_range(r)
                above = (r - 1) % N_RANKS
                below = (r + 1) % N_RANKS
                arrows.append({'from': r, 'to': below, 'dir': 'down',
                                'row': fim - 1, 'prog': 0.0,
                                'torus': r == N_RANKS - 1})
                arrows.append({'from': r, 'to': above, 'dir': 'up',
                                'row': ini, 'prog': 0.0,
                                'torus': r == 0})
            self.anim['arrows'] = arrows

        elif ph == 'compute':
            self.next_grid, self.changes = compute_next(self.grid)
            self.changed = {
                (i, j)
                for i in range(GRID_ROWS)
                for j in range(GRID_COLS)
                if self.next_grid[i][j] != self.grid[i][j]
            }
            self.anim['scan'] = [0.0] * N_RANKS

        elif ph == 'allreduce':
            self.total_ch = sum(self.changes)
            self.anim['particles'] = [
                {'rank': r, 'val': self.changes[r], 'prog': 0.0,
                 'delay': r * 0.18}
                for r in range(N_RANKS)
            ]
            self.anim['show_sum'] = False

        elif ph == 'swap':
            if self.next_grid is not None:
                self.grid      = self.next_grid
                self.next_grid = None
            self.gen += 1
            if self.total_ch == 0 or self.gen >= MAX_GEN:
                self.stable = True

    def step_phase(self):
        if self.stable:
            return
        idx = self.PHASES.index(self.phase)
        self._enter(self.PHASES[(idx + 1) % len(self.PHASES)])

    def update(self, dt):
        if not self.running or self.stable:
            return
        self.ph_timer += dt * self.speed
        dur            = self.PH_DUR[self.phase]
        self.ph_prog   = clamp(self.ph_timer / dur, 0.0, 1.0)

        if self.phase == 'ghost':
            for arrow in self.anim['arrows']:
                arrow['prog'] = clamp(ease_out(self.ph_prog * 1.5), 0.0, 1.0)

        elif self.phase == 'compute':
            if self.hybrid:
                band = RPR / self.n_threads
                for r in range(N_RANKS):
                    self.anim['scan'][r] = self.ph_prog * band
            else:
                for r in range(N_RANKS):
                    self.anim['scan'][r] = self.ph_prog * RPR

        elif self.phase == 'allreduce':
            for p in self.anim['particles']:
                local_t = clamp(
                    (self.ph_prog - p['delay']) / max(1.0 - p['delay'], 0.01),
                    0.0, 1.0
                )
                p['prog'] = ease_out(local_t)
            self.anim['show_sum'] = self.ph_prog > 0.72

        if self.ph_timer >= dur:
            idx = self.PHASES.index(self.phase)
            self._enter(self.PHASES[(idx + 1) % len(self.PHASES)])
            if self.stable:
                self.running = False


# ═══════════════════════════════════════════════════════
#  RENDERIZACAO: GRID
# ═══════════════════════════════════════════════════════
def draw_grid(surf, state, t_abs):
    for r in range(N_RANKS):
        ini, fim = rank_range(r)
        ty = rank_top(r)
        rw = GRID_COLS * CELL
        rh = RPR * CELL

        alpha_rect(surf, RANK_DEAD[r], (GX, ty, rw, rh), 200)

        for i in range(ini, fim):
            lrow = i - ini
            cy   = ty + lrow * CELL
            for j in range(GRID_COLS):
                cx    = GX + j * CELL
                alive = state.grid[i][j]
                color = RANK_ALIVE[r] if alive else RANK_DEAD[r]

                # Fase compute: exibe next_grid para linhas ja varridas
                if state.phase == 'compute' and state.next_grid is not None:
                    scan_pos = state.anim['scan'][r]
                    if state.hybrid:
                        n_th    = state.n_threads
                        band    = RPR / n_th
                        tidx    = lrow * n_th // RPR
                        in_band = lrow - tidx * band
                        if in_band <= scan_pos:
                            na    = state.next_grid[i][j]
                            color = RANK_ALIVE[r] if na else RANK_DEAD[r]
                            if (i, j) in state.changed:
                                color = lerp_color(color, COL_WHITE, 0.6)
                    else:
                        if lrow <= scan_pos:
                            na    = state.next_grid[i][j]
                            color = RANK_ALIVE[r] if na else RANK_DEAD[r]
                            if (i, j) in state.changed:
                                color = lerp_color(color, COL_WHITE, 0.6)

                # Fase ghost: pulsa linhas de borda
                if state.phase == 'ghost':
                    if lrow == 0 or lrow == RPR - 1:
                        pulse = abs(math.sin(t_abs * 4.5)) * 0.5 + 0.15
                        color = lerp_color(color, RANK_GHOST[r], pulse)

                # Fase swap: pisca celulas que mudaram
                if state.phase == 'swap' and (i, j) in state.changed:
                    flash = clamp(1 - state.ph_prog * 2.5, 0.0, 1.0) * 0.7
                    color = lerp_color(color, COL_WHITE, flash)

                pygame.draw.rect(surf, color, (cx + 1, cy + 1, CELL - 2, CELL - 2))

        # Borda do rank
        pygame.draw.rect(surf, RANK_ALIVE[r], (GX, ty, rw, rh), 2, border_radius=2)

        # Badge "Rank N" a esquerda
        bw, bh = 60, 22
        bx = GX - bw - 8
        by = ty + rh // 2 - bh // 2
        draw_rrect(surf, RANK_ALIVE[r], (bx, by, bw, bh), 4)
        draw_txt(surf, f'Rank {r}', F_SML, COL_WHITE, bx + bw // 2, by + 4, 'center')

        # Scan lines (fase compute)
        if state.phase == 'compute':
            if state.hybrid:
                n_th = state.n_threads
                band = RPR / n_th
                for tidx in range(n_th):
                    if tidx > 0:
                        div_y = ty + int(tidx * band * CELL)
                        s = pygame.Surface((rw, 1), pygame.SRCALPHA)
                        s.fill((255, 255, 255, 60))
                        surf.blit(s, (GX, div_y))
                        draw_txt(surf, f'T{tidx}', F_XSM, RANK_ALIVE[r],
                                 GX + 3, div_y + 2)
                    scan_abs = tidx * band + state.anim['scan'][r]
                    scan_y   = ty + int(clamp(scan_abs, 0, RPR) * CELL)
                    if scan_y <= ty + rh:
                        pygame.draw.line(surf, PHASE_COL['compute'],
                                         (GX, scan_y), (GX + rw, scan_y), 2)
                draw_txt(surf, 'T0', F_XSM, RANK_ALIVE[r], GX + 3, ty + 2)
            else:
                scan_y = ty + int(clamp(state.anim['scan'][r], 0, RPR) * CELL)
                if scan_y <= ty + rh:
                    pygame.draw.line(surf, PHASE_COL['compute'],
                                     (GX, scan_y), (GX + rw, scan_y), 2)

    # Separadores entre secoes de rank
    for r in range(N_RANKS - 1):
        sep_y = rank_top(r) + RPR * CELL
        pygame.draw.rect(surf, COL_BG, (GX, sep_y, GRID_W, RSEP))

    if state.phase == 'ghost':
        draw_ghost_arrows(surf, state)


def draw_ghost_arrows(surf, state):
    ax = GX + GRID_W + 12

    for arrow in state.anim.get('arrows', []):
        prog  = arrow['prog']
        if prog <= 0.01:
            continue
        r     = arrow['from']
        d     = arrow['dir']
        torus = arrow['torus']
        to_r  = arrow['to']
        col   = PHASE_COL['ghost']

        ini_f, fim_f = rank_range(r)
        src_local = arrow['row'] - ini_f
        src_y = rank_top(r) + src_local * CELL + CELL // 2

        if not torus:
            dst_y = rank_top(to_r) if d == 'down' else rank_top(to_r) + RPR * CELL
            cur_y = int(lerp(src_y, dst_y, prog))
            pygame.draw.line(surf, col, (ax, src_y), (ax, cur_y), 2)
            if prog > 0.65:
                s = 1 if d == 'down' else -1
                pygame.draw.polygon(surf, col, [
                    (ax, cur_y + s * 7),
                    (ax - 5, cur_y - s * 3),
                    (ax + 5, cur_y - s * 3),
                ])
        else:
            # Toro: linha tracejada saindo pela borda
            edge_y = (rank_top(r) + RPR * CELL + RSEP + 8) if d == 'down' \
                     else (rank_top(r) - 8)
            cur_y  = int(lerp(src_y, edge_y, min(prog * 2.0, 1.0)))
            total  = abs(cur_y - src_y)
            sign   = 1 if cur_y > src_y else -1
            for di in range(0, max(total, 1), 10):
                y0 = src_y + sign * di
                y1 = src_y + sign * min(di + 5, total)
                pygame.draw.line(surf, col, (ax, y0), (ax, y1), 2)
            if prog > 0.3:
                draw_txt(surf, '(toro)', F_XSM, col, ax + 6,
                         (src_y + cur_y) // 2 - 7)


# ═══════════════════════════════════════════════════════
#  RENDERIZACAO: PAINEL LATERAL
# ═══════════════════════════════════════════════════════
def draw_side_panel(surf, state, t_abs):
    pygame.draw.rect(surf, COL_PBG, (SIDE_X - 5, 0, SIDE_W + 15, WIN_H))
    cx = SIDE_X + SIDE_W // 2
    y  = 15

    draw_txt(surf, 'Modelo Fases Paralelas', F_BIG, COL_TEXT, cx, y, 'center')
    y += 34

    # Badge de modo
    if state.hybrid:
        ml  = f'MPI + OpenMP  |  {N_RANKS} ranks x {state.n_threads} threads'
        mc  = (255, 160, 50)
    else:
        ml  = f'MPI puro  |  {N_RANKS} processos'
        mc  = (100, 180, 255)
    alpha_rect(surf, mc, (SIDE_X, y, SIDE_W - 10, 26), 55, 5)
    draw_txt(surf, ml, F_SML, mc, cx, y + 6, 'center')
    y += 36

    # Barra de geracao
    draw_txt(surf, f'Geracao  {state.gen} / {MAX_GEN}', F_MED, COL_TEXT, SIDE_X, y)
    y += 20
    bw = SIDE_W - 10
    pygame.draw.rect(surf, COL_BORD, (SIDE_X, y, bw, 6), border_radius=3)
    filled = int(bw * state.gen / MAX_GEN)
    pygame.draw.rect(surf, (80, 200, 80), (SIDE_X, y, filled, 6), border_radius=3)
    y += 16

    pygame.draw.line(surf, COL_BORD, (SIDE_X, y), (SIDE_X + SIDE_W - 10, y))
    y += 10

    # Indicador de fase com barra de progresso
    phcol = PHASE_COL[state.phase]
    alpha_rect(surf, phcol, (SIDE_X, y, SIDE_W - 10, 38), 35, 6)
    pygame.draw.rect(surf, phcol, (SIDE_X, y, SIDE_W - 10, 38), 2, border_radius=6)
    pw = int((SIDE_W - 14) * state.ph_prog)
    if pw > 2:
        alpha_rect(surf, phcol, (SIDE_X + 2, y + 2, pw, 34), 25, 4)
    draw_txt(surf, PHASE_LABEL[state.phase], F_MED, phcol, SIDE_X + 10, y + 11)
    y += 46

    for line in PHASE_DESC[state.phase]:
        if line:
            draw_txt(surf, f'* {line}', F_SML, COL_DIM, SIDE_X + 5, y)
            y += 17
    y += 8

    pygame.draw.line(surf, COL_BORD, (SIDE_X, y), (SIDE_X + SIDE_W - 10, y))
    y += 10

    # Estatisticas por rank
    draw_txt(surf, 'Mudancas por Rank:', F_MED, COL_TEXT, SIDE_X, y)
    y += 22

    max_ch = max(state.changes) if max(state.changes) > 0 else 1
    for r in range(N_RANKS):
        rc = RANK_ALIVE[r]
        draw_rrect(surf, rc, (SIDE_X, y, 54, 20), 4)
        draw_txt(surf, f'Rank {r}', F_SML, COL_WHITE, SIDE_X + 27, y + 3, 'center')
        draw_txt(surf, f'{state.changes[r]:4d}', F_MONO, COL_TEXT, SIDE_X + 60, y + 2)
        draw_txt(surf, 'cel.', F_SML, COL_DIM, SIDE_X + 100, y + 3)
        bar_x  = SIDE_X + 130
        bar_w2 = SIDE_W - 140
        pygame.draw.rect(surf, COL_BORD, (bar_x, y + 5, bar_w2, 10), border_radius=3)
        if state.changes[r] > 0:
            fb = int(bar_w2 * state.changes[r] / max_ch)
            pygame.draw.rect(surf, rc, (bar_x, y + 5, fb, 10), border_radius=3)
        y += 26

    total_col = (90, 255, 150) if state.total_ch == 0 else COL_TEXT
    draw_txt(surf, f'Total: {state.total_ch} mudancas', F_MED, total_col, SIDE_X, y)
    y += 22

    pygame.draw.line(surf, COL_BORD, (SIDE_X, y), (SIDE_X + SIDE_W - 10, y))
    y += 10

    if   state.phase == 'allreduce':
        draw_allreduce_vis(surf, state, y)
    elif state.phase == 'ghost':
        draw_ghost_diagram(surf, state, y, t_abs)
    elif state.phase == 'compute':
        draw_compute_info(surf, state, y)
    elif state.phase == 'swap':
        draw_swap_message(surf, state, y)


def draw_allreduce_vis(surf, state, y):
    cx = SIDE_X + SIDE_W // 2
    draw_txt(surf, 'MPI_Allreduce(SUM)', F_BIG, PHASE_COL['allreduce'], cx, y, 'center')
    y += 30

    box_w = 54
    box_h = 38
    sp    = (SIDE_W - 10) / N_RANKS
    box_tops = []

    for r in range(N_RANKS):
        bx = int(SIDE_X + (r + 0.5) * sp - box_w // 2)
        by = y
        box_tops.append((bx, by))
        draw_rrect(surf, RANK_ALIVE[r], (bx, by, box_w, box_h), 4)
        draw_txt(surf, str(state.changes[r]), F_BIG, COL_WHITE,
                 bx + box_w // 2, by + 5, 'center')
        draw_txt(surf, f'R{r}', F_SML, COL_WHITE,
                 bx + box_w // 2, by + box_h + 4, 'center')

    dest_y = y + box_h + 55

    for p in state.anim.get('particles', []):
        if p['prog'] <= 0.01:
            continue
        r    = p['rank']
        bx_c = box_tops[r][0] + box_w // 2
        src_y2 = box_tops[r][1] + box_h

        cur_x = int(lerp(bx_c, cx, ease_out(p['prog'])))
        cur_y = int(lerp(src_y2, dest_y, ease_out(p['prog'])))

        if p['prog'] < 0.98:
            pygame.draw.line(surf, RANK_ALIVE[r], (bx_c, src_y2), (cur_x, cur_y), 2)
        pygame.draw.circle(surf, RANK_ALIVE[r], (cur_x, cur_y), 11)
        draw_txt(surf, str(p['val']), F_SML, COL_WHITE, cur_x, cur_y - 7, 'center')

    if state.anim.get('show_sum', False):
        result_col = (90, 255, 150) if state.total_ch == 0 else (255, 100, 100)
        draw_rrect(surf, result_col, (cx - 60, dest_y - 10, 120, 42), 6)
        draw_txt(surf, f'Soma = {state.total_ch}', F_BIG, COL_BG, cx, dest_y - 2, 'center')
        msg = 'ESTAVEL -> ENCERRA' if state.total_ch == 0 else 'CONTINUA ->'
        draw_txt(surf, msg, F_SML, result_col, cx, dest_y + 38, 'center')


def draw_ghost_diagram(surf, state, y, t_abs):
    cx = SIDE_X + SIDE_W // 2
    draw_txt(surf, 'Comunicacao MPI_Sendrecv', F_BIG, PHASE_COL['ghost'], cx, y, 'center')
    y += 28

    bh  = 28
    gap = 10
    bw  = SIDE_W - 60
    bx  = SIDE_X + 20
    ax2 = bx + bw + 8
    ax3 = ax2 + 12

    rank_ys = []
    for r in range(N_RANKS):
        ry2 = y + r * (bh + gap)
        rank_ys.append(ry2)
        draw_rrect(surf, RANK_ALIVE[r], (bx, ry2, bw, bh), 4)
        draw_txt(surf, f'Rank {r}', F_SML, COL_WHITE, bx + 10, ry2 + 7)
        ini2, fim2 = rank_range(r)
        draw_txt(surf, f'L{ini2}-{fim2-1}', F_XSM, COL_WHITE, bx + bw - 48, ry2 + 8)

    col = PHASE_COL['ghost']
    for r in range(N_RANKS - 1):
        y0 = rank_ys[r] + bh
        y1 = rank_ys[r + 1]
        # Flecha para baixo (rank r envia para r+1)
        pygame.draw.line(surf, col, (ax2, y0), (ax2, y1), 2)
        pygame.draw.polygon(surf, col, [(ax2, y1 + 5), (ax2-4, y1-3), (ax2+4, y1-3)])
        # Flecha para cima (rank r+1 envia para r)
        pygame.draw.line(surf, col, (ax3, y0), (ax3, y1), 2)
        pygame.draw.polygon(surf, col, [(ax3, y0 - 5), (ax3-4, y0+3), (ax3+4, y0+3)])

    # Indicador de toro
    toro_x = bx - 12
    mid_y  = (rank_ys[0] + rank_ys[-1] + bh) // 2
    draw_txt(surf, '<->', F_SML, col, toro_x, mid_y - 8, 'center')
    draw_txt(surf, 'toro', F_XSM, col, toro_x, mid_y + 8, 'center')

    y += N_RANKS * (bh + gap) + 15
    draw_txt(surf, '* Cada rank envia E recebe ao mesmo tempo.', F_SML, COL_DIM, SIDE_X + 5, y)
    y += 17
    draw_txt(surf, '* MPI_Sendrecv: nao-bloqueante, sem deadlock.', F_SML, COL_DIM, SIDE_X + 5, y)


def draw_compute_info(surf, state, y):
    draw_txt(surf, 'Regras de Conway:', F_BIG, PHASE_COL['compute'], SIDE_X, y)
    y += 26
    rules = [
        ('Celula VIVA + 2 ou 3 vizinhos vivos', '-> sobrevive'),
        ('Celula VIVA + < 2 ou > 3 vizinhos',   '-> morre'),
        ('Celula MORTA + exatamente 3 vizinhos', '-> nasce'),
    ]
    for cond, result in rules:
        draw_txt(surf, f'* {cond}', F_SML, COL_DIM,  SIDE_X + 5, y)
        draw_txt(surf, result,      F_SML, COL_TEXT, SIDE_X + 15, y + 15)
        y += 32

    if state.hybrid:
        y += 5
        cx = SIDE_X + SIDE_W // 2
        msg = f'{N_RANKS} ranks x {state.n_threads} threads = {N_RANKS * state.n_threads} unidades paralelas'
        draw_txt(surf, msg, F_MED, (255, 160, 50), cx, y, 'center')


def draw_swap_message(surf, state, y):
    cx    = SIDE_X + SIDE_W // 2
    swcol = PHASE_COL['swap']
    draw_txt(surf, 'atual  <->  proximo', F_BIG, swcol, cx, y + 10, 'center')
    y += 48
    if state.total_ch == 0:
        draw_txt(surf, 'Tabuleiro estavel!', F_BIG, (90, 255, 150), cx, y, 'center')
        draw_txt(surf, 'Simulacao encerrada.', F_MED, COL_DIM, cx, y + 28, 'center')
    elif state.gen >= MAX_GEN:
        draw_txt(surf, f'{MAX_GEN} geracoes concluidas', F_BIG, swcol, cx, y, 'center')
    else:
        draw_txt(surf, f'Geracao {state.gen} completa', F_BIG, swcol, cx, y, 'center')
        draw_txt(surf, f'Total mudancas: {state.total_ch}', F_MED, COL_TEXT, cx, y + 28, 'center')


# ═══════════════════════════════════════════════════════
#  RENDERIZACAO: CONTROLES
# ═══════════════════════════════════════════════════════
def draw_controls(surf, state):
    pygame.draw.rect(surf, (16, 16, 24), (0, CTRL_Y, WIN_W, WIN_H - CTRL_Y))
    pygame.draw.line(surf, COL_BORD, (0, CTRL_Y), (WIN_W, CTRL_Y))

    btn_y = CTRL_Y + 15

    # Play/Pause
    pp_col = (50, 140, 70) if not state.running else (140, 60, 60)
    pp_lbl = '>  Play' if not state.running else '||  Pausar'
    draw_rrect(surf, pp_col, (20, btn_y, 110, 36), 6)
    draw_txt(surf, pp_lbl, F_MED, COL_WHITE, 75, btn_y + 9, 'center')

    # Step
    draw_rrect(surf, (55, 60, 95), (142, btn_y, 95, 36), 6)
    draw_txt(surf, '>>| Step', F_MED, COL_TEXT, 189, btn_y + 9, 'center')

    # Reset
    draw_rrect(surf, (95, 50, 50), (248, btn_y, 95, 36), 6)
    draw_txt(surf, '<<  Reset', F_MED, COL_TEXT, 295, btn_y + 9, 'center')

    # Slider de velocidade
    sp_x, sp_y, sp_w = 365, btn_y + 20, 155
    draw_txt(surf, f'Velocidade: {state.speed:.1f}x', F_SML, COL_DIM, 365, btn_y + 2)
    pygame.draw.rect(surf, COL_BORD, (sp_x, sp_y, sp_w, 4), border_radius=2)
    t_speed = (state.speed - 0.3) / (3.0 - 0.3)
    kx = sp_x + int(t_speed * sp_w)
    pygame.draw.circle(surf, (140, 160, 255), (kx, sp_y + 2), 8)

    # Modo
    mode_lx = 545
    draw_txt(surf, 'Modo:', F_SML, COL_DIM, mode_lx, btn_y + 2)
    mc1 = (40, 80, 165) if not state.hybrid else (35, 40, 70)
    draw_rrect(surf, mc1, (mode_lx, btn_y + 20, 95, 20), 4)
    draw_txt(surf, 'MPI puro', F_SML,
             COL_WHITE if not state.hybrid else COL_DIM,
             mode_lx + 47, btn_y + 24, 'center')
    mc2 = (120, 70, 20) if state.hybrid else (35, 40, 70)
    draw_rrect(surf, mc2, (mode_lx + 100, btn_y + 20, 95, 20), 4)
    draw_txt(surf, 'MPI+OMP', F_SML,
             COL_WHITE if state.hybrid else COL_DIM,
             mode_lx + 147, btn_y + 24, 'center')

    # Threads (modo hibrido)
    if state.hybrid:
        draw_txt(surf, 'Threads:', F_SML, COL_DIM, 758, btn_y + 2)
        for i, nth in enumerate([2, 4, 8]):
            tx = 758 + i * 55
            tc = (120, 70, 20) if state.n_threads == nth else (40, 48, 80)
            draw_rrect(surf, tc, (tx, btn_y + 20, 50, 20), 4)
            draw_txt(surf, str(nth), F_SML, COL_WHITE, tx + 25, btn_y + 24, 'center')

    # Status a direita
    sx_right = WIN_W - 10
    if state.stable:
        msg = 'Estavel!' if state.total_ch == 0 else f'{MAX_GEN} geracoes'
        draw_txt(surf, msg, F_BIG, (90, 255, 150), sx_right, btn_y + 5, 'right')
    else:
        phcol = PHASE_COL[state.phase]
        draw_txt(surf, f'. {PHASE_LABEL[state.phase]}', F_SML, phcol,
                 sx_right, btn_y + 4, 'right')
        draw_txt(surf, f'Gen {state.gen}/{MAX_GEN}  |  Mudancas: {state.total_ch}',
                 F_SML, COL_DIM, sx_right, btn_y + 22, 'right')

    # Dicas de teclado
    draw_txt(surf,
             'SPACE=play/pausar   ENTER=step   R=reset   H=hibrido   +/-=velocidade',
             F_XSM, (50, 55, 85), WIN_W // 2, CTRL_Y + 82, 'center')


# ═══════════════════════════════════════════════════════
#  INTERACAO
# ═══════════════════════════════════════════════════════
speed_dragging = False

def handle_click(pos, state):
    mx, my = pos
    btn_y  = CTRL_Y + 15

    if pygame.Rect(20, btn_y, 110, 36).collidepoint(mx, my):
        if not state.stable:
            state.running = not state.running

    elif pygame.Rect(142, btn_y, 95, 36).collidepoint(mx, my):
        state.running = False
        state.step_phase()

    elif pygame.Rect(248, btn_y, 95, 36).collidepoint(mx, my):
        state.reset()

    # Slider
    sp_x, sp_y, sp_w = 365, btn_y + 20, 155
    if pygame.Rect(sp_x - 10, sp_y - 10, sp_w + 20, 24).collidepoint(mx, my):
        t_s = clamp((mx - sp_x) / sp_w, 0, 1)
        state.speed = round(0.3 + t_s * 2.7, 1)

    # Modo
    mode_lx = 545
    if pygame.Rect(mode_lx, btn_y + 20, 95, 20).collidepoint(mx, my):
        state.hybrid = False
    elif pygame.Rect(mode_lx + 100, btn_y + 20, 95, 20).collidepoint(mx, my):
        state.hybrid = True

    # Threads
    if state.hybrid:
        for i, nth in enumerate([2, 4, 8]):
            if pygame.Rect(758 + i * 55, btn_y + 20, 50, 20).collidepoint(mx, my):
                state.n_threads = nth


# ═══════════════════════════════════════════════════════
#  LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════
def main():
    global speed_dragging
    sim   = SimState()
    t_abs = 0.0

    while True:
        dt    = clock.tick(60) / 1000.0
        t_abs += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not sim.stable:
                        sim.running = not sim.running
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    sim.running = False
                    sim.step_phase()
                elif event.key == pygame.K_r:
                    sim.reset()
                elif event.key == pygame.K_h:
                    sim.hybrid = not sim.hybrid
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    sim.speed = min(3.0, round(sim.speed + 0.2, 1))
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim.speed = max(0.3, round(sim.speed - 0.2, 1))

            if event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(event.pos, sim)
                # Iniciar drag do slider
                sp_x2 = 365
                sp_w2 = 155
                sp_y2 = CTRL_Y + 35
                t_s2  = (sim.speed - 0.3) / 2.7
                kx2   = sp_x2 + int(t_s2 * sp_w2)
                if abs(event.pos[0] - kx2) < 14 and abs(event.pos[1] - sp_y2) < 14:
                    speed_dragging = True

            if event.type == pygame.MOUSEBUTTONUP:
                speed_dragging = False

            if event.type == pygame.MOUSEMOTION and speed_dragging:
                sp_x2, sp_w2 = 365, 155
                t_s2 = clamp((event.pos[0] - sp_x2) / sp_w2, 0, 1)
                sim.speed = round(0.3 + t_s2 * 2.7, 1)

        sim.update(dt)

        # ── Render ──────────────────────────────────────────
        screen.fill(COL_BG)

        draw_grid(screen, sim, t_abs)
        draw_side_panel(screen, sim, t_abs)
        draw_controls(screen, sim)

        # Barra de titulo
        pygame.draw.rect(screen, (14, 14, 22), (0, 0, WIN_W, GY - 5))
        draw_txt(screen, 'Conway Paralelo',
                 F_TITLE, COL_TEXT, 20, 15)

        # Indicador de fase no topo (centro)
        phcol = PHASE_COL[sim.phase]
        #alpha_rect(screen, phcol, (WIN_W // 2 - 185, 8, 370, 34), 40, 6)
        #pygame.draw.rect(screen, phcol, (WIN_W // 2 - 185, 8, 370, 34), 2, border_radius=6)
        draw_txt(screen, PHASE_LABEL[sim.phase], F_MED, phcol, WIN_W // 2, 17, 'center')

        pygame.display.flip()


if __name__ == '__main__':
    main()
