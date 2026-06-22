# ============================================================
# graficos.py — Funções de visualização
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from dados import CANDIDATOS, PARTIDOS, CORES, PESQUISAS, REJEICAO

# Tema escuro
FUNDO  = '#0d1117'
PAINEL = '#161b22'
GRADE  = '#21262d'
TEXTO  = '#e6edf3'
SUBTXT = '#8b949e'
ACENTO = '#bc8cff'
LIMIAR = '#e3b341'
VERDE  = '#3fb950'


def _setup_ax(ax, xgrid=False):
    ax.set_facecolor(PAINEL)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color=GRADE, lw=0.4, zorder=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRADE, lw=0.4, zorder=0)
    else:
        ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SUBTXT, length=0)


def _zona_cor(p):
    if p >= 70: return '#3fb950'
    if p >= 30: return '#e3b341'
    return '#f85149'


def fig_probabilidades(resultado):
    """Barras horizontais de P(eleito) + pares mais prováveis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=FUNDO,
                             gridspec_kw={'width_ratios': [1.2, 1]})

    # P(eleito)
    ax = axes[0]
    _setup_ax(ax, xgrid=True)
    ax.yaxis.grid(False)
    cords = sorted(CANDIDATOS, key=lambda c: resultado['p_eleito'][c])
    y = np.arange(len(cords))
    for yi, c in enumerate(cords):
        pv  = resultado['p_eleito'][c] / 100
        cor = _zona_cor(resultado['p_eleito'][c])
        ax.barh(yi, pv, height=0.55, color=cor, alpha=0.88, zorder=3)
        ax.text(pv + 0.012, yi, f"{resultado['p_eleito'][c]:.1f}%",
                va='center', color=cor, fontsize=11, fontweight='bold')
        ax.text(0.008, yi, f'({PARTIDOS[c]})',
                va='center', color=SUBTXT, fontsize=8.5)
    ax.axvline(0.5, color=LIMIAR, lw=0.8, ls='--', alpha=0.8)
    ax.text(0.515, -0.5, '50%', color=LIMIAR, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(cords, color=TEXTO, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'],
                       color=SUBTXT, fontsize=9)
    ax.set_xlabel('Probabilidade de eleição (top-2)', color=SUBTXT, fontsize=10)
    ax.set_title('P(eleito) por candidato', color=TEXTO,
                 fontsize=11, fontweight='bold', pad=8)
    leg = [mpatches.Patch(color='#3fb950', label='Favorito (>70%)'),
           mpatches.Patch(color='#e3b341', label='Competitivo (30–70%)'),
           mpatches.Patch(color='#f85149', label='Improvável (<30%)')]
    ax.legend(handles=leg, loc='lower right', fontsize=8.5,
              facecolor=PAINEL, edgecolor=GRADE, labelcolor=TEXTO,
              framealpha=0.92)

    # Pares
    ax2 = axes[1]
    _setup_ax(ax2, xgrid=True)
    ax2.yaxis.grid(False)
    par_items = sorted(resultado['pares'].items(), key=lambda x: x[1])
    py = np.arange(len(par_items))
    for yi, (par, pv) in enumerate(par_items):
        c1n = par.split(' + ')[0]
        cor = CORES.get(c1n, '#8b949e')
        ax2.barh(yi, pv / 100, height=0.55, color=cor, alpha=0.75, zorder=3)
        ax2.text(pv / 100 + 0.01, yi, f'{pv:.1f}%',
                 va='center', color=TEXTO, fontsize=9.5, fontweight='bold')
    ax2.set_yticks(py)
    ax2.set_yticklabels([p.replace(' + ', ' +\n') for p, _ in par_items],
                        color=TEXTO, fontsize=8.5)
    ax2.set_xlim(0, 1.05)
    ax2.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax2.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'],
                        color=SUBTXT, fontsize=9)
    ax2.set_xlabel('Prob. do par ocupar as 2 vagas', color=SUBTXT, fontsize=10)
    ax2.set_title('Pares mais prováveis', color=TEXTO,
                  fontsize=11, fontweight='bold', pad=8)

    n = resultado['n_sim']
    fig.suptitle(f'Senado AM 2026 | {len(PESQUISAS)} institutos | {n:,} simulações',
                 color=TEXTO, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def fig_espaguete(resultado):
    """Trajetórias Monte Carlo até outubro/2026."""
    sims = resultado['sims']
    meses = list(range(5))
    mes_labels = ['mai/26', 'jun/26', 'jul/26', 'ago/26', 'out/26']
    N_TRAJ = 250
    np.random.seed(7)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=FUNDO)
    _setup_ax(ax)

    for idx_c, cand in enumerate(CANDIDATOS):
        mu_c  = resultado['MU'][cand]
        sg_c  = resultado['SIGMA'][cand]
        cor   = CORES[cand]
        trajs = np.zeros((N_TRAJ, 5))
        for m in range(5):
            frac  = m / 4
            sg_m  = sg_c * (1 + frac * 0.5)
            if m == 4:
                idx_ = np.random.choice(len(sims), N_TRAJ, replace=False)
                vals = sims[idx_, idx_c]
            else:
                vals = np.clip(np.random.normal(mu_c, sg_m, N_TRAJ), 0, None)
            trajs[:, m] = vals
        for t in range(N_TRAJ):
            ax.plot(meses, trajs[t], color=cor, alpha=0.05, lw=0.35, zorder=2)
        p10 = np.percentile(trajs, 10, axis=0)
        p90 = np.percentile(trajs, 90, axis=0)
        ax.fill_between(meses, p10, p90, color=cor, alpha=0.08, zorder=1)
        med = np.median(trajs, axis=0)
        ax.plot(meses, med, color=cor, lw=2.2, zorder=6)
        ax.text(4.08, med[-1], f'{cand}\n({PARTIDOS[cand]})',
                color=cor, fontsize=7.5, va='center', ha='left')

    ax.axvline(4, color=ACENTO, lw=0.9, ls='--', alpha=0.8)
    ax.text(3.88, 46, '1º turno\n04/out/26', color=ACENTO,
            fontsize=8, ha='right', style='italic')
    ax.set_xticks(meses)
    ax.set_xticklabels(mes_labels, color=SUBTXT, fontsize=9)
    ax.set_ylim(0, 52)
    ax.set_yticks(range(0, 53, 5))
    ax.set_yticklabels([f'{v}%' for v in range(0, 53, 5)], color=SUBTXT)
    ax.set_xlim(-0.2, 6.5)
    ax.set_ylabel('Intenção de voto simulada (%)', color=SUBTXT, fontsize=10)
    fig.suptitle(f'Trajetórias Monte Carlo — Senado AM 2026 | {N_TRAJ} trajetórias',
                 color=TEXTO, fontsize=13, fontweight='bold', y=0.97)
    ax.set_title('Faixa P10–P90 | Linha sólida = mediana',
                 color=SUBTXT, fontsize=9, pad=6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def fig_segunda_vaga(resultado):
    """2ª vaga condicional — Neto 1º e Braga 1º lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=FUNDO)

    for ax, sims_c, p_2a, lider, excluir, n_c in [
        (axes[0], resultado['sims_neto'],  resultado['p_2a_neto'],
         'Alberto Neto',  'Alberto Neto',  resultado['n_neto']),
        (axes[1], resultado['sims_braga'], resultado['p_2a_braga'],
         'Eduardo Braga', 'Eduardo Braga', resultado['n_braga']),
    ]:
        _setup_ax(ax)
        for idx, c in enumerate(CANDIDATOS):
            if c == excluir:
                continue
            vals = sims_c[:, idx]
            if vals.std() < 0.3:
                continue
            xr = np.linspace(max(0, vals.min() - 1), min(50, vals.max() + 1), 300)
            try:
                kde = gaussian_kde(vals, bw_method=0.18)
                yk  = kde(xr)
                ax.fill_between(xr, yk, color=CORES[c], alpha=0.22)
                ax.plot(xr, yk, color=CORES[c], lw=2.0,
                        label=f'{c} ({PARTIDOS[c]})')
                med = np.median(vals)
                ax.axvline(med, color=CORES[c], lw=1.3, ls='--', alpha=0.8)
                ax.text(med + 0.4, max(yk) * 0.5, f'{med:.1f}%',
                        color=CORES[c], fontsize=8.5, va='center')
            except Exception:
                pass

        txt = "\n".join([f"P({c}) = {p_2a[c]:.1f}%"
                         for c in sorted(p_2a, key=lambda x: p_2a[x],
                                         reverse=True)[:4]])
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                color=ACENTO, fontsize=10.5, fontweight='bold',
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.55', facecolor=PAINEL,
                          edgecolor=ACENTO, alpha=0.95, lw=0.9))
        ax.set_xlim(0, 50)
        ax.set_xticks(range(0, 51, 5))
        ax.set_xticklabels([f'{v}%' for v in range(0, 51, 5)],
                           color=SUBTXT, fontsize=9)
        ax.set_ylabel('Densidade', color=SUBTXT, fontsize=10)
        ax.set_xlabel('Intenção de voto (%)', color=SUBTXT, fontsize=10)
        ax.legend(fontsize=8.5, facecolor=PAINEL, edgecolor=GRADE,
                  labelcolor=TEXTO, framealpha=0.92, loc='upper left')
        ax.set_title(f'2ª vaga | Condicional: {lider} em 1º\n({n_c:,} sim.)',
                     color=TEXTO, fontsize=11, fontweight='bold', pad=8)

    fig.suptitle('Disputa pela 2ª vaga — Senado AM 2026',
                 color=TEXTO, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def fig_rejeicao():
    """Gráfico de rejeição dos candidatos."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=FUNDO)
    _setup_ax(ax, xgrid=True)
    ax.yaxis.grid(False)

    items = sorted(REJEICAO.items(), key=lambda x: x[1])
    y = np.arange(len(items))
    for yi, (c, rv) in enumerate(items):
        cor = CORES.get(c, '#8b949e')
        ax.barh(yi, rv / 100, height=0.55, color=cor, alpha=0.75, zorder=3)
        ax.text(rv / 100 + 0.005, yi, f'{rv:.1f}%',
                va='center', color=cor, fontsize=11, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels([f'{c} ({PARTIDOS.get(c, "")})' for c, _ in items],
                       color=TEXTO, fontsize=10, fontweight='bold')
    ax.set_xlim(0, 0.55)
    ax.set_xticks([0, .1, .2, .3, .4, .5])
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%'],
                       color=SUBTXT, fontsize=9)
    ax.set_xlabel('% que não votaria de jeito nenhum', color=SUBTXT, fontsize=10)
    fig.suptitle('Rejeição dos candidatos ao Senado AM 2026',
                 color=TEXTO, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig
