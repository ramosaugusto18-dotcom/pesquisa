# ============================================================
# modelo.py — Lógica Monte Carlo eleitoral
# ============================================================

import numpy as np
from collections import defaultdict
from dados import CANDIDATOS, PESQUISAS


def rodar_modelo(n_sim=10000, fator_desconto_2v=1.0, seed=42):
    """
    Roda o Monte Carlo com todas as pesquisas de PESQUISAS.
    
    Parâmetros:
        n_sim            : número de simulações (padrão 10.000)
        fator_desconto_2v: peso do 2º voto nas pesquisas optB (1.0 = igual ao 1º)
        seed             : semente aleatória para reprodutibilidade
    
    Retorna dict com resultados.
    """

    # ── 1. Montar lista de entradas ponderáveis ─────────────────────────────
    entradas = []
    for nome_inst, info in PESQUISAS.items():
        mar = info['margem']
        for cand, pct in info['dados'].items():
            if pct == 0.0:
                continue
            if cand not in CANDIDATOS:
                continue
            entradas.append((nome_inst, cand, pct, mar))

    # ── 2. Média ponderada ──────────────────────────────────────────────────
    pct_pond  = defaultdict(float)
    peso_tot  = defaultdict(float)
    sd_vals   = defaultdict(list)

    for inst, cand, pct, mar in entradas:
        w = 1 / (mar / 1.96) ** 2
        pct_pond[cand] += pct * w
        peso_tot[cand] += w
        sd_vals[cand].append(pct)

    mu_raw = {c: pct_pond[c] / peso_tot[c] if peso_tot[c] > 0 else 0.1
              for c in CANDIDATOS}

    # Normalizar para votos válidos (remove ~10% brancos/nulos)
    total = sum(mu_raw.values())
    fator_norm = (100 - 10) / total
    mu_val = {c: mu_raw[c] * fator_norm for c in CANDIDATOS}
    soma_final = sum(mu_val.values())
    MU = {c: mu_val[c] / soma_final * 100 for c in CANDIDATOS}

    # ── 3. Sigma (incerteza total) ──────────────────────────────────────────
    meses_restantes = 4.3
    fator_tend = 1 + 0.5 * (meses_restantes / 12)

    sd_within  = {c: np.mean([m for _, cc, _, m in entradas if cc == c]) / 1.96
                  for c in CANDIDATOS}
    sd_between = {c: np.std(sd_vals[c], ddof=1) if len(sd_vals[c]) > 1 else 1.0
                  for c in CANDIDATOS}
    SIGMA = {c: np.sqrt(sd_within[c]**2 + sd_between[c]**2) * fator_tend
             for c in CANDIDATOS}

    # ── 4. Monte Carlo ──────────────────────────────────────────────────────
    np.random.seed(seed)
    mu_arr    = np.array([MU[c]    for c in CANDIDATOS])
    sigma_arr = np.array([SIGMA[c] for c in CANDIDATOS])

    sims = np.zeros((n_sim, len(CANDIDATOS)))
    for i in range(n_sim):
        p = np.random.normal(mu_arr, sigma_arr)
        p = np.clip(p, 0, None)
        sims[i] = p / p.sum() * 100

    # ── 5. Probabilidades ───────────────────────────────────────────────────
    p_eleito = {}
    for idx, c in enumerate(CANDIDATOS):
        p_eleito[c] = np.mean(
            [np.argsort(row)[::-1][0] == idx or
             np.argsort(row)[::-1][1] == idx
             for row in sims]
        ) * 100

    # Pares
    pares = defaultdict(int)
    for row in sims:
        top2 = tuple(sorted([CANDIDATOS[i] for i in np.argsort(row)[::-1][:2]]))
        pares[top2] += 1
    top_pares = sorted(pares, key=lambda x: pares[x], reverse=True)[:6]
    pares_pct = {f"{p[0]} + {p[1]}": pares[p] / n_sim * 100 for p in top_pares}

    # 2ª vaga condicional — Neto 1º
    neto_idx  = CANDIDATOS.index('Alberto Neto')
    braga_idx = CANDIDATOS.index('Eduardo Braga')

    mask_neto  = [np.argsort(row)[::-1][0] == neto_idx  for row in sims]
    mask_braga = [np.argsort(row)[::-1][0] == braga_idx for row in sims]
    sims_neto  = sims[mask_neto]
    sims_braga = sims[mask_braga]

    p_2a_neto = {
        c: np.mean([np.argsort(row)[::-1][1] == CANDIDATOS.index(c)
                    for row in sims_neto]) * 100
        for c in CANDIDATOS if c != 'Alberto Neto'
    } if len(sims_neto) > 0 else {}

    p_2a_braga = {
        c: np.mean([np.argsort(row)[::-1][1] == CANDIDATOS.index(c)
                    for row in sims_braga]) * 100
        for c in CANDIDATOS if c != 'Eduardo Braga'
    } if len(sims_braga) > 0 else {}

    return {
        'MU':          MU,
        'SIGMA':       SIGMA,
        'p_eleito':    p_eleito,
        'pares':       pares_pct,
        'top_pares':   [f"{p[0]} + {p[1]}" for p in top_pares],
        'sims':        sims,
        'sims_neto':   sims_neto,
        'sims_braga':  sims_braga,
        'n_neto':      int(sum(mask_neto)),
        'n_braga':     int(sum(mask_braga)),
        'p_2a_neto':   p_2a_neto,
        'p_2a_braga':  p_2a_braga,
        'n_sim':       n_sim,
    }
