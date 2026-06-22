# ============================================================
# app.py — Dashboard Streamlit | Senado AM 2026
# Para rodar localmente:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd

from dados import CANDIDATOS, PARTIDOS, CORES, PESQUISAS, REJEICAO
from modelo import rodar_modelo
from graficos import (fig_probabilidades, fig_espaguete,
                      fig_segunda_vaga, fig_rejeicao)

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Senado AM 2026 — Monte Carlo",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS mínimo para tema escuro
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e6edf3; }
    .metric-label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — controles ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configurações")
    st.markdown("---")

    n_sim = st.select_slider(
        "Número de simulações",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
        help="Mais simulações = mais preciso, mas mais lento"
    )

    st.markdown("---")
    st.subheader("📋 Pesquisas ativas")
    st.caption("Desmarque para remover do modelo")

    pesquisas_ativas = {}
    for nome in PESQUISAS:
        pesquisas_ativas[nome] = st.checkbox(nome, value=True)

    institutos_selecionados = [n for n, v in pesquisas_ativas.items() if v]
    n_ativas = len(institutos_selecionados)

    if n_ativas == 0:
        st.error("Selecione pelo menos 1 pesquisa!")
        st.stop()

    st.markdown("---")
    st.caption(f"**{n_ativas}** pesquisas selecionadas")
    n_total = sum(PESQUISAS[n]['n'] for n in institutos_selecionados)
    st.caption(f"**{n_total:,}** entrevistas no total")

# ── Rodar modelo ─────────────────────────────────────────────────────────────
# Cache para não recalcular toda vez que o usuário rola a página
@st.cache_data(show_spinner="Rodando 10.000 simulações Monte Carlo...")
def calcular(n_sim, institutos):
    # Filtrar apenas os institutos selecionados
    import dados as _d
    pesquisas_backup = _d.PESQUISAS.copy()
    _d.PESQUISAS = {k: v for k, v in pesquisas_backup.items() if k in institutos}
    resultado = rodar_modelo(n_sim=n_sim)
    _d.PESQUISAS = pesquisas_backup
    return resultado

resultado = calcular(n_sim, tuple(institutos_selecionados))

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("🗳️ Senado AM 2026 — Projeção Monte Carlo")
st.caption(f"Modelo com **{n_ativas} institutos** | **{n_sim:,} simulações** | "
           f"Eleição: 4 de outubro de 2026")
st.markdown("---")

# ── Métricas rápidas ──────────────────────────────────────────────────────────
cols = st.columns(len(CANDIDATOS))
for i, c in enumerate(sorted(CANDIDATOS,
                              key=lambda x: resultado['p_eleito'][x],
                              reverse=True)):
    p = resultado['p_eleito'][c]
    mu = resultado['MU'][c]
    with cols[i]:
        cor_emoji = "🟢" if p >= 70 else "🟡" if p >= 30 else "🔴"
        st.metric(
            label=f"{cor_emoji} {c}",
            value=f"{p:.1f}%",
            delta=f"média {mu:.1f}%",
            delta_color="off"
        )

st.markdown("---")

# ── Abas principais ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Probabilidades",
    "🍝 Trajetórias",
    "🥈 2ª Vaga",
    "❌ Rejeição",
    "📋 Tabela de pesquisas",
])

with tab1:
    st.subheader("Probabilidade de eleição (top-2) e pares mais prováveis")
    fig = fig_probabilidades(resultado)
    st.pyplot(fig, use_container_width=True)

    # Par dominante em destaque
    par_top = resultado['top_pares'][0]
    pct_par = resultado['pares'][par_top]
    st.info(f"**Par mais provável:** {par_top} com **{pct_par:.1f}%** de probabilidade "
            f"de ocupar as 2 vagas.")

with tab2:
    st.subheader("Trajetórias simuladas até outubro de 2026")
    st.caption("Cada linha = uma eleição imaginária. Faixa sombreada = intervalo P10–P90.")
    fig = fig_espaguete(resultado)
    st.pyplot(fig, use_container_width=True)

with tab3:
    st.subheader("Disputa pela 2ª vaga — cenários condicionais")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Neto em 1º → Braga leva 2ª",
                  f"{resultado['p_2a_neto'].get('Eduardo Braga', 0):.1f}%")
    with col2:
        st.metric("Braga em 1º → Neto leva 2ª",
                  f"{resultado['p_2a_braga'].get('Alberto Neto', 0):.1f}%")
    fig = fig_segunda_vaga(resultado)
    st.pyplot(fig, use_container_width=True)

with tab4:
    st.subheader("Rejeição dos candidatos")
    st.caption("Fonte: Veritá Eleições Mai/2026 — "
               "pergunta: 'Em quem você não votaria de jeito nenhum?'")
    fig = fig_rejeicao()
    st.pyplot(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.success("**Plínio Valério** tem a menor rejeição do ciclo: **1,3%**. "
                   "Candidato de consenso — quase ninguém recusa votar nele.")
    with col2:
        st.warning("**Alberto Neto** tem a maior rejeição: **36,6%**. "
                   "Candidato polarizador — base fiel, mas afasta eleitores do campo oposto.")

with tab5:
    st.subheader("Pesquisas no modelo")

    # Tabela resumo
    rows = []
    for nome, info in PESQUISAS.items():
        ativo = nome in institutos_selecionados
        rows.append({
            'Instituto':  nome,
            'TSE':        info['tse'],
            'Campo':      info['campo'],
            'n':          info['n'],
            'Margem':     f"±{info['margem']}pp",
            'Método':     info['metodo'],
            'Tipo':       'Opção B' if info['tipo'] == 'optB' else '1º voto',
            'Ativo':      '✅' if ativo else '❌',
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Médias ponderadas
    st.markdown("### Médias ponderadas (entrada do Monte Carlo)")
    mu_rows = [{'Candidato': c, 'Partido': PARTIDOS[c],
                'Média (%)': f"{resultado['MU'][c]:.1f}",
                'Sigma (pp)': f"{resultado['SIGMA'][c]:.2f}"}
               for c in sorted(CANDIDATOS,
                               key=lambda x: resultado['MU'][x], reverse=True)]
    st.dataframe(pd.DataFrame(mu_rows), use_container_width=True, hide_index=True)

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Modelo Monte Carlo eleitoral | Senado AM 2026 | "
    "Média ponderada por 1/variância | "
    "Sigma = √(sd_within² + sd_between²) × fator_tendência"
)
