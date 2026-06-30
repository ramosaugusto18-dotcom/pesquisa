# ============================================================
# app.py — Dashboard Senado AM 2026
# Com upload de PDF, formulário e chat analítico
# streamlit run app.py
# ============================================================

import json, base64, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import defaultdict

try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

# ── Configuração ──────────────────────────────────────────────
st.set_page_config(
    page_title="Senado AM 2026 — Monte Carlo",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.stApp{background-color:#0d1117;color:#e6edf3}
.block-container{padding-top:1.5rem}
h1,h2,h3{color:#e6edf3}
div[data-testid="stForm"]{border:none;padding:0}
div[data-testid="stChatMessage"]{background-color:#161b22;border-radius:8px;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────
CANDIDATOS = ['Alberto Neto','Eduardo Braga','Plinio Valerio',
              'Wilson Lima','Marcos Rotta','Marcelo Ramos']
PARTIDOS   = {'Alberto Neto':'PL','Eduardo Braga':'MDB',
               'Plinio Valerio':'PSDB','Wilson Lima':'UNIAO',
               'Marcos Rotta':'Avante','Marcelo Ramos':'PT'}
CORES      = {'Alberto Neto':'#f85149','Eduardo Braga':'#58a6ff',
               'Plinio Valerio':'#e3b341','Wilson Lima':'#3fb950',
               'Marcos Rotta':'#8b949e','Marcelo Ramos':'#ff7b72'}
FUNDO='#0d1117'; PAINEL='#161b22'; GRADE='#21262d'
TEXTO='#e6edf3'; SUBTXT='#8b949e'; ACENTO='#bc8cff'; LIMIAR='#e3b341'

# ── Pesquisas padrão ──────────────────────────────────────────
PESQUISAS_PADRAO = [
    {'nome':'OpManauara','campo':'01/mar','n':1200,'margem':3.0,'metodo':'Presencial',
     'Alberto Neto':19.0,'Eduardo Braga':23.0,'Plinio Valerio':11.0,
     'Wilson Lima':0.0,'Marcos Rotta':12.0,'Marcelo Ramos':9.0},
    {'nome':'Quaest','campo':'07/mar','n':1500,'margem':3.0,'metodo':'Presencial',
     'Alberto Neto':28.0,'Eduardo Braga':28.0,'Plinio Valerio':14.0,
     'Wilson Lima':0.0,'Marcos Rotta':6.0,'Marcelo Ramos':10.0},
    {'nome':'AtlasIntel Mar','campo':'10/mar','n':1138,'margem':3.0,'metodo':'Online',
     'Alberto Neto':24.0,'Eduardo Braga':19.3,'Plinio Valerio':17.4,
     'Wilson Lima':0.0,'Marcos Rotta':2.4,'Marcelo Ramos':15.3},
    {'nome':'RealTime','campo':'13/mar','n':1500,'margem':2.0,'metodo':'Telefone',
     'Alberto Neto':28.4,'Eduardo Braga':17.0,'Plinio Valerio':14.0,
     'Wilson Lima':0.0,'Marcos Rotta':9.0,'Marcelo Ramos':8.0},
    {'nome':'Veritá Mar','campo':'18/mar','n':1220,'margem':3.0,'metodo':'URA',
     'Alberto Neto':27.0,'Eduardo Braga':20.7,'Plinio Valerio':6.4,
     'Wilson Lima':0.0,'Marcos Rotta':4.3,'Marcelo Ramos':8.8},
    {'nome':'Iveritas','campo':'20/mar','n':2604,'margem':3.0,'metodo':'Presencial (Opção B)',
     'Alberto Neto':31.78,'Eduardo Braga':25.69,'Plinio Valerio':22.78,
     'Wilson Lima':0.0,'Marcos Rotta':7.24,'Marcelo Ramos':9.88},
    {'nome':'AtlasIntel Mai','campo':'05/mai','n':1415,'margem':3.0,'metodo':'Online (Opção B)',
     'Alberto Neto':23.7,'Eduardo Braga':21.6,'Plinio Valerio':15.5,
     'Wilson Lima':7.5,'Marcos Rotta':4.7,'Marcelo Ramos':17.9},
    {'nome':'Census','campo':'10/mai','n':2000,'margem':2.2,'metodo':'CATI (Opção B)',
     'Alberto Neto':22.95,'Eduardo Braga':26.78,'Plinio Valerio':15.85,
     'Wilson Lima':13.66,'Marcos Rotta':10.38,'Marcelo Ramos':10.38},
    {'nome':'IPEN','campo':'22/mai','n':1200,'margem':2.8,'metodo':'Presencial (Opção B)',
     'Alberto Neto':20.27,'Eduardo Braga':34.17,'Plinio Valerio':10.44,
     'Wilson Lima':16.47,'Marcos Rotta':9.76,'Marcelo Ramos':8.88},
    {'nome':'Veritá Mai','campo':'28/mai','n':1220,'margem':2.8,'metodo':'Presencial (Opção B)',
     'Alberto Neto':29.76,'Eduardo Braga':26.37,'Plinio Valerio':16.28,
     'Wilson Lima':7.72,'Marcos Rotta':3.86,'Marcelo Ramos':16.01},
]

# ── Estado da sessão ──────────────────────────────────────────
if 'pesquisas'     not in st.session_state:
    st.session_state.pesquisas = [p.copy() for p in PESQUISAS_PADRAO]
if 'ativas'        not in st.session_state:
    st.session_state.ativas = {p['nome']:True for p in PESQUISAS_PADRAO}
if 'chat_history'  not in st.session_state:
    st.session_state.chat_history = []

# ── Monte Carlo ───────────────────────────────────────────────
def rodar_modelo(pesquisas_sel, n_sim=10000):
    pct_pond=defaultdict(float); peso_tot=defaultdict(float); sd_vals=defaultdict(list)
    for p in pesquisas_sel:
        w=1/(p['margem']/1.96)**2
        for c in CANDIDATOS:
            pct=p.get(c,0.0)
            if pct<=0: continue
            pct_pond[c]+=pct*w; peso_tot[c]+=w; sd_vals[c].append(pct)
    mu_raw={c:pct_pond[c]/peso_tot[c] if peso_tot[c]>0 else 0.1 for c in CANDIDATOS}
    total=sum(mu_raw.values()); fator=90/total
    mu_v={c:mu_raw[c]*fator for c in CANDIDATOS}
    soma=sum(mu_v.values()); MU={c:mu_v[c]/soma*100 for c in CANDIDATOS}
    sd_w={c:np.mean([p['margem'] for p in pesquisas_sel if p.get(c,0)>0])/1.96
          if any(p.get(c,0)>0 for p in pesquisas_sel) else 1.5 for c in CANDIDATOS}
    sd_b={c:np.std(sd_vals[c],ddof=1) if len(sd_vals[c])>1 else 1.0 for c in CANDIDATOS}
    ft=1+0.5*(4.0/12)
    SIGMA={c:np.sqrt(sd_w[c]**2+sd_b[c]**2)*ft for c in CANDIDATOS}
    np.random.seed(42)
    mu_a=np.array([MU[c] for c in CANDIDATOS]); sg_a=np.array([SIGMA[c] for c in CANDIDATOS])
    sims=np.zeros((n_sim,len(CANDIDATOS)))
    for i in range(n_sim):
        p2=np.clip(np.random.normal(mu_a,sg_a),0,None); sims[i]=p2/p2.sum()*100
    p_el={}
    for idx,c in enumerate(CANDIDATOS):
        p_el[c]=np.mean([np.argsort(r)[::-1][0]==idx or
                         np.argsort(r)[::-1][1]==idx for r in sims])*100
    pares=defaultdict(int)
    for r in sims:
        t=tuple(sorted([CANDIDATOS[i] for i in np.argsort(r)[::-1][:2]])); pares[t]+=1
    top=sorted(pares,key=lambda x:pares[x],reverse=True)[:6]
    ni=CANDIDATOS.index('Alberto Neto'); bi=CANDIDATOS.index('Eduardo Braga')
    mn=[np.argsort(r)[::-1][0]==ni for r in sims]
    mb=[np.argsort(r)[::-1][0]==bi for r in sims]
    sn=sims[mn]; sb=sims[mb]
    p2n={c:np.mean([np.argsort(r)[::-1][1]==CANDIDATOS.index(c) for r in sn])*100
         for c in CANDIDATOS if c!='Alberto Neto'} if sn.shape[0]>0 else {}
    p2b={c:np.mean([np.argsort(r)[::-1][1]==CANDIDATOS.index(c) for r in sb])*100
         for c in CANDIDATOS if c!='Eduardo Braga'} if sb.shape[0]>0 else {}
    return {'MU':MU,'SIGMA':SIGMA,'p_el':p_el,'sims':sims,
            'sn':sn,'sb':sb,'n_sn':int(sum(mn)),'n_sb':int(sum(mb)),
            'p2n':p2n,'p2b':p2b,'n_sim':n_sim,
            'pares':{f"{t[0]} + {t[1]}":pares[t]/n_sim*100 for t in top}}

def setup_ax(ax, xgrid=False):
    ax.set_facecolor(PAINEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True,color=GRADE,lw=0.4,zorder=0)
    if xgrid: ax.xaxis.grid(True,color=GRADE,lw=0.4,zorder=0)
    else: ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(colors=SUBTXT,length=0)

def zona_cor(p):
    if p>=70: return '#3fb950'
    if p>=30: return '#e3b341'
    return '#f85149'

# ── Contexto para o chat ──────────────────────────────────────
def montar_contexto(pesquisas_ativas, resultado):
    """Monta o system prompt com todos os dados do modelo atual."""

    # Tabela de pesquisas
    linhas_pesq = []
    for p in pesquisas_ativas:
        nums = " | ".join([f"{c.split()[0]}={p.get(c,0):.1f}%" for c in CANDIDATOS if p.get(c,0)>0])
        linhas_pesq.append(
            f"- {p['nome']} ({p['campo']}, n={p['n']:,}, ±{p['margem']}pp, {p['metodo']}): {nums}"
        )

    # Médias ponderadas
    linhas_mu = []
    for c in sorted(CANDIDATOS, key=lambda x: resultado['MU'][x], reverse=True):
        linhas_mu.append(
            f"- {c} ({PARTIDOS[c]}): média={resultado['MU'][c]:.1f}% | "
            f"sigma={resultado['SIGMA'][c]:.2f}pp | P(eleito)={resultado['p_el'][c]:.1f}%"
        )

    # Pares
    linhas_pares = [f"- {par}: {pct:.1f}%" for par,pct in
                    sorted(resultado['pares'].items(), key=lambda x:x[1], reverse=True)]

    # 2ª vaga condicional
    p2n_str = " | ".join([f"{c.split()[0]}={v:.1f}%"
                           for c,v in sorted(resultado['p2n'].items(),
                                             key=lambda x:x[1], reverse=True)[:3]])
    p2b_str = " | ".join([f"{c.split()[0]}={v:.1f}%"
                           for c,v in sorted(resultado['p2b'].items(),
                                             key=lambda x:x[1], reverse=True)[:3]])

    # Análise de tendência por candidato
    tendencias = []
    for c in CANDIDATOS:
        vals = [(p['campo'], p.get(c,0)) for p in pesquisas_ativas if p.get(c,0)>0]
        if len(vals) >= 2:
            primeiro = vals[0][1]; ultimo = vals[-1][1]
            delta = ultimo - primeiro
            sinal = "▲" if delta > 1 else "▼" if delta < -1 else "→"
            tendencias.append(f"- {c}: {primeiro:.1f}% → {ultimo:.1f}% ({sinal}{abs(delta):.1f}pp)")

    context = f"""Você é um analista eleitoral especializado nas eleições do Amazonas 2026.
Você tem acesso completo ao modelo Monte Carlo do Senado AM 2026 e deve responder perguntas
com base nos dados reais abaixo. Seja direto, preciso e use os números do modelo.

═══════════════════════════════════════════════
DADOS DO MODELO — {len(pesquisas_ativas)} PESQUISAS ATIVAS
═══════════════════════════════════════════════

PESQUISAS NO MODELO:
{chr(10).join(linhas_pesq)}

MÉDIAS PONDERADAS (entrada Monte Carlo):
{chr(10).join(linhas_mu)}

PARES MAIS PROVÁVEIS:
{chr(10).join(linhas_pares)}

2ª VAGA CONDICIONAL:
- Se Alberto Neto for 1º: {p2n_str}
- Se Eduardo Braga for 1º: {p2b_str}

TENDÊNCIA POR CANDIDATO (1ª pesquisa → última):
{chr(10).join(tendencias) if tendencias else '(dados insuficientes)'}

METODOLOGIA DO MODELO:
- Ponderação: 1/variância (menor margem = maior peso)
- Sigma = √(sd_within² + sd_between²) × fator_tendência
- n_simulações={resultado['n_sim']:,}
- Eleição: 4 de outubro de 2026

INSTITUTOS E PESOS (quanto maior o peso, mais influência no modelo):
{chr(10).join([f"- {p['nome']}: peso={1/(p['margem']/1.96)**2:.2f} (n={p['n']:,}, ±{p['margem']}pp)" for p in pesquisas_ativas])}

Responda sempre em português, de forma analítica e objetiva.
Use os dados acima para fundamentar suas análises.
Quando comparar institutos, explique diferenças metodológicas (presencial vs online vs CATI vs URA).
"""
    return context


# ── Extração via Claude API ───────────────────────────────────
def extrair_pesquisa_com_claude(arquivo_bytes, mime_type, api_key):
    """
    Envia PDF ou imagem para o Claude e extrai os dados da pesquisa.
    Retorna dict com os dados ou lança exceção com mensagem clara.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = """Analise este documento de pesquisa eleitoral para o Senado do Amazonas 2026.

Candidatos relevantes:
- Alberto Neto (PL) — também chamado "Capitão Alberto Neto"
- Eduardo Braga (MDB)
- Plinio Valerio (PSDB) — também grafado "Plínio Valério"
- Wilson Lima (União Brasil / UNIAO)
- Marcos Rotta (Avante / PDT)
- Marcelo Ramos (PT)

INSTRUÇÕES DE CÁLCULO:
1. Se a pesquisa tiver 1º voto E 2º voto SEPARADOS:
   - Calcule o consolidado Opção B:
   - soma[candidato] = 1v[candidato] + 2v[candidato]
   - total = soma de todos os candidatos
   - optB[candidato] = soma[candidato] / total * 100
   - Use esses valores como os percentuais finais

2. Se a pesquisa tiver apenas um número por candidato (consolidado ou só 1º voto):
   - Use esse número diretamente como percentual

3. Use 0.0 para candidatos que não foram testados na pesquisa.

IMPORTANTE: Retorne SOMENTE o JSON abaixo, sem nenhum texto antes ou depois,
sem marcadores de código, sem explicações. Apenas o JSON puro:

{"nome": "Nome do instituto", "campo": "DD/mmm", "n": 1500, "margem": 3.0, "metodo": "Presencial", "Alberto Neto": 28.5, "Eduardo Braga": 25.0, "Plinio Valerio": 15.0, "Wilson Lima": 10.0, "Marcos Rotta": 5.0, "Marcelo Ramos": 8.0, "observacao": "nota breve sobre metodologia"}"""

    b64 = base64.standard_b64encode(arquivo_bytes).decode("utf-8")

    # Montar conteúdo da mensagem
    if mime_type == "application/pdf":
        conteudo = [
            {"type": "document",
             "source": {"type": "base64", "media_type": mime_type, "data": b64}},
            {"type": "text", "text": prompt}
        ]
    else:
        conteudo = [
            {"type": "image",
             "source": {"type": "base64", "media_type": mime_type, "data": b64}},
            {"type": "text", "text": prompt}
        ]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": conteudo}],
    )

    # Extrair texto da resposta
    if not response.content:
        raise ValueError("O Claude retornou uma resposta vazia. Tente novamente.")

    texto_bruto = response.content[0].text.strip()

    if not texto_bruto:
        raise ValueError("O Claude retornou texto vazio. Verifique se o arquivo está legível.")

    # Limpar marcadores de código se existirem
    texto = texto_bruto
    texto = re.sub(r'^```json\s*', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'^```\s*', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'\s*```\s*$', '', texto, flags=re.MULTILINE)
    texto = texto.strip()

    # Tentar extrair JSON mesmo que haja texto antes/depois
    if not texto.startswith('{'):
        match = re.search(r'\{[\s\S]*\}', texto)
        if match:
            texto = match.group(0)
        else:
            raise ValueError(
                f"Não foi possível encontrar JSON na resposta do Claude.\n\n"
                f"Resposta recebida:\n{texto_bruto[:500]}"
            )

    try:
        dados = json.loads(texto)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON inválido na resposta do Claude: {e}\n\n"
            f"Texto recebido:\n{texto[:500]}"
        )

    # Validar campos mínimos
    campos_obrigatorios = ['nome', 'n', 'margem']
    faltando = [c for c in campos_obrigatorios if c not in dados]
    if faltando:
        raise ValueError(
            f"Dados incompletos — campos não encontrados: {faltando}\n"
            f"Dados extraídos: {dados}"
        )

    # Garantir que todos os candidatos existam
    for c in CANDIDATOS:
        if c not in dados:
            dados[c] = 0.0

    # Garantir tipos corretos
    dados['n'] = int(dados.get('n', 0))
    dados['margem'] = float(dados.get('margem', 3.0))
    for c in CANDIDATOS:
        dados[c] = float(dados.get(c, 0.0))

    return dados


def chat_com_claude(mensagem, historico, contexto, api_key):
    """Envia mensagem para o Claude com contexto completo do modelo."""
    client = anthropic.Anthropic(api_key=api_key)
    messages = []
    for msg in historico:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": mensagem})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=contexto,
        messages=messages,
    )
    return response.content[0].text


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configurações")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Chave API Anthropic",
        type="password",
        placeholder="sk-ant-...",
        help="Usada no upload de PDF e no chat analítico. console.anthropic.com"
    )

    st.markdown("---")
    n_sim = st.select_slider("Simulações",
        options=[1000,5000,10000,25000,50000], value=10000)

    st.markdown("---")
    st.subheader("📋 Pesquisas ativas")
    for p in st.session_state.pesquisas:
        nome = p['nome']
        if nome not in st.session_state.ativas:
            st.session_state.ativas[nome] = True
        checked = st.checkbox(
            f"{nome} ({p['campo']}, n={p['n']:,})",
            value=st.session_state.ativas[nome],
            key=f"ck_{nome}"
        )
        st.session_state.ativas[nome] = checked

    ativas = [p for p in st.session_state.pesquisas
              if st.session_state.ativas.get(p['nome'],True)]

    st.markdown("---")
    st.caption(f"**{len(ativas)}** pesquisas | **{sum(p['n'] for p in ativas):,}** entrevistas")

    if st.button("🔄 Restaurar originais"):
        st.session_state.pesquisas = [p.copy() for p in PESQUISAS_PADRAO]
        st.session_state.ativas = {p['nome']:True for p in PESQUISAS_PADRAO}
        st.rerun()

if not ativas:
    st.error("Selecione pelo menos 1 pesquisa."); st.stop()

resultado = rodar_modelo(ativas, n_sim)

# ── CABEÇALHO ─────────────────────────────────────────────────
st.title("🗳️ Senado AM 2026 — Monte Carlo")
st.caption(f"{len(ativas)} institutos · {n_sim:,} simulações · Eleição 04/out/2026")
st.markdown("---")

cols = st.columns(len(CANDIDATOS))
for i,c in enumerate(sorted(CANDIDATOS, key=lambda x:resultado['p_el'][x], reverse=True)):
    e = "🟢" if resultado['p_el'][c]>=70 else "🟡" if resultado['p_el'][c]>=30 else "🔴"
    with cols[i]:
        st.metric(f"{e} {c}", f"{resultado['p_el'][c]:.1f}%",
                  f"média {resultado['MU'][c]:.1f}%", delta_color="off")
st.markdown("---")

# ── ABAS ──────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 Probabilidades",
    "🥈 2ª Vaga",
    "💬 Chat analítico",
    "📄 Upload de pesquisa",
    "📋 Dados",
])

# ─── ABA 1 — PROBABILIDADES ───────────────────────────────────
with tab1:
    fig,axes = plt.subplots(1,2,figsize=(16,6),facecolor=FUNDO,
                            gridspec_kw={'width_ratios':[1.2,1]})
    ax = axes[0]; setup_ax(ax,xgrid=True); ax.yaxis.grid(False)
    cords = sorted(CANDIDATOS, key=lambda c:resultado['p_el'][c])
    y = np.arange(len(cords))
    for yi,c in enumerate(cords):
        pv=resultado['p_el'][c]/100; cor=zona_cor(resultado['p_el'][c])
        ax.barh(yi,pv,height=0.55,color=cor,alpha=0.88,zorder=3)
        ax.text(pv+0.012,yi,f"{resultado['p_el'][c]:.1f}%",
                va='center',color=cor,fontsize=11,fontweight='bold')
        ax.text(0.008,yi,f'({PARTIDOS[c]})',va='center',color=SUBTXT,fontsize=8.5)
    ax.axvline(0.5,color=LIMIAR,lw=0.8,ls='--',alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(cords,color=TEXTO,fontsize=11,fontweight='bold')
    ax.set_xlim(0,1.18); ax.set_xticks([0,.2,.4,.6,.8,1.0])
    ax.set_xticklabels(['0%','20%','40%','60%','80%','100%'],color=SUBTXT)
    ax.set_xlabel('Probabilidade de eleição (top-2)',color=SUBTXT)
    ax.set_title('P(eleito)',color=TEXTO,fontsize=11,fontweight='bold',pad=8)
    ax2 = axes[1]; setup_ax(ax2,xgrid=True); ax2.yaxis.grid(False)
    par_items = sorted(resultado['pares'].items(),key=lambda x:x[1])
    py = np.arange(len(par_items))
    for yi,(par,pv) in enumerate(par_items):
        cor = CORES.get(par.split(' + ')[0],'#8b949e')
        ax2.barh(yi,pv/100,height=0.55,color=cor,alpha=0.75,zorder=3)
        ax2.text(pv/100+0.01,yi,f'{pv:.1f}%',va='center',color=TEXTO,fontsize=9.5,fontweight='bold')
    ax2.set_yticks(py)
    ax2.set_yticklabels([p.replace(' + ',' +\n') for p,_ in par_items],color=TEXTO,fontsize=8.5)
    ax2.set_xlim(0,1.05); ax2.set_xticks([0,.2,.4,.6,.8,1.0])
    ax2.set_xticklabels(['0%','20%','40%','60%','80%','100%'],color=SUBTXT)
    ax2.set_xlabel('Prob. do par ocupar as 2 vagas',color=SUBTXT)
    ax2.set_title('Pares mais prováveis',color=TEXTO,fontsize=11,fontweight='bold',pad=8)
    fig.suptitle(f'Senado AM 2026 | {len(ativas)} institutos | {n_sim:,} simulações',
                 color=TEXTO,fontsize=13,fontweight='bold',y=1.01)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True)

# ─── ABA 2 — 2ª VAGA ──────────────────────────────────────────
with tab2:
    c1,c2 = st.columns(2)
    with c1: st.metric("Neto 1º → Braga leva 2ª",
                        f"{resultado['p2n'].get('Eduardo Braga',0):.1f}%")
    with c2: st.metric("Braga 1º → Neto leva 2ª",
                        f"{resultado['p2b'].get('Alberto Neto',0):.1f}%")
    fig2,axes2 = plt.subplots(1,2,figsize=(16,7),facecolor=FUNDO)
    for ax5,sims_c,p_2a,lider,excluir,n_c in [
        (axes2[0],resultado['sn'],resultado['p2n'],'Alberto Neto','Alberto Neto',resultado['n_sn']),
        (axes2[1],resultado['sb'],resultado['p2b'],'Eduardo Braga','Eduardo Braga',resultado['n_sb'])]:
        setup_ax(ax5)
        for idx,c in enumerate(CANDIDATOS):
            if c==excluir or sims_c.shape[0]==0: continue
            vals=sims_c[:,idx]
            if vals.std()<0.3: continue
            xr=np.linspace(max(0,vals.min()-1),min(50,vals.max()+1),300)
            try:
                kde=gaussian_kde(vals,bw_method=0.18); yk=kde(xr)
                ax5.fill_between(xr,yk,color=CORES[c],alpha=0.22)
                ax5.plot(xr,yk,color=CORES[c],lw=2.0,label=f'{c} ({PARTIDOS[c]})')
                med=np.median(vals)
                ax5.axvline(med,color=CORES[c],lw=1.3,ls='--',alpha=0.8)
                ax5.text(med+0.4,max(yk)*0.5,f'{med:.1f}%',color=CORES[c],fontsize=8.5,va='center')
            except: pass
        txt="\n".join([f"P({c}) = {p_2a[c]:.1f}%"
                       for c in sorted(p_2a,key=lambda x:p_2a[x],reverse=True)[:4]])
        ax5.text(0.97,0.97,txt,transform=ax5.transAxes,color=ACENTO,fontsize=10.5,
                 fontweight='bold',va='top',ha='right',
                 bbox=dict(boxstyle='round,pad=0.55',facecolor=PAINEL,edgecolor=ACENTO,alpha=0.95,lw=0.9))
        ax5.set_xlim(0,50); ax5.set_xticks(range(0,51,5))
        ax5.set_xticklabels([f'{v}%' for v in range(0,51,5)],color=SUBTXT,fontsize=9)
        ax5.set_ylabel('Densidade',color=SUBTXT); ax5.set_xlabel('Intenção de voto (%)',color=SUBTXT)
        ax5.legend(fontsize=8.5,facecolor=PAINEL,edgecolor=GRADE,labelcolor=TEXTO,framealpha=0.92,loc='upper left')
        ax5.set_title(f'2ª vaga | {lider} em 1º ({n_c:,} sim.)',color=TEXTO,fontsize=11,fontweight='bold',pad=8)
    fig2.suptitle('Disputa pela 2ª vaga',color=TEXTO,fontsize=13,fontweight='bold',y=1.01)
    plt.tight_layout(); st.pyplot(fig2,use_container_width=True)

# ─── ABA 3 — CHAT ANALÍTICO ───────────────────────────────────
with tab3:
    st.subheader("💬 Chat analítico")
    st.caption(
        "Faça perguntas sobre as pesquisas, candidatos, tendências e metodologia. "
        "O assistente tem acesso completo aos dados do modelo atual."
    )

    if not api_key:
        st.warning("Cole sua chave da API Anthropic na barra lateral para usar o chat.")
    elif not ANTHROPIC_OK:
        st.error("Instale o pacote anthropic: `pip install anthropic`")
    else:
        # Sugestões de perguntas
        st.markdown("**Sugestões:**")
        sugestoes = [
            "Qual pesquisa é mais confiável e por quê?",
            "Como está a tendência de Braga nas últimas pesquisas?",
            "Qual instituto favorece mais o Neto?",
            "Quais candidatos têm trajetória crescente?",
            "Explica a diferença entre as pesquisas presenciais e online",
            "O que o modelo diz sobre a disputa pela 2ª vaga?",
        ]
        cols_sug = st.columns(3)
        for i, sug in enumerate(sugestoes):
            with cols_sug[i % 3]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state['pergunta_rapida'] = sug

        st.markdown("---")

        # Histórico do chat
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"],
                                     avatar="🗳️" if msg["role"]=="assistant" else "👤"):
                    st.markdown(msg["content"])

        # Input do usuário
        pergunta_inicial = st.session_state.pop('pergunta_rapida', '') \
            if 'pergunta_rapida' in st.session_state else ''

        pergunta = st.chat_input("Faça uma pergunta sobre as pesquisas ou candidatos...")

        # Usar sugestão clicada ou pergunta digitada
        mensagem = pergunta or pergunta_inicial

        if mensagem:
            # Mostrar mensagem do usuário
            with st.chat_message("user", avatar="👤"):
                st.markdown(mensagem)
            st.session_state.chat_history.append(
                {"role":"user","content":mensagem}
            )

            # Gerar resposta
            with st.chat_message("assistant", avatar="🗳️"):
                with st.spinner("Analisando..."):
                    contexto = montar_contexto(ativas, resultado)
                    resposta = chat_com_claude(
                        mensagem,
                        st.session_state.chat_history[:-1],
                        contexto,
                        api_key
                    )
                st.markdown(resposta)

            st.session_state.chat_history.append(
                {"role":"assistant","content":resposta}
            )
            st.rerun()

        # Limpar histórico
        if st.session_state.chat_history:
            if st.button("🗑️ Limpar conversa", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

# ─── ABA 4 — UPLOAD ───────────────────────────────────────────
with tab4:
    st.subheader("📄 Upload de pesquisa")
    st.markdown("Faça upload do PDF ou imagem. O Claude extrai os dados automaticamente.")

    if not api_key:
        st.warning("Cole sua chave da API Anthropic na barra lateral.")
    else:
        arquivo = st.file_uploader(
            "Selecione o arquivo",
            type=["pdf","png","jpg","jpeg","webp"],
        )

        if arquivo:
            mime_map = {"pdf":"application/pdf","png":"image/png",
                        "jpg":"image/jpeg","jpeg":"image/jpeg","webp":"image/webp"}
            ext  = arquivo.name.split(".")[-1].lower()
            mime = mime_map.get(ext,"image/png")

            if mime != "application/pdf":
                st.image(arquivo, caption="Pré-visualização", width=500)
            else:
                st.info(f"📄 {arquivo.name} ({arquivo.size/1024:.0f} KB)")

            if st.button("🤖 Extrair dados com Claude", type="primary"):
                if not ANTHROPIC_OK:
                    st.error("Instale: `pip install anthropic`")
                else:
                    with st.spinner("Claude está lendo... (pode levar até 15s para PDFs grandes)"):
                        try:
                            dados = extrair_pesquisa_com_claude(
                                arquivo.read(), mime, api_key
                            )
                            st.session_state['extracao_pendente'] = dados
                            st.success("Dados extraídos! Confirme abaixo.")
                        except ValueError as e:
                            st.error(f"❌ {e}")
                            st.markdown("**Possíveis soluções:**")
                            st.markdown("- PDF com muitas páginas: recorte só a página do Senado e suba como imagem (PNG)")
                            st.markdown("- Arquivo ilegível: tente uma imagem com melhor resolução")
                            st.markdown("- Chave de API inválida: verifique na barra lateral")
                            st.markdown("- Alternativa: use a aba ➕ para inserir manualmente")
                        except Exception as e:
                            err = str(e)
                            if "invalid_api_key" in err or "authentication" in err.lower():
                                st.error("❌ Chave de API inválida. Verifique na barra lateral.")
                            elif "rate_limit" in err.lower():
                                st.error("❌ Limite de requisições. Aguarde 1 minuto e tente novamente.")
                            elif "overloaded" in err.lower():
                                st.error("❌ API sobrecarregada. Tente em alguns segundos.")
                            else:
                                st.error(f"❌ Erro: {e}")
                                st.markdown("Use a aba **➕ Inserir manual** como alternativa.")

    if 'extracao_pendente' in st.session_state:
        dados = st.session_state['extracao_pendente']
        st.markdown("---")
        st.subheader("✅ Confirme os dados extraídos")
        col_info, col_votos = st.columns(2)
        with col_info:
            st.markdown(f"**Instituto:** {dados.get('nome','?')}")
            st.markdown(f"**Campo:** {dados.get('campo','?')}")
            st.markdown(f"**n:** {dados.get('n',0):,}")
            st.markdown(f"**Margem:** ±{dados.get('margem','?')}pp")
            st.markdown(f"**Método:** {dados.get('metodo','?')}")
            if dados.get('observacao'):
                st.info(dados['observacao'])
        with col_votos:
            st.markdown("**Intenção de voto:**")
            for c in CANDIDATOS:
                pct = dados.get(c,0.0)
                if pct > 0:
                    st.markdown(f"`{c}` **{pct:.1f}%** {'█'*int(pct/2)}")

        col_conf, col_cancel = st.columns(2)
        with col_conf:
            if st.button("✅ Confirmar e adicionar", type="primary", use_container_width=True):
                nomes = [p['nome'] for p in st.session_state.pesquisas]
                nome_novo = dados.get('nome','Nova')
                if nome_novo in nomes: nome_novo += " (2)"
                nova = {k:v for k,v in dados.items() if k!='observacao'}
                nova['nome'] = nome_novo
                for c in CANDIDATOS:
                    if c not in nova: nova[c] = 0.0
                st.session_state.pesquisas.append(nova)
                st.session_state.ativas[nome_novo] = True
                del st.session_state['extracao_pendente']
                st.success(f"✅ **{nome_novo}** adicionada!")
                st.rerun()
        with col_cancel:
            if st.button("❌ Cancelar", use_container_width=True):
                del st.session_state['extracao_pendente']
                st.rerun()

# ─── ABA 5 — DADOS ────────────────────────────────────────────
with tab5:
    st.subheader("Pesquisas no modelo")
    rows = []
    for p in st.session_state.pesquisas:
        ativo = st.session_state.ativas.get(p['nome'],True)
        row = {'Instituto':p['nome'],'Campo':p['campo'],'n':p['n'],
               'Margem':f"±{p['margem']}pp",'Método':p['metodo'],
               'Status':'✅' if ativo else '❌',
               'Peso':f"{1/(p['margem']/1.96)**2:.2f}×"}
        for c in CANDIDATOS: row[c.split()[0]] = f"{p.get(c,0):.1f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Médias ponderadas")
    mu_rows = [{'Candidato':c,'Partido':PARTIDOS[c],
                'Média (%)':f"{resultado['MU'][c]:.1f}",
                'Sigma (pp)':f"{resultado['SIGMA'][c]:.2f}",
                'P(eleito)':f"{resultado['p_el'][c]:.1f}%"}
               for c in sorted(CANDIDATOS, key=lambda x:resultado['MU'][x], reverse=True)]
    st.dataframe(pd.DataFrame(mu_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💾 Exportar")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        json_str = json.dumps(st.session_state.pesquisas, ensure_ascii=False, indent=2)
        st.download_button("⬇️ Baixar pesquisas (JSON)", data=json_str,
                           file_name="pesquisas_senado_am_2026.json",
                           mime="application/json")
    with col_exp2:
        chat_str = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
        st.download_button("⬇️ Baixar histórico do chat (JSON)", data=chat_str,
                           file_name="chat_senado_am_2026.json",
                           mime="application/json")

st.markdown("---")
st.caption("Modelo Monte Carlo | Senado AM 2026 | Média ponderada por 1/variância")
