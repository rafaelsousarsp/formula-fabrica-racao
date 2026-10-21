"""
Simulador de ração de mínimo custo — RSP Consultoria (projeto de portfólio, dados 100% fictícios).
Rodar:  pip install -r requirements.txt   e   streamlit run app.py
"""
import os, base64
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from scipy.optimize import linprog

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(AQUI, "Base_Fabrica_Racao.xlsx")
LOGO = os.path.join(AQUI, "logo_rsp.png")
NUTR = [("em_kcal_kg", "em_min_kcal_kg", "Energia metabolizável", "kcal/kg"), ("pb_pct", "pb_min_pct", "Proteína bruta", "%"),
        ("lis_dig_pct", "lis_dig_min_pct", "Lisina digestível", "%"), ("met_cis_dig_pct", "met_cis_dig_min_pct", "Met+Cis digestível", "%"),
        ("thr_dig_pct", "thr_dig_min_pct", "Treonina digestível", "%"), ("ca_pct", "ca_min_pct", "Cálcio", "%"),
        ("p_disp_pct", "p_disp_min_pct", "Fósforo disponível", "%"), ("na_pct", "na_min_pct", "Sódio", "%")]
COR = {"Milho grão": "#E0A800", "Farelo de soja 46%": "#3F7D3A", "Óleo de soja": "#F2C76E", "Farelo de trigo": "#B98B5A",
       "DDG de milho": "#8B5E34", "Calcário calcítico": "#C9CDD2", "Fosfato bicálcico": "#9AA3AD", "Sal comum": "#6B7580",
       "L-Lisina HCl 78%": "#3485BB", "DL-Metionina 99%": "#5BA3D0", "L-Treonina 98%": "#8CC0E3", "Premix vitamínico-mineral": "#1C1D21"}
AZUL, GRAF, OURO, VERDE, VERM = "#3485BB", "#1C1D21", "#D4A017", "#2F6B3B", "#B5452E"
br = lambda v, d=2: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")

st.set_page_config(page_title="Ração de mínimo custo · RSP", page_icon="🌽", layout="wide")
st.markdown(f"""<style>
html, body, [class*="css"] {{font-family: "Segoe UI", system-ui, sans-serif;}}
.block-container {{padding-top: 1.2rem; max-width: 1400px;}}
.hdr {{background:{GRAF}; border-bottom:4px solid {OURO}; border-radius:14px; padding:14px 22px; display:flex; align-items:center; gap:16px; margin-bottom:14px}}
.hdr h1 {{color:#fff; font-size:1.55rem; margin:0; font-weight:700}} .hdr p {{color:#C9CDD2; margin:0; font-size:.9rem}}
.kpi {{background:#fff; border:1px solid #DDE3EA; border-radius:14px; padding:14px 16px; height:100%}}
.kpi .l {{color:#6B6F76; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em}} .kpi .v {{font-size:1.75rem; font-weight:700; color:{GRAF}}}
.kpi .s {{font-size:.85rem}}
.frase {{background:#EEF5FB; border:1px solid #C7DDF0; border-left:5px solid {AZUL}; border-radius:12px; padding:12px 16px; margin:10px 0 4px; color:{GRAF}}}
.rod {{color:#8A8F86; font-size:.8rem; margin-top:18px}}
</style>""", unsafe_allow_html=True)

@st.cache_data
def carregar():
    x = pd.read_excel(BASE, sheet_name=None)
    return x["dIngredientes"], x["dExigencias"], x["fPrecosIngredientes"], x["fProducao"]
ING, EXIG, PRECOS, PROD = carregar()

def otimizar(precos, e):
    n = len(ING); A_ub, b_ub = [], []
    for ci, ce, *_ in NUTR: A_ub.append(-ING[ci].values); b_ub.append(-e[ce])
    A_ub.append(ING["ca_pct"].values); b_ub.append(e["ca_min_pct"] * 1.6)
    bounds = [(0.004, 0.004) if r.ingrediente.startswith("Premix") else (0, r.inclusao_max_pct / 100) for _, r in ING.iterrows()]
    r = linprog(np.asarray(precos, float), A_ub=np.array(A_ub), b_ub=b_ub, A_eq=[np.ones(n)], b_eq=[1.0], bounds=bounds, method="highs")
    if not r.success: return None, None, None
    return r.x, float(r.x @ np.asarray(precos, float) * 1000), -r.ineqlin.marginals

# ---------------- barra lateral ----------------
with st.sidebar:
    st.markdown("### Parâmetros")
    fase = st.selectbox("Fase", list(EXIG.fase), index=3)
    meses = sorted(PRECOS.mes.unique()); mes = st.selectbox("Mês de preços", meses, index=len(meses) - 1, format_func=lambda m: pd.Timestamp(m).strftime("%m/%Y"))
    pm = PRECOS[PRECOS.mes == mes].set_index("ingrediente").preco_rs_kg
    base = ING.ingrediente.map(pm).values.astype(float)
    st.markdown("### Preços (variação sobre o mês)")
    vm = st.slider("Milho grão", -30, 30, 0, 1, format="%d%%")
    vf = st.slider("Farelo de soja 46%", -30, 30, 0, 1, format="%d%%")
    vo = st.slider("Óleo de soja", -30, 30, 0, 1, format="%d%%")
    ca = st.number_input("Conversão alimentar (kg ração / kg suíno)", 2.0, 3.2, 2.45, 0.05)
    st.caption("CA 2,45 = média da granja do projeto 05.")

idx = {n: i for i, n in enumerate(ING.ingrediente)}
precos = base.copy()
for nome, v in (("Milho grão", vm), ("Farelo de soja 46%", vf), ("Óleo de soja", vo)): precos[idx[nome]] *= 1 + v / 100
e = EXIG[EXIG.fase == fase].iloc[0]
x0, c0, _ = otimizar(base, e)
x1, c1, sombra = otimizar(precos, e)

logo = f'<img src="data:image/png;base64,{base64.b64encode(open(LOGO, "rb").read()).decode()}" style="height:52px;border-radius:10px">' if os.path.exists(LOGO) else ""
st.markdown(f'<div class="hdr">{logo}<div><h1>Ração de mínimo custo</h1><p>Programação linear por fase · {fase} ({e.faixa_peso}) · preços de {pd.Timestamp(mes).strftime("%m/%Y")}</p></div></div>', unsafe_allow_html=True)
if x1 is None:
    st.error("Sem solução viável com esses parâmetros."); st.stop()

d = c1 - c0
k = st.columns(4)
def kpi(col, l, v, s="", cor="#6B6F76"): col.markdown(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s" style="color:{cor}">{s}</div></div>', unsafe_allow_html=True)
kpi(k[0], "Custo da fórmula", f"R$ {br(c1, 0)}/t", f"{'▲' if d > 0 else '▼' if d < 0 else '■'} R$ {br(abs(d), 2)}/t vs. preços do mês", VERM if d > 0.005 else VERDE if d < -0.005 else "#6B6F76")
kpi(k[1], "Custo de ração por kg de suíno", f"R$ {br(c1 * ca / 1000)}", f"CA {br(ca)} · Δ R$ {br(d * ca / 1000, 3)}/kg")
kpi(k[2], "Milho na fórmula", f"{br(x1[idx['Milho grão']] * 100, 1)}%", f"R$ {br(precos[idx['Milho grão']])}/kg")
kpi(k[3], "Farelo de soja na fórmula", f"{br(x1[idx['Farelo de soja 46%']] * 100, 1)}%", f"R$ {br(precos[idx['Farelo de soja 46%']])}/kg")

# sensibilidade ±10% (sobre os preços do simulador)
tor = []
for nome in ING.ingrediente:
    if nome.startswith("Premix"): continue
    ds = []
    for v in (-0.1, 0.1):
        p2 = precos.copy(); p2[idx[nome]] *= 1 + v
        ds.append(otimizar(p2, e)[1] - c1)
    tor.append((nome, ds[0], ds[1]))
tor = pd.DataFrame(tor, columns=["ing", "menos", "mais"]).assign(a=lambda t: t.mais.abs() + t.menos.abs()).sort_values("a")
top = tor.iloc[-1]
ativos = [(NUTR[j][2], sombra[j] * e[NUTR[j][1]] * 0.01 * 1000) for j in range(len(NUTR)) if sombra[j] > 1e-6]
cara = max(ativos, key=lambda t: t[1]) if ativos else ("—", 0)
st.markdown(f'<div class="frase">Se o <b>{top.ing.lower()}</b> subir 10%, a ração de {fase} sobe <b>R$ {br(top.mais)}/t</b> '
            f'e o custo do suíno sobe <b>R$ {br(top.mais * ca / 1000, 3)}/kg</b>. A exigência que mais pesa no custo é <b>{cara[0].lower()}</b>: '
            f'subir 1% custa R$ {br(cara[1])}/t.</div>', unsafe_allow_html=True)

c_a, c_b = st.columns([1.15, 1])
with c_a:
    st.markdown("##### Fórmula ótima")
    f = pd.DataFrame({"ing": ING.ingrediente, "pct": x1 * 100}).query("pct > 0.001").sort_values("pct")
    fig = go.Figure(go.Bar(x=f.pct, y=f.ing, orientation="h", marker_color=[COR[i] for i in f.ing],
                           text=[f"{br(v, 2 if v < 1 else 1)}%" for v in f.pct], textposition="outside"))
    fig.update_layout(height=380, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(visible=False, range=[0, f.pct.max() * 1.18]),
                      plot_bgcolor="#fff", paper_bgcolor="#fff", font=dict(family="Segoe UI", size=13))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with c_b:
    st.markdown("##### Nutrientes: atingido × exigência")
    rows = []
    for (ci, ce, nm, un), j in zip(NUTR, range(len(NUTR))):
        lv = float(ING[ci].values @ x1); rows.append((nm, lv / e[ce] * 100, lv, e[ce], un))
    n = pd.DataFrame(rows, columns=["n", "pct", "lv", "ex", "un"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=n.pct, y=n.n, orientation="h", marker_color=[AZUL if p < 100.5 else "#9CC3E0" for p in n.pct],
                           text=[f"{br(p, 0)}%" for p in n.pct], textposition="outside",
                           hovertext=[f"{br(a, 2)} / mín. {br(b, 2)} {u}" for a, b, u in zip(n.lv, n.ex, n.un)]))
    fig.add_vline(x=100, line_dash="dash", line_color=GRAF)
    fig.update_layout(height=380, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, max(130, n.pct.max() * 1.1)], ticksuffix="%"),
                      plot_bgcolor="#fff", paper_bgcolor="#fff", font=dict(family="Segoe UI", size=13))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Barra azul-escura em 100% = restrição ativa (é ela que limita o custo).")

c_c, c_d = st.columns([1.15, 1])
with c_c:
    st.markdown("##### Sensibilidade: ±10% no preço de cada ingrediente (R$/t)")
    fig = go.Figure([go.Bar(y=tor.ing, x=tor.menos, orientation="h", name="−10%", marker_color=VERDE, text=[br(v) for v in tor.menos], textposition="outside"),
                     go.Bar(y=tor.ing, x=tor.mais, orientation="h", name="+10%", marker_color=VERM, text=[br(v) for v in tor.mais], textposition="outside")])
    lim = max(tor.a.max() / 2, 1) * 1.35
    fig.update_layout(barmode="overlay", height=420, margin=dict(l=0, r=10, t=10, b=0), xaxis=dict(range=[-lim, lim]),
                      legend=dict(orientation="h", y=1.08), plot_bgcolor="#fff", paper_bgcolor="#fff", font=dict(family="Segoe UI", size=13))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with c_d:
    st.markdown("##### Preço sombra das exigências")
    ps = pd.DataFrame([(NUTR[j][2], f"{br(e[NUTR[j][1]], 2)} {NUTR[j][3]}", sombra[j] * e[NUTR[j][1]] * 0.01 * 1000) for j in range(len(NUTR))],
                      columns=["Exigência", "Mínimo", "R$/t para +1%"]).sort_values("R$/t para +1%", ascending=False)
    ps["R$/t para +1%"] = ps["R$/t para +1%"].map(lambda v: f"R$ {br(v)}")
    st.dataframe(ps, hide_index=True, use_container_width=True)
    st.caption("Quanto a tonelada encarece se a exigência mínima subir 1%. Zero = a fórmula já sobra nesse nutriente.")
    tab = pd.DataFrame({"Ingrediente": ING.ingrediente, "R$/kg": precos, "Inclusão %": x1 * 100}).query("`Inclusão %` > 0.001")
    tab["R$/t na fórmula"] = tab["R$/kg"] * tab["Inclusão %"] * 10
    st.dataframe(tab.style.format({"R$/kg": lambda v: br(v), "Inclusão %": lambda v: br(v, 2), "R$/t na fórmula": lambda v: br(v)}), hide_index=True, use_container_width=True)

st.markdown('<div class="rod">Dados fictícios para demonstração · valores nutricionais didáticos (validar com Tabelas Brasileiras e nutricionista) · RSP Consultoria</div>', unsafe_allow_html=True)
