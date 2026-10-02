"""Football Player Analytics & Scouting — Streamlit front end.

Run:  streamlit run app.py

Nothing is trained here. The notebook (football_ml.ipynb) writes models/artifacts.joblib
and models/players_scored.csv; this app only loads and looks things up.
"""
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODELS = Path(__file__).parent / "models"

st.set_page_config(page_title="Football Analytics", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------- palette / theme
INK, MUTED, LINE = "#0F1115", "#6B7280", "#E6E7E0"
BG, CARD = "#F6F7F1", "#FFFFFF"
GREEN, COBALT, ORANGE = "#00A651", "#2457F5", "#FF7A00"
PROFILE_COLOR = {"Attacking": ORANGE, "Playmaking": COBALT, "Defensive": GREEN}
PROFILE_NOTE = {
    "Attacking": "High shot volume, low passing and defending. Mostly forwards.",
    "Playmaking": "Ball-playing profile — most passes and best pass accuracy. ~69% are defenders, so read it as build-up passers, not only creative midfielders.",
    "Defensive": "Most tackles and strong interception numbers. A defender / midfielder mix.",
}
LAB = {"Shots_p90": "Shots / 90", "Passes_p90": "Passes / 90", "PassAcc": "Pass accuracy",
       "Tackles_p90": "Tackles / 90", "Interceptions_p90": "Interceptions / 90"}


def css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --green:{GREEN}; --cobalt:{COBALT}; --orange:{ORANGE}; }}
header[data-testid="stHeader"], #MainMenu, footer {{ display:none; }}
.stApp {{ background:{BG}; }}
html, body, .stApp {{ font-family:'Inter',sans-serif; color:{INK}; }}
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown, .stApp button,
.stApp input, .stApp select, .stApp textarea {{ font-family:'Inter',sans-serif; }}
/* never override Streamlit's Material icon glyph font (expander chevron, checkboxes, etc.) */
[data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"] {{
   font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important; }}
.block-container {{ max-width:1180px; padding-top:1.2rem; padding-bottom:3rem; }}
h1,h2,h3 {{ font-family:'Archivo',sans-serif; letter-spacing:-.01em; }}

/* ---- top bar ---- */
.topbar {{ display:flex; align-items:center; gap:.6rem; padding:.2rem 0 .1rem; }}
.brand {{ font-family:'Archivo',sans-serif; font-weight:900; font-size:1.35rem; letter-spacing:-.02em; }}
.brand .dot {{ color:{GREEN}; }}
.brand .sub {{ font-family:'Inter'; font-weight:500; font-size:.72rem; color:{MUTED}; letter-spacing:.14em; text-transform:uppercase; margin-left:.1rem; }}
hr.rule {{ border:none; border-top:1px solid {LINE}; margin:.5rem 0 1.1rem; }}

/* ---- nav buttons (keys start with nav_) ---- */
[class*="st-key-nav_"] button {{ border-radius:999px !important; border:1px solid {LINE} !important;
   font-weight:700 !important; font-family:'Archivo',sans-serif !important; letter-spacing:.01em;
   background:{CARD} !important; color:{INK} !important; padding:.35rem 0 !important; box-shadow:none !important; }}
[class*="st-key-nav_"] button[kind="primary"] {{ background:{INK} !important; border-color:{INK} !important; }}
[class*="st-key-nav_"] button[kind="primary"], [class*="st-key-nav_"] button[kind="primary"] * {{ color:#fff !important; }}
[class*="st-key-nav_"] button:hover {{ border-color:{INK} !important; }}

/* ---- generic buttons (view / similar) ---- */
[class*="st-key-go_"] button, [class*="st-key-sim_"] button, [class*="st-key-sr_"] button {{
   border-radius:8px !important; border:1px solid {LINE} !important; background:{CARD} !important;
   color:{INK} !important; font-weight:600 !important; box-shadow:none !important; }}
[class*="st-key-go_"] button:hover, [class*="st-key-sim_"] button:hover, [class*="st-key-sr_"] button:hover {{
   border-color:{GREEN} !important; color:{GREEN} !important; }}

/* ---- text pieces ---- */
.eyebrow {{ font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:{MUTED}; }}
.hero-name {{ font-family:'Archivo'; font-weight:900; font-size:3rem; line-height:1; margin:.1rem 0 .3rem; }}
.meta {{ color:{MUTED}; font-size:.95rem; font-weight:500; }}
.section {{ font-family:'Archivo'; font-weight:800; font-size:1.25rem; margin:.2rem 0 .7rem; text-transform:uppercase; letter-spacing:.02em; }}
.note {{ color:{MUTED}; font-size:.8rem; font-style:italic; margin-top:.3rem; }}
.page-title {{ font-family:'Archivo'; font-weight:900; font-size:2.3rem; line-height:1.02; margin:.1rem 0 .1rem; }}
.page-sub {{ color:{MUTED}; font-size:1rem; margin-bottom:1rem; }}

/* ---- badges ---- */
.badge {{ display:inline-block; padding:.28rem .7rem; border-radius:999px; font-weight:700; font-size:.8rem;
   letter-spacing:.02em; margin-right:.4rem; }}
.b-high {{ background:rgba(0,166,81,.12); color:{GREEN}; }}
.b-low {{ background:rgba(107,114,128,.14); color:{MUTED}; }}

/* ---- cards / tiles ---- */
.card {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:1.1rem 1.2rem; height:100%; }}
.tile {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:.8rem .9rem; }}
.tile .k {{ font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; }}
.tile .v {{ font-family:'Archivo'; font-weight:800; font-size:1.5rem; margin-top:.1rem; }}
.tile .s {{ font-size:.75rem; color:{MUTED}; }}

/* ---- attribute bars ---- */
.attr {{ margin-bottom:.8rem; }}
.attr .top {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.28rem; }}
.attr .nm {{ font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:{INK}; }}
.attr .vl {{ font-family:'Archivo'; font-weight:800; font-size:1.05rem; }}
.attr .track {{ height:9px; background:#EFEFE9; border-radius:999px; overflow:hidden; }}
.attr .fill {{ height:100%; border-radius:999px; }}
.attr .pc {{ font-size:.72rem; color:{MUTED}; margin-top:.22rem; }}

/* ---- similar / result cards ---- */
.simcard {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:.75rem .9rem; margin-bottom:.1rem; }}
.simcard .nm {{ font-weight:800; font-family:'Archivo'; font-size:1.02rem; }}
.simcard .mt {{ color:{MUTED}; font-size:.8rem; margin-bottom:.4rem; }}
.simcard .track {{ height:7px; background:#EFEFE9; border-radius:999px; overflow:hidden; margin-top:.3rem; }}
.simcard .fill {{ height:100%; border-radius:999px; }}

.arche {{ background:{CARD}; border:1px solid {LINE}; border-top:5px solid var(--c); border-radius:16px; padding:1.1rem 1.2rem; height:100%; }}
.arche .nm {{ font-family:'Archivo'; font-weight:900; font-size:1.3rem; }}
.arche .ct {{ color:{MUTED}; font-size:.82rem; font-weight:600; margin-bottom:.5rem; }}
.arche .ds {{ font-size:.86rem; color:#333; min-height:3.4rem; }}
.arche .row {{ display:flex; justify-content:space-between; font-size:.83rem; padding:.2rem 0; border-top:1px solid {LINE}; }}
.arche .row b {{ font-family:'Archivo'; }}
div[data-testid="stDataFrame"] {{ border-radius:12px; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data / helpers
@st.cache_resource
def load_art():
    return joblib.load(MODELS / "artifacts.joblib")


@st.cache_data
def load_players():
    d = pd.read_csv(MODELS / "players_scored.csv")
    d["norm"] = d.Player.map(lambda s: "".join(c for c in unicodedata.normalize("NFKD", str(s))
                                              if not unicodedata.combining(c)).lower())
    return d


art = load_art()
df = load_players()
FEATURES = art["features"]
Xs = art["scaler_km"].transform(df[FEATURES])                     # standardised space (same as notebook)
PCTL = {f: df[f].rank(pct=True) * 100 for f in FEATURES}          # percentile rank per feature
SCORE_PCTL = df.Score.rank(pct=True) * 100


def pct_of(feature, value):
    return float((df[feature] < value).mean() * 100)


def search(query, limit=8):
    q = "".join(c for c in unicodedata.normalize("NFKD", query) if not unicodedata.combining(c)).lower().strip()
    if not q:
        return df.iloc[0:0]
    hit = df[df.norm.str.contains(q, regex=False)]
    return hit.sort_values("Score", ascending=False).head(limit)


def similar(pid, n=5):
    d = np.linalg.norm(Xs - Xs[pid], axis=1)
    order = np.argsort(d)[1:n + 1]
    sims = 100 * (1 - d[order] / d.max())
    return list(zip(order, sims))


def fnum(x, f):
    return f"{x:.1f}" if f == "PassAcc" else f"{x:.2f}"


# ---------------------------------------------------------------- visual builders
def gauge_html(score, pctl, is_high):
    frac = max(0.02, min(1.0, pctl / 100))
    r, C = 52, 2 * np.pi * 52
    dash = C * frac
    col = GREEN if is_high else MUTED
    return f"""
<div style="text-align:center;background:{CARD};border:1px solid {LINE};border-radius:16px;padding:1rem .6rem;">
  <div class="eyebrow" style="margin-bottom:.4rem;">Performance score</div>
  <svg width="150" height="150" viewBox="0 0 130 130">
    <circle cx="65" cy="65" r="{r}" fill="none" stroke="#EFEFE9" stroke-width="12"/>
    <circle cx="65" cy="65" r="{r}" fill="none" stroke="{col}" stroke-width="12" stroke-linecap="round"
      stroke-dasharray="{dash:.1f} {C:.1f}" transform="rotate(-90 65 65)"/>
    <text x="65" y="60" text-anchor="middle" font-family="Archivo" font-weight="900" font-size="30" fill="{INK}">{score:.2f}</text>
    <text x="65" y="80" text-anchor="middle" font-family="Inter" font-size="11" fill="{MUTED}">top {100 - pctl:.0f}%</text>
  </svg>
  <div style="margin-top:.4rem;">
    <span class="badge {'b-high' if is_high else 'b-low'}">{'HIGH PERFORMER' if is_high else 'LOW PERFORMER'}</span>
  </div>
</div>"""


def attr_bars_html(row):
    out = []
    for f in FEATURES:
        p = pct_of(f, row[f])
        col = GREEN if p >= 66 else (COBALT if p >= 33 else ORANGE)
        out.append(f"""
<div class="attr">
  <div class="top"><span class="nm">{LAB[f]}</span><span class="vl">{fnum(row[f], f)}{'%' if f=='PassAcc' else ''}</span></div>
  <div class="track"><div class="fill" style="width:{p:.0f}%;background:{col};"></div></div>
  <div class="pc">{p:.0f}th percentile</div>
</div>""")
    return "".join(out)


def radar_fig(row):
    vals = [pct_of(f, row[f]) for f in FEATURES]
    labels = [LAB[f].replace(" / 90", "/90") for f in FEATURES]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself",
        fillcolor="rgba(0,166,81,.18)", line=dict(color=GREEN, width=2.5),
        hovertemplate="%{theta}: %{r:.0f}th pct<extra></extra>"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor=LINE, linecolor=LINE),
                   angularaxis=dict(gridcolor=LINE, tickfont=dict(size=11, color=INK, family="Inter"))),
        showlegend=False, margin=dict(l=40, r=40, t=30, b=30), height=300,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
    return fig


def pca_fig(color_mode, highlight_pid=None):
    fig = go.Figure()
    cd = np.stack([df.Team, df.Position, df.Profile, df.Score], axis=-1)
    ht = "<b>%{customdata[0]}... </b>"  # replaced below
    if color_mode == "By profile":
        for prof, c in PROFILE_COLOR.items():
            s = df[df.Profile == prof]
            fig.add_scatter(x=s.PC1, y=s.PC2, mode="markers", name=prof,
                            marker=dict(size=6, color=c, opacity=.65, line=dict(width=0)),
                            customdata=np.stack([s.Player, s.Team, s.Position, s.Score], axis=-1),
                            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}"
                                          "<br>Score %{customdata[3]:.2f}<extra></extra>")
    else:
        fig.add_scatter(x=df.PC1, y=df.PC2, mode="markers",
                        marker=dict(size=6, color=df.Score, colorscale="Viridis", cmax=df.Score.quantile(.98),
                                    opacity=.7, colorbar=dict(title="Score", thickness=12, len=.7)),
                        customdata=np.stack([df.Player, df.Team, df.Position, df.Score], axis=-1),
                        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}"
                                      "<br>Score %{customdata[3]:.2f}<extra></extra>")
    if highlight_pid is not None:
        h = df.iloc[highlight_pid]
        fig.add_scatter(x=[h.PC1], y=[h.PC2], mode="markers+text", name=h.Player,
                          marker=dict(size=16, color=INK, symbol="star", line=dict(width=2, color="#fff")),
                          text=[h.Player], textposition="top center",
                          textfont=dict(family="Archivo", size=12, color=INK), hoverinfo="skip")
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                      legend=dict(orientation="h", y=1.08, x=0),
                      xaxis=dict(title="", showgrid=True, gridcolor=LINE, zeroline=False, showticklabels=False),
                      yaxis=dict(title="", showgrid=True, gridcolor=LINE, zeroline=False, showticklabels=False))
    return fig


def bar_fig(series, color, title):
    s = series.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[LAB.get(i, i) for i in s.index], orientation="h",
                           marker_color=color, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(family="Archivo", size=14)), height=260,
                      margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                      xaxis=dict(gridcolor=LINE, zeroline=True, zerolinecolor=LINE), yaxis=dict(ticksuffix="  "))
    return fig


def confusion_fig(y, pred, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=[1, 0])
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred High", "Pred Low"], y=["High", "Low"], colorscale="Greens",
                               text=cm, texttemplate="%{text}", showscale=False,
                               textfont=dict(size=18, family="Archivo")))
    fig.update_layout(title=dict(text=title, font=dict(family="Archivo", size=14)), height=260,
                      margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                      yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------- state + nav
css()
ss = st.session_state
ss.setdefault("page", "Players")
ss.setdefault("pid", int(df.index[df.Player == "Bukayo Saka"][0]) if (df.Player == "Bukayo Saka").any() else 0)

top = st.columns([3.2, 1, 1, 1, 1])
with top[0]:
    st.markdown('<div class="topbar"><span class="brand">FOOTBALL<span class="dot">.</span>SCOUT</span>'
                '<span class="sub">analytics & scouting</span></div>', unsafe_allow_html=True)
for col, label in zip(top[1:], ["Players", "Analytics", "Profiles", "Scouting"]):
    with col:
        if st.button(label, key=f"nav_{label}", use_container_width=True,
                     type="primary" if ss.page == label else "secondary"):
            ss.page = label
            st.rerun()
st.markdown('<hr class="rule">', unsafe_allow_html=True)


def go_player(pid):
    ss.pid = int(pid)
    ss.page = "Players"
    st.rerun()


# ================================================================= PLAYERS
if ss.page == "Players":
    q = st.text_input("Search players", placeholder="🔍  Search a player — try Saka, Rodri, Doku…",
                      label_visibility="collapsed", key="psearch")
    if q:
        res = search(q)
        if res.empty:
            st.info("No players found. Try a different name.")
        else:
            st.caption("Select a player:")
            cols = st.columns(2)
            for n, (idx, r) in enumerate(res.iterrows()):
                with cols[n % 2]:
                    if st.button(f"{r.Player} · {r.Team} · {r.Position}", key=f"sr_{idx}", use_container_width=True):
                        go_player(idx)
        st.markdown('<hr class="rule">', unsafe_allow_html=True)

    p = df.iloc[ss.pid]
    is_high = p.Pred_Category == "High"
    pcol = PROFILE_COLOR[p.Profile]

    st.markdown(f'<div class="eyebrow">Scouting report</div>'
                f'<div class="hero-name">{p.Player}</div>'
                f'<div class="meta">{p.Team} · {p.League} · {p.Position} · {int(p.Minutes):,} minutes · age {str(p.Age).split("-")[0]}</div>',
                unsafe_allow_html=True)
    st.write("")

    a, b, c = st.columns([1, 1.15, 1])
    with a:
        st.markdown(gauge_html(p.Score, float(SCORE_PCTL.iloc[ss.pid]), is_high), unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="eyebrow" style="text-align:center;margin-bottom:.2rem;">Profile shape</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(radar_fig(p), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c:
        st.markdown(f"""<div class="card">
          <div class="eyebrow">Player profile</div>
          <div style="font-family:Archivo;font-weight:900;font-size:1.6rem;color:{pcol};margin:.1rem 0 .2rem;">{p.Profile.upper()}</div>
          <div class="note" style="margin-top:0;">{PROFILE_NOTE[p.Profile]}</div>
          <div class="row" style="margin-top:.9rem;border-top:1px solid {LINE};padding-top:.7rem;">
            <div class="tile" style="border:none;padding:0;">
              <div class="k">Model prediction</div><div class="v">{p.Pred_Score:.2f}</div>
              <div class="s">Random Forest</div></div>
          </div>
          <div style="margin-top:.7rem;"><span class="k" style="font-size:.7rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:{MUTED};">Actual category</span>
            <div style="font-family:Archivo;font-weight:800;font-size:1.1rem;">{'High' if p.Label==1 else 'Low'}</div></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="note">Performance score = goals + assists per 90. These build the target, so the models never '
                'use them as inputs (avoids leakage). The ring shows where this score ranks among all players.</div>',
                unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section">Key attributes</div>', unsafe_allow_html=True)
        st.markdown(attr_bars_html(p), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section">Similar players</div>', unsafe_allow_html=True)
        for idx, sim in similar(ss.pid, 5):
            r = df.iloc[idx]
            cc = PROFILE_COLOR[r.Profile]
            st.markdown(f"""<div class="simcard">
              <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <span class="nm">{r.Player}</span>
                <span style="font-family:Archivo;font-weight:800;color:{cc};">{sim:.1f}%</span></div>
              <div class="mt">{r.Team} · {r.Position} · {r.Profile}</div>
              <div class="track"><div class="fill" style="width:{sim:.0f}%;background:{cc};"></div></div>
            </div>""", unsafe_allow_html=True)
            if st.button("View player →", key=f"sim_{idx}", use_container_width=True):
                go_player(idx)

    st.write("")
    with st.expander("WHAT IF?  —  adjust this player's stats and watch the models respond"):
        st.caption("Scenario values — not the player's real statistics. The models re-predict live.")
        wi, sc = {}, st.columns(5)
        for col, f in zip(sc, FEATURES):
            lo, hi = float(df[f].min()), float(df[f].quantile(0.995))
            wi[f] = col.slider(LAB[f], lo, round(hi, 1), float(min(max(p[f], lo), hi)), key=f"wi_{f}")
        row = pd.DataFrame([wi])[FEATURES]
        rf = art["reg_models"]["Random Forest"].predict(row)[0]
        lr = art["reg_models"]["Linear Regression"].predict(row)[0]
        prob = art["clf_models"]["Logistic Regression"].predict_proba(row)[0, 1]
        cl = art["cluster_names"][int(art["kmeans"].predict(art["scaler_km"].transform(row))[0])]
        st.markdown('<div class="section" style="font-size:1rem;margin-top:.6rem;">Model response</div>', unsafe_allow_html=True)
        m = st.columns(4)
        for col, (k, v, s) in zip(m, [("Random Forest", f"{rf:.2f}", f"was {p.Pred_Score:.2f}"),
                                      ("Linear Regression", f"{lr:.2f}", "score"),
                                      ("P(High performer)", f"{prob:.0%}", "logistic"),
                                      ("Cluster", cl, "k-means")]):
            col.markdown(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>',
                         unsafe_allow_html=True)

# ================================================================= ANALYTICS
elif ss.page == "Analytics":
    st.markdown('<div class="page-title">Model performance</div>'
                '<div class="page-sub">How the trained models do — shown with both a single test split and 5-fold '
                'cross-validation, so no model is judged on one lucky split. This is the supporting evidence behind the predictions.</div>',
                unsafe_allow_html=True)
    cv, reg, clf = art["cv_results"], art["reg_results"], art["clf_results"]

    st.markdown('<div class="section">Regression · predict the score</div>', unsafe_allow_html=True)
    rc = st.columns(2)
    for col, mname in zip(rc, reg.index):
        cvm = cv.loc[("Regression (R2)", mname)]
        col.markdown(f"""<div class="card"><div class="eyebrow">{mname}</div>
          <div style="font-family:Archivo;font-weight:900;font-size:2.4rem;">R² {reg.loc[mname,'R2']:.3f}</div>
          <div class="meta">CV {cvm['CV mean']:.3f} ± {cvm['CV std']:.3f}</div>
          <div class="row" style="display:flex;gap:1.4rem;margin-top:.6rem;">
            <span><b style="font-family:Archivo;">{reg.loc[mname,'MAE']:.3f}</b><br><span class="note" style="font-style:normal;">MAE</span></span>
            <span><b style="font-family:Archivo;">{reg.loc[mname,'RMSE']:.3f}</b><br><span class="note" style="font-style:normal;">RMSE</span></span>
          </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="note">Test performance varies by split; 5-fold CV suggests the two regressors are broadly '
                'comparable, so neither is declared the winner.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section">Classification · High vs Low</div>', unsafe_allow_html=True)
    cc = st.columns(2)
    for col, mname in zip(cc, clf.index):
        cvm = cv.loc[("Classification (Accuracy)", mname)]
        col.markdown(f"""<div class="card"><div class="eyebrow">{mname}</div>
          <div style="font-family:Archivo;font-weight:900;font-size:2.4rem;">{clf.loc[mname,'Accuracy']*100:.1f}%</div>
          <div class="meta">accuracy · CV {cvm['CV mean']*100:.1f}% ± {cvm['CV std']*100:.1f}%</div>
          <div class="row" style="display:flex;gap:1.4rem;margin-top:.6rem;">
            <span><b style="font-family:Archivo;">{clf.loc[mname,'Precision']:.2f}</b><br><span class="note" style="font-style:normal;">Precision</span></span>
            <span><b style="font-family:Archivo;">{clf.loc[mname,'Recall']:.2f}</b><br><span class="note" style="font-style:normal;">Recall</span></span>
            <span><b style="font-family:Archivo;">{clf.loc[mname,'F1']:.2f}</b><br><span class="note" style="font-style:normal;">F1</span></span>
          </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="note">Logistic Regression is slightly ahead of the Decision Tree on both cross-validation and '
                'test accuracy.</div>', unsafe_allow_html=True)

    st.write("")
    with st.expander("Explainability — what drives the prediction"):
        imp = art["importances"]
        g = st.columns(2)
        g[0].plotly_chart(bar_fig(imp["Random Forest importance"], GREEN, "Random Forest feature importance"),
                          use_container_width=True, config={"displayModeBar": False})
        g[1].plotly_chart(bar_fig(imp["Linear coefficient (scaled)"], COBALT, "Linear Regression coefficients"),
                          use_container_width=True, config={"displayModeBar": False})
        tp = art["test_preds"]
        h = st.columns(2)
        for col, (name, pr) in zip(h, tp["clf"].items()):
            col.plotly_chart(confusion_fig(tp["y_clf"], pr, name), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="note">Position strongly shapes these results (forwards shoot and score more), so part of the '
                    'signal is "is this a forward?" rather than pure quality — a limit of the five-stat feature set.</div>',
                    unsafe_allow_html=True)

# ================================================================= PROFILES
elif ss.page == "Profiles":
    st.markdown('<div class="page-title">Player archetypes</div>'
                '<div class="page-sub">K-Means grouped players without seeing labels or positions. We named the groups '
                'afterwards from their measured statistics.</div>', unsafe_allow_html=True)
    prof = art["cluster_profile"].rename(index=art["cluster_names"])
    posmix = pd.crosstab(df.Profile, df.Position, normalize="index") * 100

    cols = st.columns(3)
    for col, name in zip(cols, ["Attacking", "Playmaking", "Defensive"]):
        c = PROFILE_COLOR[name]
        rows = "".join(f'<div class="row"><span>{LAB[f]}</span><b>{fnum(prof.loc[name,f],f)}{"%" if f=="PassAcc" else ""}</b></div>'
                       for f in FEATURES)
        top_pos = posmix.loc[name].idxmax()
        col.markdown(f"""<div class="arche" style="--c:{c};">
          <div class="nm" style="color:{c};">{name.upper()}</div>
          <div class="ct">{int(prof.loc[name,'Players'])} players · mostly {top_pos}s ({posmix.loc[name,top_pos]:.0f}%)</div>
          <div class="ds">{PROFILE_NOTE[name]}</div>
          <div style="margin-top:.6rem;">{rows}</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section">Player map</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub" style="margin-top:-.3rem;">Every player placed by statistical similarity (PCA). '
                f'The first two components explain {100*art["pca"].explained_variance_ratio_.sum():.1f}% of the variance. Hover any point.</div>',
                unsafe_allow_html=True)
    mode = st.radio("Colour by", ["By profile", "By performance"], horizontal=True, label_visibility="collapsed")
    star = ss.pid if st.checkbox(f"Highlight current player ({df.iloc[ss.pid].Player})", value=True) else None
    st.plotly_chart(pca_fig(mode, star), use_container_width=True, config={"displayModeBar": False})

    with st.expander("How many groups? (elbow & silhouette)"):
        k = art["k_table"]
        gg = st.columns(2)
        f1 = go.Figure(go.Scatter(x=k.K, y=k.Inertia, mode="lines+markers", line=dict(color=INK)))
        f1.update_layout(title=dict(text="Elbow (inertia)", font=dict(family="Archivo", size=14)), height=260,
                         margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Inter"), xaxis=dict(title="K", gridcolor=LINE), yaxis=dict(gridcolor=LINE))
        f2 = go.Figure(go.Scatter(x=k.K, y=k.Silhouette, mode="lines+markers", line=dict(color=ORANGE)))
        f2.add_vline(x=art["k"], line_dash="dash", line_color=GREEN)
        f2.update_layout(title=dict(text="Silhouette", font=dict(family="Archivo", size=14)), height=260,
                         margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Inter"), xaxis=dict(title="K", gridcolor=LINE), yaxis=dict(gridcolor=LINE))
        gg[0].plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})
        gg[1].plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="note">K = {art["k"]} chosen as a useful level of detail. Silhouette ≈ '
                    f'{k.loc[k.K==art["k"],"Silhouette"].iloc[0]:.2f} is low, so the groups overlap — they are tendencies, not hard boundaries.</div>',
                    unsafe_allow_html=True)

# ================================================================= SCOUTING
elif ss.page == "Scouting":
    st.markdown('<div class="page-title">Find your next player</div>'
                f'<div class="page-sub">Filter {len(df):,} players by profile and performance.</div>', unsafe_allow_html=True)

    sq = st.text_input("name", placeholder="🔍  Search by name (optional)", label_visibility="collapsed", key="scsearch")
    f = st.columns(5)
    pos = f[0].selectbox("Position", ["Any", "Forward", "Midfielder", "Defender"])
    msc = f[1].slider("Min score", 0.0, float(df.Score.max()), 0.30, 0.05)
    mpa = f[2].slider("Min pass acc %", 0, 95, 70)
    mtk = f[3].slider("Min tackles / 90", 0.0, 5.0, 0.0, 0.1)
    prf = f[4].selectbox("Profile", ["Any"] + sorted(df.Profile.unique()))

    q = df[(df.Score >= msc) & (df.PassAcc >= mpa) & (df.Tackles_p90 >= mtk)]
    if pos != "Any":
        q = q[q.Position == pos]
    if prf != "Any":
        q = q[q.Profile == prf]
    if sq:
        qn = "".join(ch for ch in unicodedata.normalize("NFKD", sq) if not unicodedata.combining(ch)).lower()
        q = q[q.norm.str.contains(qn, regex=False)]

    head = st.columns([2, 1.3])
    head[0].markdown(f'<div style="font-family:Archivo;font-weight:900;font-size:1.5rem;">{len(q)} players found</div>',
                     unsafe_allow_html=True)
    sort = head[1].selectbox("Sort by", ["Performance score", "Pass accuracy", "Tackles / 90", "Shots / 90"],
                             label_visibility="collapsed")
    skey = {"Performance score": "Score", "Pass accuracy": "PassAcc", "Tackles / 90": "Tackles_p90", "Shots / 90": "Shots_p90"}[sort]
    q = q.sort_values(skey, ascending=False)

    if q.empty:
        st.info("No players match these filters. Loosen them a little.")
    else:
        shown = q.head(24)
        cols = st.columns(3)
        for n, (idx, r) in enumerate(shown.iterrows()):
            c = PROFILE_COLOR[r.Profile]
            with cols[n % 3]:
                st.markdown(f"""<div class="simcard" style="margin-bottom:.4rem;">
                  <div style="display:flex;justify-content:space-between;align-items:baseline;">
                    <span class="nm">{r.Player}</span>
                    <span style="font-family:Archivo;font-weight:900;font-size:1.2rem;">{r.Score:.2f}</span></div>
                  <div class="mt">{r.Team} · {r.Position}
                    <span class="badge" style="background:{c}1f;color:{c};font-size:.68rem;padding:.12rem .5rem;">{r.Profile}</span></div>
                  <div style="display:flex;gap:1rem;font-size:.8rem;color:{MUTED};margin-top:.2rem;">
                    <span>Shots {r.Shots_p90:.2f}</span><span>Pass {r.PassAcc:.0f}%</span><span>Tkl {r.Tackles_p90:.2f}</span></div>
                </div>""", unsafe_allow_html=True)
                if st.button("View player →", key=f"go_{idx}", use_container_width=True):
                    go_player(idx)
        if len(q) > 24:
            st.caption(f"Showing the top 24 of {len(q)} by {sort.lower()}. Narrow the filters to see more specific matches.")

    with st.expander("Detailed table view"):
        tbl = q[["Player", "Team", "Position", "Profile", "Score", "Pred_Score", "Pred_Category",
                 "Shots_p90", "PassAcc", "Tackles_p90"]].rename(columns={
            "Pred_Score": "Pred. score", "Pred_Category": "Pred. category", "Shots_p90": "Shots/90",
            "PassAcc": "Pass %", "Tackles_p90": "Tackles/90"}).round(2)
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=360)
        st.markdown('<div class="note">Predicted score / category are application outputs for players represented in the '
                    'dataset, not a held-out evaluation.</div>', unsafe_allow_html=True)
