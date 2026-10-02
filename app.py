"""Football Player Analytics & Scouting — Streamlit front end (expanded representation).

Run:  streamlit run app.py

Nothing is trained here. football_ml.ipynb writes models/artifacts.joblib + models/players_scored.csv;
this app loads them and looks things up. Player search uses a Trie index (see search_index.py).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai_scout
import scouting
from search_index import build_index

st.set_page_config(page_title="Football Analytics", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------- palette / theme
INK, MUTED, LINE = "#0F1115", "#6B7280", "#E6E7E0"
BG, CARD = "#F6F7F1", "#FFFFFF"
GREEN, COBALT, ORANGE = "#00A651", "#2457F5", "#FF7A00"
PALETTE = [ORANGE, COBALT, GREEN, "#8B5CF6", "#E11D48", "#0EA5E9", "#F59E0B", "#14B8A6"]
LAB = {"xG_p90": "xG / 90", "Shots_p90": "Shots / 90", "KeyPasses_p90": "Key passes / 90",
       "xAG_p90": "xAG / 90", "ProgPasses_p90": "Prog. passes / 90", "ProgCarries_p90": "Prog. carries / 90",
       "Passes_p90": "Passes / 90", "PassAcc": "Pass accuracy", "Tackles_p90": "Tackles / 90",
       "Interceptions_p90": "Interceptions / 90", "AerialWinPct": "Aerial win %"}
HEADLINE = ["xG_p90", "Shots_p90", "KeyPasses_p90", "xAG_p90", "ProgCarries_p90", "ProgPasses_p90",
            "Tackles_p90", "Interceptions_p90", "AerialWinPct"]


@st.cache_resource
def _loaded():
    b = scouting.load()
    return b, build_index(b["df"])


B, INDEX = _loaded()
art, df, SIM = B["art"], B["df"], B["sim"]
PROF, CATS, GROUPS = B["prof"], B["cats"], art["groups"]
PCT, PCTP = B["pct_overall"], B["pct_pos"]
FEATURES = art["features"]                       # baseline 5 (what-if + baseline models)
CAT_NAMES = [c.replace("idx_", "") for c in CATS]
PROFILES = sorted(df.Profile.unique())
PROFILE_COLOR = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(PROFILES)}
SCORE_PCTL = df.Score.rank(pct=True) * 100


@st.cache_data(show_spinner="Asking the AI Scout…")
def ai_call(kind, *args):
    fn = {"report": ai_scout.player_report, "compare": ai_scout.compare_report,
          "search": ai_scout.natural_search, "shortlist": ai_scout.shortlist_report}[kind]
    return fn(*args)


def fnum(v, f):
    return f"{v:.1f}" if ("Acc" in f or "Pct" in f) else f"{v:.2f}"


# ---------------------------------------------------------------- CSS
def css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
header[data-testid="stHeader"], #MainMenu, footer {{ display:none; }}
.stApp {{ background:{BG}; }}
html, body, .stApp {{ font-family:'Inter',sans-serif; color:{INK}; }}
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown, .stApp button,
.stApp input, .stApp select, .stApp textarea {{ font-family:'Inter',sans-serif; }}
[data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"] {{
   font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important; }}
.block-container {{ max-width:1180px; padding-top:1.2rem; padding-bottom:3rem; }}
h1,h2,h3 {{ font-family:'Archivo',sans-serif; letter-spacing:-.01em; }}
.topbar {{ display:flex; align-items:center; gap:.6rem; padding:.2rem 0 .1rem; }}
.brand {{ font-family:'Archivo'; font-weight:900; font-size:1.35rem; letter-spacing:-.02em; }}
.brand .dot {{ color:{GREEN}; }}
.brand .sub {{ font-family:'Inter'; font-weight:500; font-size:.72rem; color:{MUTED}; letter-spacing:.14em; text-transform:uppercase; margin-left:.1rem; }}
hr.rule {{ border:none; border-top:1px solid {LINE}; margin:.5rem 0 1.1rem; }}
[class*="st-key-nav_"] button {{ border-radius:999px !important; border:1px solid {LINE} !important;
   font-weight:700 !important; font-family:'Archivo' !important; background:{CARD} !important; color:{INK} !important;
   padding:.35rem 0 !important; box-shadow:none !important; }}
[class*="st-key-nav_"] button[kind="primary"] {{ background:{INK} !important; border-color:{INK} !important; }}
[class*="st-key-nav_"] button[kind="primary"], [class*="st-key-nav_"] button[kind="primary"] * {{ color:#fff !important; }}
[class*="st-key-go_"] button, [class*="st-key-sim_"] button, [class*="st-key-sr_"] button,
[class*="st-key-sl_"] button, [class*="st-key-ai_"] button, [class*="st-key-cmp_"] button {{
   border-radius:8px !important; border:1px solid {LINE} !important; background:{CARD} !important;
   color:{INK} !important; font-weight:600 !important; box-shadow:none !important; }}
[class*="st-key-go_"] button:hover, [class*="st-key-sim_"] button:hover, [class*="st-key-sr_"] button:hover,
[class*="st-key-sl_"] button:hover, [class*="st-key-ai_"] button:hover, [class*="st-key-cmp_"] button:hover {{
   border-color:{GREEN} !important; color:{GREEN} !important; }}
.eyebrow {{ font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:{MUTED}; }}
.hero-name {{ font-family:'Archivo'; font-weight:900; font-size:3rem; line-height:1; margin:.1rem 0 .3rem; }}
.meta {{ color:{MUTED}; font-size:.95rem; font-weight:500; }}
.section {{ font-family:'Archivo'; font-weight:800; font-size:1.25rem; margin:.2rem 0 .7rem; text-transform:uppercase; letter-spacing:.02em; }}
.note {{ color:{MUTED}; font-size:.8rem; font-style:italic; margin-top:.3rem; }}
.page-title {{ font-family:'Archivo'; font-weight:900; font-size:2.3rem; line-height:1.02; }}
.page-sub {{ color:{MUTED}; font-size:1rem; margin-bottom:1rem; }}
.badge {{ display:inline-block; padding:.28rem .7rem; border-radius:999px; font-weight:700; font-size:.8rem; margin-right:.4rem; }}
.b-high {{ background:rgba(0,166,81,.12); color:{GREEN}; }}
.b-low {{ background:rgba(107,114,128,.14); color:{MUTED}; }}
.card {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:1.1rem 1.2rem; height:100%; }}
.tile {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:.8rem .9rem; }}
.tile .k {{ font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; }}
.tile .v {{ font-family:'Archivo'; font-weight:800; font-size:1.5rem; margin-top:.1rem; }}
.tile .s {{ font-size:.75rem; color:{MUTED}; }}
.chip {{ display:inline-block; padding:.25rem .6rem; border-radius:8px; font-size:.78rem; font-weight:700; margin:.15rem .25rem .15rem 0; }}
.attr {{ margin-bottom:.7rem; }}
.attr .top {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.25rem; }}
.attr .nm {{ font-size:.76rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }}
.attr .vl {{ font-family:'Archivo'; font-weight:800; font-size:1rem; }}
.attr .track {{ height:9px; background:#EFEFE9; border-radius:999px; overflow:hidden; }}
.attr .fill {{ height:100%; border-radius:999px; }}
.attr .pc {{ font-size:.7rem; color:{MUTED}; margin-top:.2rem; }}
.simcard {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:.75rem .9rem; margin-bottom:.1rem; }}
.simcard .nm {{ font-weight:800; font-family:'Archivo'; font-size:1.02rem; }}
.simcard .mt {{ color:{MUTED}; font-size:.8rem; }}
.simcard .track {{ height:7px; background:#EFEFE9; border-radius:999px; overflow:hidden; margin-top:.3rem; }}
.simcard .fill {{ height:100%; border-radius:999px; }}
.arche {{ background:{CARD}; border:1px solid {LINE}; border-top:5px solid var(--c); border-radius:16px; padding:1.1rem 1.2rem; height:100%; }}
.arche .nm {{ font-family:'Archivo'; font-weight:900; font-size:1.25rem; }}
.arche .ct {{ color:{MUTED}; font-size:.82rem; font-weight:600; margin-bottom:.5rem; }}
.arche .row {{ display:flex; justify-content:space-between; font-size:.83rem; padding:.18rem 0; border-top:1px solid {LINE}; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- visual builders
def cat_color(i):
    return PALETTE[i % len(PALETTE)]


def gauge_html(score, pctl, is_high):
    frac = max(0.02, min(1.0, pctl / 100)); C = 2 * np.pi * 52; dash = C * frac
    col = GREEN if is_high else MUTED
    return f"""<div style="text-align:center;background:{CARD};border:1px solid {LINE};border-radius:16px;padding:1rem .6rem;">
  <div class="eyebrow" style="margin-bottom:.4rem;">Performance score</div>
  <svg width="150" height="150" viewBox="0 0 130 130">
    <circle cx="65" cy="65" r="52" fill="none" stroke="#EFEFE9" stroke-width="12"/>
    <circle cx="65" cy="65" r="52" fill="none" stroke="{col}" stroke-width="12" stroke-linecap="round"
      stroke-dasharray="{dash:.1f} {C:.1f}" transform="rotate(-90 65 65)"/>
    <text x="65" y="60" text-anchor="middle" font-family="Archivo" font-weight="900" font-size="30" fill="{INK}">{score:.2f}</text>
    <text x="65" y="80" text-anchor="middle" font-family="Inter" font-size="11" fill="{MUTED}">top {100 - pctl:.0f}%</text>
  </svg>
  <div style="margin-top:.4rem;"><span class="badge {'b-high' if is_high else 'b-low'}">{'HIGH PERFORMER' if is_high else 'LOW PERFORMER'}</span></div>
</div>"""


def category_radar(row, color=GREEN, name=None):
    vals = [float(row["idx_" + c]) for c in CAT_NAMES]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=CAT_NAMES + [CAT_NAMES[0]], fill="toself",
                                    name=name or "", fillcolor="rgba(0,166,81,.18)", line=dict(color=color, width=2.5),
                                    hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                      radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor=LINE),
                      angularaxis=dict(gridcolor=LINE, tickfont=dict(size=11, color=INK, family="Inter"))),
                      showlegend=False, margin=dict(l=40, r=40, t=30, b=30), height=320,
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
    return fig


def attr_bars(row, feats):
    out = []
    for f in feats:
        p = float(PCT.iloc[row.name][f]) if f in PCT.columns else 0.0
        col = GREEN if p >= 66 else (COBALT if p >= 33 else ORANGE)
        out.append(f"""<div class="attr"><div class="top"><span class="nm">{LAB.get(f,f)}</span>
          <span class="vl">{fnum(row[f], f)}{'%' if ('Acc' in f or 'Pct' in f) else ''}</span></div>
          <div class="track"><div class="fill" style="width:{p:.0f}%;background:{col};"></div></div>
          <div class="pc">{p:.0f}th percentile overall</div></div>""")
    return "".join(out)


# ---------------------------------------------------------------- state + nav
css()
ss = st.session_state
ss.setdefault("page", "Players")
ss.setdefault("pid", int(df.index[df.Player == "Bukayo Saka"][0]) if (df.Player == "Bukayo Saka").any() else 0)
ss.setdefault("shortlist", [])
ss.setdefault("cmp_a", df.iloc[ss.pid].Player)
ss.setdefault("cmp_b", df[df.Player != df.iloc[ss.pid].Player].sort_values("Score", ascending=False).iloc[0].Player)

PAGES = ["Players", "Analytics", "Profiles", "Scouting", "Compare", "AI Scout"]
st.markdown('<div class="topbar"><span class="brand">FOOTBALL<span class="dot">.</span>SCOUT</span>'
            '<span class="sub">scouting intelligence</span></div>', unsafe_allow_html=True)
nav = st.columns(len(PAGES))
for col, label in zip(nav, PAGES):
    sl = f" ({len(ss.shortlist)})" if label == "AI Scout" and ss.shortlist else ""
    with col:
        if st.button(label + sl, key=f"nav_{label}", use_container_width=True,
                     type="primary" if ss.page == label else "secondary"):
            ss.page = label; st.rerun()
st.markdown('<hr class="rule">', unsafe_allow_html=True)


def go_player(pid):
    ss.pid = int(pid); ss.page = "Players"; st.rerun()


def toggle_shortlist(name):
    (ss.shortlist.remove(name) if name in ss.shortlist else ss.shortlist.append(name))


def pid_of(name):
    m = df.index[df.Player == name]
    return int(m[0]) if len(m) else None


def similar_rows(pid, n=5, same_pos=False):
    d = np.linalg.norm(SIM - SIM[pid], axis=1)
    order = [j for j in np.argsort(d) if j != pid]
    if same_pos:
        order = [j for j in order if df.Position.iloc[j] == df.Position.iloc[pid]]
    return [(j, 100 * (1 - d[j] / d.max())) for j in order[:n]]


# ================================================================= PLAYERS
if ss.page == "Players":
    q = st.text_input("Search players", placeholder="🔍  Search a player — type a name (Trie autocomplete)",
                      label_visibility="collapsed", key="psearch")
    if q:
        hits = INDEX.search(q, limit=5)       # DSA: O(len(query)) prefix lookup, top-5 by score
        if not hits:
            st.info("No players found. Try a different name.")
        else:
            st.caption("Top matches:")
            cols = st.columns(len(hits))
            for col, idx in zip(cols, hits):
                r = df.iloc[idx]
                if col.button(f"{r.Player}\n{r.Team} · {r.Position}", key=f"sr_{idx}", use_container_width=True):
                    go_player(idx)
        st.markdown('<hr class="rule">', unsafe_allow_html=True)

    p = df.iloc[ss.pid]
    is_high = p.Pred_Category == "High"
    pcol = PROFILE_COLOR.get(p.Profile, GREEN)
    st.markdown(f'<div class="eyebrow">Scouting report</div><div class="hero-name">{p.Player}</div>'
                f'<div class="meta">{p.Team} · {p.League} · {p.Position} · {int(p.Minutes):,} minutes · age {str(p.Age).split("-")[0]}</div>',
                unsafe_allow_html=True)

    act = st.columns([1, 1, 1, 2])
    in_list = p.Player in ss.shortlist
    if act[0].button("✓ In shortlist" if in_list else "＋ Add to shortlist", key="sl_card", use_container_width=True):
        toggle_shortlist(p.Player); st.rerun()
    if act[1].button("Compare →", key="cmp_card", use_container_width=True):
        ss.cmp_a = p.Player; ss.page = "Compare"; st.rerun()
    if act[2].button("Generate AI report", key="ai_card", use_container_width=True, disabled=not ai_scout.available()):
        ss.ai_report = (p.Player, *ai_call("report", p.Player))
    if not ai_scout.available():
        act[3].caption("AI report needs a Gemini API key (see AI Scout tab).")
    if ss.get("ai_report") and ss.ai_report[0] == p.Player:
        _, ok, txt = ss.ai_report
        with st.container(border=True):
            st.markdown(f"**AI scout report — {p.Player}**"); st.markdown(txt if ok else f":orange[{txt}]")
    st.write("")

    a, b, c = st.columns([1, 1.15, 1])
    with a:
        st.markdown(gauge_html(p.Score, float(SCORE_PCTL.iloc[ss.pid]), is_high), unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="eyebrow" style="text-align:center;">Role fingerprint</div>', unsafe_allow_html=True)
        st.plotly_chart(category_radar(p, pcol), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c:
        chips = "".join(f'<span class="chip" style="background:{cat_color(i)}1f;color:{cat_color(i)};">{nm} {int(p["idx_"+nm])}</span>'
                        for i, nm in enumerate(CAT_NAMES))
        st.markdown(f"""<div class="card"><div class="eyebrow">Role profile</div>
          <div style="font-family:Archivo;font-weight:900;font-size:1.5rem;color:{pcol};margin:.1rem 0 .4rem;">{p.Profile}</div>
          <div class="eyebrow">Category indices (0-100)</div><div style="margin-top:.3rem;">{chips}</div>
          <div class="row" style="display:flex;gap:1.4rem;margin-top:.7rem;border-top:1px solid {LINE};padding-top:.6rem;">
            <span><b style="font-family:Archivo;font-size:1.2rem;">{p.Pred_Score:.2f}</b><br><span class="note" style="font-style:normal;">model prediction</span></span>
            <span><b style="font-family:Archivo;font-size:1.2rem;">{'High' if p.Label==1 else 'Low'}</b><br><span class="note" style="font-style:normal;">actual category</span></span>
          </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="note">Category indices are average percentile ranks within each football concept. '
                'Performance score = goals + assists per 90 (never a model input). Profiles are data-derived from the '
                '~28-feature style space; position is used only to validate them.</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section">Key attributes</div>', unsafe_allow_html=True)
        half = (len(HEADLINE) + 1) // 2
        hc = st.columns(2)
        hc[0].markdown(attr_bars(p, HEADLINE[:half]), unsafe_allow_html=True)
        hc[1].markdown(attr_bars(p, HEADLINE[half:]), unsafe_allow_html=True)
        with st.expander("All metrics by football concept"):
            for g, feats in GROUPS.items():
                st.markdown(f'<div class="eyebrow" style="margin-top:.4rem;">{g} · index {int(p["idx_"+g])}/100</div>', unsafe_allow_html=True)
                st.markdown(attr_bars(p, feats), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section">Similar players</div>', unsafe_allow_html=True)
        same_pos = st.toggle("Same position only", value=False, key="simpos")
        for idx, sim in similar_rows(ss.pid, 5, same_pos):
            r = df.iloc[idx]; cc = PROFILE_COLOR.get(r.Profile, GREEN)
            st.markdown(f"""<div class="simcard"><div style="display:flex;justify-content:space-between;align-items:baseline;">
              <span class="nm">{r.Player}</span><span style="font-family:Archivo;font-weight:800;color:{cc};">{sim:.1f}%</span></div>
              <div class="mt">{r.Team} · {r.Position} · {r.Profile}</div>
              <div class="track"><div class="fill" style="width:{sim:.0f}%;background:{cc};"></div></div></div>""",
                        unsafe_allow_html=True)
            if st.button("View player →", key=f"sim_{idx}", use_container_width=True):
                go_player(idx)

    st.write("")
    with st.expander("WHAT IF?  —  adjust the baseline stats and watch the models respond"):
        st.caption("Scenario values — not the player's real statistics. The baseline models re-predict live.")
        wi, sc = {}, st.columns(5)
        for col, f in zip(sc, FEATURES):
            lo, hi = float(df[f].min()), float(df[f].quantile(0.995))
            wi[f] = col.slider(LAB.get(f, f), lo, round(hi, 1), float(min(max(p[f], lo), hi)), key=f"wi_{f}")
        row = pd.DataFrame([wi])[FEATURES]
        rf = art["reg_models"]["Random Forest"].predict(row)[0]
        lr = art["reg_models"]["Linear Regression"].predict(row)[0]
        prob = art["clf_models"]["Logistic Regression"].predict_proba(row)[0, 1]
        m = st.columns(3)
        for col, (k, v) in zip(m, [("Random Forest score", f"{rf:.2f}"), ("Linear Regression score", f"{lr:.2f}"),
                                   ("P(High performer)", f"{prob:.0%}")]):
            col.markdown(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)

# ================================================================= ANALYTICS
elif ss.page == "Analytics":
    st.markdown('<div class="page-title">Model performance</div>'
                '<div class="page-sub">Honest evaluation: every model shown with 5-fold cross-validation and a held-out '
                'test split. Supporting evidence behind the predictions — not the first thing scouts need.</div>',
                unsafe_allow_html=True)
    cv, reg, clf = art["cv_results"], art["reg_results"], art["clf_results"]

    st.markdown('<div class="section">Does a richer feature set classify better?</div>', unsafe_allow_html=True)
    st.caption("Baseline = 5 original stats · Expanded-A = ~25 style features (no xG) · Expanded-B = + xG/xAG. "
               "Goals/assists are never inputs. Judged on cross-validation, not a single split.")
    exp = art["expanded_clf"].reset_index()
    show = exp[["Representation", "Model", "CV acc", "CV F1", "CV AUC", "Test acc", "Test AUC"]].copy()
    for cc in ["CV acc", "CV F1", "CV AUC", "Test acc", "Test AUC"]:
        show[cc] = show[cc].map(lambda x: f"{x:.3f}")
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.markdown('<div class="note">The expanded set gives a real, cross-validated lift (e.g. Logistic CV accuracy '
                f'{exp[(exp.Representation=="Baseline (5)")&(exp.Model=="Logistic Regression")]["CV acc"].iloc[0]:.3f} → '
                f'{exp[(exp.Representation=="Expanded-A (no xG)")&(exp.Model=="Logistic Regression")]["CV acc"].iloc[0]:.3f}). '
                'Expanded-B adds xG, which is outcome-correlated, so its edge is expected and flagged.</div>',
                unsafe_allow_html=True)

    st.write(""); st.markdown('<div class="section">Importance by football concept</div>', unsafe_allow_html=True)
    gi = art["grouped_importance"].sort_values()
    fig = go.Figure(go.Bar(x=gi.values, y=gi.index, orientation="h", marker_color=GREEN,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"), xaxis=dict(gridcolor=LINE))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Baseline experiment — regression & classification (5 features)"):
        r = reg.copy()
        r.insert(0, "CV R² (mean ± std)", [f"{cv.loc[('Regression (R2)', m), 'CV mean']:.3f} ± {cv.loc[('Regression (R2)', m), 'CV std']:.3f}" for m in r.index])
        st.dataframe(r.rename(columns={"R2": "Test R²"}), use_container_width=True)
        cc2 = clf.copy()
        cc2.insert(0, "CV accuracy", [f"{cv.loc[('Classification (Accuracy)', m), 'CV mean']:.3f} ± {cv.loc[('Classification (Accuracy)', m), 'CV std']:.3f}" for m in cc2.index])
        st.dataframe(cc2.rename(columns={"Accuracy": "Test accuracy"}), use_container_width=True)
        st.caption("Regressors are effectively tied on CV; Logistic edges the tree. Position drives much of the signal "
                   "(forwards shoot and score more), a limit of the feature set.")

# ================================================================= PROFILES
elif ss.page == "Profiles":
    st.markdown('<div class="page-title">Player archetypes</div>'
                f'<div class="page-sub">K-Means on ~{len(PROF)} style features (position not used as input) found '
                f'{art["rich_k"]} data-driven roles, named from their measured category indices.</div>', unsafe_allow_html=True)
    prof = art["rich_cluster_profile"].rename(index=art["rich_cluster_names"])
    posmix = pd.crosstab(df.Profile, df.Position, normalize="index") * 100
    cols = st.columns(len(prof))
    for col, name in zip(cols, prof.index):
        c = PROFILE_COLOR.get(name, GREEN)
        cat_means = prof.loc[name, CATS].astype(float)
        top3 = cat_means.sort_values(ascending=False).head(3)
        rows = "".join(f'<div class="arche row"><span>{k.replace("idx_","")}</span><b>{v:.0f}</b></div>' for k, v in top3.items())
        tp = posmix.loc[name].idxmax()
        col.markdown(f"""<div class="arche" style="--c:{c};"><div class="nm" style="color:{c};">{name}</div>
          <div class="ct">{int(prof.loc[name,'Players'])} players · mostly {tp}s ({posmix.loc[name,tp]:.0f}%)</div>
          <div class="eyebrow">top category indices</div>{rows}</div>""", unsafe_allow_html=True)

    st.write(""); st.markdown('<div class="section">Player map</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub" style="margin-top:-.3rem;">Players placed by style similarity (PCA of {len(PROF)} '
                f'features; first two components explain {100*np.sum(art["pca_var"]):.1f}%). Hover any point.</div>', unsafe_allow_html=True)
    mode = st.radio("Colour by", ["By profile", "By performance"], horizontal=True, label_visibility="collapsed")
    star = ss.pid if st.checkbox(f"Highlight current player ({df.iloc[ss.pid].Player})", value=True) else None
    fig = go.Figure()
    if mode == "By profile":
        for prof_name in PROFILES:
            s = df[df.Profile == prof_name]
            fig.add_scatter(x=s.PC1, y=s.PC2, mode="markers", name=prof_name,
                            marker=dict(size=6, color=PROFILE_COLOR[prof_name], opacity=.65),
                            customdata=np.stack([s.Player, s.Team, s.Position, s.Score], -1),
                            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>Score %{customdata[3]:.2f}<extra></extra>")
    else:
        fig.add_scatter(x=df.PC1, y=df.PC2, mode="markers",
                        marker=dict(size=6, color=df.Score, colorscale="Viridis", cmax=df.Score.quantile(.98),
                                    opacity=.7, colorbar=dict(title="Score", thickness=12, len=.7)),
                        customdata=np.stack([df.Player, df.Team, df.Position, df.Score], -1),
                        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>Score %{customdata[3]:.2f}<extra></extra>")
    if star is not None:
        h = df.iloc[star]
        fig.add_scatter(x=[h.PC1], y=[h.PC2], mode="markers+text", marker=dict(size=16, color=INK, symbol="star", line=dict(width=2, color="#fff")),
                        text=[h.Player], textposition="top center", textfont=dict(family="Archivo", size=12), hoverinfo="skip", showlegend=False)
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter"), legend=dict(orientation="h", y=1.08, x=0),
                      xaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, showticklabels=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("How many roles? (K diagnostics)"):
        st.dataframe(art["rich_k_table"], hide_index=True, use_container_width=True)
        st.caption(f"K = {art['rich_k']} chosen from silhouette + inertia + minimum cluster size. Silhouette is modest, "
                   "so roles are tendencies with real overlap — we don't invent finer roles than the data supports.")

# ================================================================= SCOUTING
elif ss.page == "Scouting":
    st.markdown('<div class="page-title">Find your next player</div>'
                f'<div class="page-sub">Filter {len(df):,} players by role and performance.</div>', unsafe_allow_html=True)
    sq = st.text_input("name", placeholder="🔍  Search by name (optional)", label_visibility="collapsed", key="scsearch")
    f = st.columns(5)
    pos = f[0].selectbox("Position", ["Any", "Forward", "Midfielder", "Defender"])
    prf = f[1].selectbox("Profile", ["Any"] + PROFILES)
    msc = f[2].slider("Min score", 0.0, float(df.Score.max()), 0.0, 0.05)
    mxg = f[3].slider("Min xG / 90", 0.0, float(df.xG_p90.quantile(.99)), 0.0, 0.05)
    mpc = f[4].slider("Min prog. carries / 90", 0.0, float(df.ProgCarries_p90.quantile(.99)), 0.0, 0.5)

    q = df[(df.Score >= msc) & (df.xG_p90 >= mxg) & (df.ProgCarries_p90 >= mpc)]
    if pos != "Any":
        q = q[q.Position == pos]
    if prf != "Any":
        q = q[q.Profile == prf]
    if sq:
        ids = INDEX.search(sq, limit=50)
        q = q[q.index.isin(ids)]

    head = st.columns([2, 1.3])
    head[0].markdown(f'<div style="font-family:Archivo;font-weight:900;font-size:1.5rem;">{len(q)} players found</div>', unsafe_allow_html=True)
    sort = head[1].selectbox("Sort by", ["Performance score", "xG / 90", "Prog. carries / 90", "Pass accuracy"], label_visibility="collapsed")
    skey = {"Performance score": "Score", "xG / 90": "xG_p90", "Prog. carries / 90": "ProgCarries_p90", "Pass accuracy": "PassAcc"}[sort]
    q = q.sort_values(skey, ascending=False)

    if q.empty:
        st.info("No players match these filters. Loosen them a little.")
    else:
        cols = st.columns(3)
        for n, (idx, r) in enumerate(q.head(24).iterrows()):
            c = PROFILE_COLOR.get(r.Profile, GREEN)
            with cols[n % 3]:
                st.markdown(f"""<div class="simcard" style="margin-bottom:.4rem;"><div style="display:flex;justify-content:space-between;align-items:baseline;">
                  <span class="nm">{r.Player}</span><span style="font-family:Archivo;font-weight:900;font-size:1.2rem;">{r.Score:.2f}</span></div>
                  <div class="mt">{r.Team} · {r.Position} <span class="chip" style="background:{c}1f;color:{c};font-size:.66rem;padding:.1rem .45rem;">{r.Profile}</span></div>
                  <div style="display:flex;gap:.9rem;font-size:.78rem;color:{MUTED};margin-top:.2rem;">
                    <span>xG {r.xG_p90:.2f}</span><span>KP {r.KeyPasses_p90:.2f}</span><span>PrgC {r.ProgCarries_p90:.1f}</span></div></div>""",
                            unsafe_allow_html=True)
                bc = st.columns(2)
                if bc[0].button("View →", key=f"go_{idx}", use_container_width=True):
                    go_player(idx)
                if bc[1].button("✓" if r.Player in ss.shortlist else "＋", key=f"sl_{idx}", use_container_width=True):
                    toggle_shortlist(r.Player); st.rerun()
        if len(q) > 24:
            st.caption(f"Showing the top 24 of {len(q)} by {sort.lower()}.")

    st.write(""); st.markdown('<div class="section">Players similar to…</div>', unsafe_allow_html=True)
    sc2 = st.columns([2, 1])
    simname = sc2[0].selectbox("player", df.sort_values("Score", ascending=False).Player.tolist(), label_visibility="collapsed", key="scout_sim")
    spos = sc2[1].toggle("Same position", value=False, key="scout_simpos")
    for idx, sim in similar_rows(pid_of(simname), 5, spos):
        r = df.iloc[idx]
        rr = st.columns([3, 1])
        rr[0].markdown(f'<div style="padding-top:.4rem;"><b style="font-family:Archivo;">{r.Player}</b> '
                       f'<span class="meta">· {r.Team} · {r.Position} · {r.Profile} · {sim:.1f}%</span></div>', unsafe_allow_html=True)
        if rr[1].button("View", key=f"go_s_{idx}", use_container_width=True):
            go_player(idx)

# ================================================================= COMPARE
elif ss.page == "Compare":
    st.markdown('<div class="page-title">Head to head</div>'
                '<div class="page-sub">Compare any two players — category indices, stats, model prediction and style similarity.</div>',
                unsafe_allow_html=True)
    names = df.sort_values("Score", ascending=False).Player.tolist()
    pick = st.columns(2)
    a_name = pick[0].selectbox("Player A", names, index=names.index(ss.cmp_a) if ss.cmp_a in names else 0, key="cmp_sel_a")
    b_name = pick[1].selectbox("Player B", names, index=names.index(ss.cmp_b) if ss.cmp_b in names else 1, key="cmp_sel_b")
    ss.cmp_a, ss.cmp_b = a_name, b_name
    ia, ib = pid_of(a_name), pid_of(b_name)
    ra, rb = df.iloc[ia], df.iloc[ib]
    dist = float(np.linalg.norm(SIM[ia] - SIM[ib]))
    sim = 100 * (1 - dist / float(np.linalg.norm(SIM - SIM[ia], axis=1).max()))

    head = st.columns([1, 1, 1])
    for col, r, cc in [(head[0], ra, COBALT), (head[2], rb, ORANGE)]:
        col.markdown(f"""<div class="card" style="text-align:center;"><div style="font-family:Archivo;font-weight:900;font-size:1.4rem;color:{cc};">{r.Player}</div>
          <div class="meta">{r.Team} · {r.Position}</div>
          <div style="font-family:Archivo;font-weight:900;font-size:2.4rem;margin-top:.3rem;">{r.Score:.2f}</div><div class="eyebrow">performance score</div>
          <div style="margin-top:.4rem;"><span class="badge" style="background:{cc}1f;color:{cc};">{r.Profile}</span></div></div>""", unsafe_allow_html=True)
    head[1].markdown(f"""<div class="card" style="text-align:center;height:100%;display:flex;flex-direction:column;justify-content:center;">
      <div class="eyebrow">style similarity</div><div style="font-family:Archivo;font-weight:900;font-size:2.6rem;color:{GREEN};">{sim:.0f}%</div>
      <div class="note" style="font-style:normal;">in the ~{len(PROF)}-feature space</div></div>""", unsafe_allow_html=True)

    st.write("")
    g = st.columns([1.1, 1])
    with g[0]:
        st.markdown('<div class="section">Role fingerprint</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for r, cc in [(ra, COBALT), (rb, ORANGE)]:
            vals = [float(r["idx_" + c]) for c in CAT_NAMES]
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=CAT_NAMES + [CAT_NAMES[0]], fill="toself", name=r.Player,
                                          line=dict(color=cc, width=2.5), opacity=.55, hovertemplate="%{theta}: %{r:.0f}<extra>" + r.Player + "</extra>"))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor=LINE),
                          angularaxis=dict(gridcolor=LINE, tickfont=dict(size=11, family="Inter"))),
                          legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=40, r=40, t=40, b=30), height=360,
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with g[1]:
        st.markdown('<div class="section">Attribute by attribute</div>', unsafe_allow_html=True)
        rows = [(c, f"{ra['idx_'+c]:.0f}", f"{rb['idx_'+c]:.0f}") for c in CAT_NAMES]
        rows += [(LAB[f], fnum(ra[f], f), fnum(rb[f], f)) for f in ["xG_p90", "KeyPasses_p90", "Tackles_p90"]]
        body = "".join(f'<div style="display:flex;align-items:center;padding:.4rem .2rem;border-top:1px solid {LINE};">'
                       f'<span style="color:{COBALT};font-weight:800;font-family:Archivo;width:60px;">{a}</span>'
                       f'<span style="color:{MUTED};font-size:.78rem;flex:1;text-align:center;">{lab}</span>'
                       f'<span style="color:{ORANGE};font-weight:800;font-family:Archivo;width:60px;text-align:right;">{b}</span></div>'
                       for lab, a, b in rows)
        st.markdown(f'<div class="card" style="padding-top:.4rem;">{body}</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("AI comparison", key="ai_cmp", disabled=not ai_scout.available()):
        ss.ai_compare = (a_name, b_name, *ai_call("compare", a_name, b_name))
    if not ai_scout.available():
        st.caption("AI comparison needs a Gemini API key (see AI Scout tab).")
    if ss.get("ai_compare") and ss.ai_compare[:2] == (a_name, b_name):
        _, _, ok, txt = ss.ai_compare
        with st.container(border=True):
            st.markdown(f"**AI comparison — {a_name} vs {b_name}**"); st.markdown(txt if ok else f":orange[{txt}]")

# ================================================================= AI SCOUT
elif ss.page == "AI Scout":
    st.markdown('<div class="page-title">AI Scout</div>'
                '<div class="page-sub">Ask in plain English. The assistant queries the real dataset through the app\'s '
                'scouting functions and explains what it finds — it never invents players or stats.</div>', unsafe_allow_html=True)
    if ai_scout.available():
        st.markdown(f'<span class="chip" style="background:rgba(0,166,81,.12);color:{GREEN};">● AI Scout online</span>', unsafe_allow_html=True)
    else:
        st.warning("**AI Scout unavailable — configure a Gemini API key.** Set `GEMINI_API_KEY` as an environment "
                   "variable or in `.streamlit/secrets.toml` (see `.env.example`). Everything else works without it.")
    st.write(""); st.markdown('<div class="section">Natural-language search</div>', unsafe_allow_html=True)
    query = st.text_input("ask", value="", placeholder="Find a winger like Saka with strong progressive carrying and ≥75% pass accuracy.",
                          label_visibility="collapsed", key="nl_query")
    st.caption("Try: “Who is statistically similar to Rodri?” · “Finishers with high xG” · “Compare Saka and Doku” · "
               "“Ball-playing defenders with strong progression”")
    if st.button("Ask AI Scout", key="ai_search", disabled=not ai_scout.available()) and query.strip():
        ss.ai_nl = (query, *ai_call("search", query))
    if ss.get("ai_nl"):
        q0, ok, txt = ss.ai_nl
        with st.container(border=True):
            st.markdown(f"**Query:** {q0}"); st.markdown(txt if ok else f":orange[{txt}]")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown(f'<div class="section">My shortlist ({len(ss.shortlist)})</div>', unsafe_allow_html=True)
    if not ss.shortlist:
        st.caption("No players yet. Add players from their profile or the Scouting tab.")
    else:
        for name in list(ss.shortlist):
            r = df.iloc[pid_of(name)]
            row = st.columns([3, 1, 1])
            row[0].markdown(f'<div style="padding-top:.45rem;"><b style="font-family:Archivo;">{r.Player}</b> '
                            f'<span class="meta">· {r.Team} · {r.Position} · {r.Profile} · score {r.Score:.2f}</span></div>', unsafe_allow_html=True)
            if row[1].button("View", key=f"go_sl_{pid_of(name)}", use_container_width=True):
                go_player(pid_of(name))
            if row[2].button("Remove", key=f"sl_rm_{pid_of(name)}", use_container_width=True):
                toggle_shortlist(name); st.rerun()
        a2 = st.columns([1, 1, 3])
        if a2[0].button("Clear all", key="cmp_clear"):
            ss.shortlist = []; st.rerun()
        if a2[1].button("Analyze shortlist", key="ai_sl", disabled=not ai_scout.available() or len(ss.shortlist) < 2):
            ss.ai_sl_out = ai_call("shortlist", tuple(ss.shortlist))
        if len(ss.shortlist) < 2:
            a2[2].caption("Add at least two players to analyse the shortlist.")
        if ss.get("ai_sl_out"):
            ok, txt = ss.ai_sl_out
            with st.container(border=True):
                st.markdown("**AI shortlist analysis**"); st.markdown(txt if ok else f":orange[{txt}]")
    st.markdown('<div class="note" style="margin-top:1rem;">The AI is an interpretation layer: stats, predictions, '
                'profiles and similarity all come from the trained models and saved data, not the language model.</div>',
                unsafe_allow_html=True)
