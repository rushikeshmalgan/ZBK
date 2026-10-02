"""Football Player Analytics & Scouting: Streamlit front end.

Run:  streamlit run app.py
Nothing is trained here. The notebook (football_ml.ipynb) writes models/artifacts.joblib and
models/players_scored.csv, and this app only loads and looks things up.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import plot_tree

MODELS = Path(__file__).parent / "models"
LABELS = {"Shots_p90": "Shots / 90", "Passes_p90": "Passes / 90", "PassAcc": "Pass accuracy %",
          "Tackles_p90": "Tackles / 90", "Interceptions_p90": "Interceptions / 90"}
PROFILE_NOTES = {
    "Attacking": "High shot volume, low passing and defending. Mostly forwards.",
    "Playmaking": "Ball-playing profile: most passes and best pass accuracy. 69% of this group are defenders, so read it as build-up passers rather than only creative midfielders.",
    "Defensive": "Most tackles and good interception numbers. A defender/midfielder mix.",
}

st.set_page_config(page_title="Football Analytics & Scouting", page_icon="⚽", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load(MODELS / "artifacts.joblib")


@st.cache_data
def load_players():
    return pd.read_csv(MODELS / "players_scored.csv")


art = load_artifacts()
df = load_players()
FEATURES = art["features"]
scaled = art["scaler_km"].transform(df[FEATURES])      # same scaled space K-Means and similarity use
df["Label_Text"] = np.where(df.Label == 1, "High", "Low")
df["Display"] = df.Player + " (" + df.Team + ")"


def similar_players(idx, n=5):
    """Nearest players in scaled feature space. Similarity % is relative to the furthest player from the query."""
    d = np.linalg.norm(scaled - scaled[idx], axis=1)
    order = np.argsort(d)[1:n + 1]
    out = df.iloc[order][["Player", "Team", "Position", "Profile"] + FEATURES].copy()
    out.insert(4, "Similarity %", (100 * (1 - d[order] / d.max())).round(1))
    return out.rename(columns=LABELS).round(2).reset_index(drop=True)


st.title("⚽ Football Player Analytics & Scouting")
st.caption(f"Big-5 European leagues, {art['season'] - 1}-{str(art['season'])[2:]} season · {len(df):,} outfield players with ≥ 450 minutes · "
           "all stats are per 90 minutes · models were trained in the notebook, nothing is retrained here")

tab_card, tab_models, tab_profiles, tab_scout = st.tabs(["🧑 Player Card", "📊 Model Analytics", "🧩 Player Profiles", "🔎 Scouting"])

# ---------------------------------------------------------------- Player Card
with tab_card:
    names = df.Display.tolist()
    default = names.index(df[df.Player == "Bukayo Saka"].Display.iloc[0]) if (df.Player == "Bukayo Saka").any() else 0
    choice = st.selectbox("Select player", names, index=default, key="card_player")
    i = df.index[df.Display == choice][0]
    p = df.loc[i]

    with st.container(border=True):
        st.subheader(p.Player)
        st.write(f"{p.Team} · {p.League} · {p.Position} · {int(p.Minutes):,} minutes")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Performance score", f"{p.Score:.2f}", help="Goals/90 + Assists/90 (project-defined score, not an official rating)")
        c2.metric("Model prediction", f"{p.Pred_Score:.2f}", help="Random Forest prediction from the 5 stats below (player was in the training data)")
        c3.metric("Category", p.Pred_Category.upper(), help=f"Logistic Regression: High = score above the median ({art['threshold']:.2f})")
        c4.metric("Profile", p.Profile.upper(), help="K-Means cluster, named from its measured profile")
        st.caption(f"Actual category from the data: {p.Label_Text}. Goals and assists build the score, so the models never see them as inputs.")

    cols = st.columns(5)
    for col, f in zip(cols, FEATURES):
        pct = (df[f] < p[f]).mean() * 100
        col.metric(LABELS[f], f"{p[f]:.1f}" if f == "PassAcc" else f"{p[f]:.2f}", f"{pct:.0f}th percentile", delta_color="off")

    st.subheader("Players with similar profiles")
    st.dataframe(similar_players(i), hide_index=True, use_container_width=True)

    with st.expander("What-if: change this player's stats and see the models react"):
        wi = {}
        sc = st.columns(5)
        for col, f in zip(sc, FEATURES):
            lo, hi = float(df[f].min()), float(df[f].quantile(0.995))
            wi[f] = col.slider(LABELS[f], lo, hi, float(min(max(p[f], lo), hi)), key=f"wi_{f}")
        row = pd.DataFrame([wi])[FEATURES]
        rf_pred = art["reg_models"]["Random Forest"].predict(row)[0]
        lr_pred = art["reg_models"]["Linear Regression"].predict(row)[0]
        prob = art["clf_models"]["Logistic Regression"].predict_proba(row)[0, 1]
        cl = art["cluster_names"][int(art["kmeans"].predict(art["scaler_km"].transform(row))[0])]
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Random Forest score", f"{rf_pred:.2f}")
        w2.metric("Linear Regression score", f"{lr_pred:.2f}")
        w3.metric("P(High performer)", f"{prob:.0%}")
        w4.metric("Cluster", cl)

# ------------------------------------------------------------ Model Analytics
with tab_models:
    st.info("Every model is shown with **two** numbers: the 5-fold cross-validation mean (on the training set) and the score on the "
            "untouched 20% test set. This avoids judging a model from one lucky or unlucky split.")
    cv, reg, clf = art["cv_results"], art["reg_results"], art["clf_results"]

    st.subheader("Regression: predict the performance score")
    r = reg.copy()
    r.insert(0, "CV R² (mean ± std)", [f"{cv.loc[('Regression (R2)', m), 'CV mean']:.3f} ± {cv.loc[('Regression (R2)', m), 'CV std']:.3f}" for m in r.index])
    r = r.rename(columns={"R2": "Test R²"})
    st.dataframe(r, use_container_width=True)
    st.caption("The two regressors are effectively tied: Random Forest's higher test R² is not confirmed by cross-validation, "
               "where Linear Regression is marginally ahead. We do not call either one the winner.")

    st.subheader("Classification: High vs Low performer")
    c = clf.copy()
    c.insert(0, "CV accuracy (mean ± std)", [f"{cv.loc[('Classification (Accuracy)', m), 'CV mean']:.3f} ± {cv.loc[('Classification (Accuracy)', m), 'CV std']:.3f}" for m in c.index])
    c = c.rename(columns={"Accuracy": "Test accuracy"})
    st.dataframe(c, use_container_width=True)
    st.caption("Logistic Regression is ahead of the Decision Tree on both cross-validation and test accuracy.")

    preds, y = art["test_preds"]["clf"], art["test_preds"]["y_clf"]
    cm_cols = st.columns(len(preds))
    for col, (name, pr) in zip(cm_cols, preds.items()):
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ConfusionMatrixDisplay(confusion_matrix(y, pr, labels=[1, 0]), display_labels=["High", "Low"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
        col.pyplot(fig)
        plt.close(fig)

    st.subheader("Explainability: what drives the prediction?")
    imp = art["importances"].rename(index=LABELS)
    e1, e2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    imp["Random Forest importance"].sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title("Random Forest feature importance")
    e1.pyplot(fig); plt.close(fig)
    coef = imp["Linear coefficient (scaled)"].sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    coef.plot.barh(ax=ax, color=["tomato" if v < 0 else "seagreen" for v in coef])
    ax.set_title("Linear Regression coefficients (scaled inputs)")
    e2.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6))
    plot_tree(art["clf_models"]["Decision Tree"], feature_names=[LABELS[f] for f in FEATURES], class_names=["Low", "High"],
              filled=True, rounded=True, fontsize=8, max_depth=3, ax=ax)
    ax.set_title("Decision Tree (top 3 levels shown)")
    st.pyplot(fig); plt.close(fig)
    st.caption("Position strongly shapes these results (forwards shoot more and score more), so part of the accuracy is "
               "'is this player a forward?'. This is a limitation of the current five-stat feature set.")

# ------------------------------------------------------------ Player Profiles
with tab_profiles:
    st.write("K-Means found the groups **without** seeing labels or positions. We named them afterwards from their measured profile.")
    prof = art["cluster_profile"].rename(index=art["cluster_names"])
    for name in prof.index:
        st.markdown(f"**{name}** ({int(prof.loc[name, 'Players'])} players): {PROFILE_NOTES.get(name, '')}")

    k = art["k_table"]
    st.caption(f"Chosen K = {art['k']} (best silhouette among K=3..6, silhouette = "
               f"{k.loc[k.K == art['k'], 'Silhouette'].iloc[0]:.3f}). A silhouette this low means the groups overlap: they are tendencies, not hard boundaries.")

    var = art["pca"].explained_variance_ratio_
    st.subheader(f"PCA view (first two components explain {100 * var.sum():.1f}% of the variance)")
    g1, g2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Profile", palette="tab10", s=16, alpha=.75, ax=ax)
    ax.set_title("Coloured by cluster profile")
    g1.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df.PC1, df.PC2, c=df.Score, cmap="viridis", s=16, alpha=.75, vmax=df.Score.quantile(.98))
    plt.colorbar(sc, ax=ax, label="Performance score"); ax.set_title("Coloured by performance score")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    g2.pyplot(fig); plt.close(fig)

    s1, s2 = st.columns(2)
    s1.subheader("Cluster means (what produced the names)")
    s1.dataframe(prof.rename(columns=LABELS), use_container_width=True)
    s2.subheader("Position mix inside each cluster")
    s2.dataframe(pd.crosstab(df.Profile, df.Position, normalize="index").round(2), use_container_width=True)

    k_fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(k.K, k.Inertia, "o-"); ax[0].set_title("Elbow curve"); ax[0].set_xlabel("K")
    ax[1].plot(k.K, k.Silhouette, "o-", color="darkorange"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("K")
    st.pyplot(k_fig); plt.close(k_fig)

# ------------------------------------------------------------------- Scouting
with tab_scout:
    st.subheader("Scouting filters")
    f1, f2, f3, f4, f5 = st.columns(5)
    pos = f1.selectbox("Position", ["All", "Forward", "Midfielder", "Defender"])
    min_score = f2.slider("Min performance score", 0.0, float(df.Score.max()), 0.30, 0.05)
    min_pass = f3.slider("Min pass accuracy %", 0, 95, 80)
    min_tkl = f4.slider("Min tackles / 90", 0.0, 5.0, 1.5, 0.1)
    prof_sel = f5.selectbox("Player profile", ["All"] + sorted(df.Profile.unique()))

    q = df[(df.Score >= min_score) & (df.PassAcc >= min_pass) & (df.Tackles_p90 >= min_tkl)]
    if pos != "All":
        q = q[q.Position == pos]
    if prof_sel != "All":
        q = q[q.Profile == prof_sel]
    shown = q.sort_values("Score", ascending=False)[["Player", "Team", "Position", "Score", "Pred_Score", "Pred_Category", "Profile", "PassAcc", "Tackles_p90"]]
    st.write(f"**{len(shown)}** players match")
    st.dataframe(shown.rename(columns={"Pred_Score": "Predicted score", "Pred_Category": "Predicted category",
                                       "PassAcc": "Pass %", "Tackles_p90": "Tackles / 90"}).round(2),
                 hide_index=True, use_container_width=True)
    st.caption("Predicted score/category are demonstration outputs for players already in the training data, not held-out evaluation results.")

    st.subheader("Find similar players")
    names = df.Display.tolist()
    d_idx = names.index(df[df.Player == "Jude Bellingham"].Display.iloc[0]) if (df.Player == "Jude Bellingham").any() else 0
    pick = st.selectbox("Select player", names, index=d_idx, key="sim_player")
    n_sim = st.slider("How many", 3, 15, 5, key="sim_n")
    st.dataframe(similar_players(df.index[df.Display == pick][0], n_sim), hide_index=True, use_container_width=True)
    st.caption("Similarity = distance in the standardised 5-stat space. 100% = identical profile, 0% = the most different player in the dataset.")
