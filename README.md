# Football Player Performance Analytics & Scouting System

A small but complete machine-learning project that turns real football player
statistics into predictions, categories, player profiles and a scouting tool.
It demonstrates the full ML workflow — **regression, classification and
clustering** — on one dataset, wrapped in a custom interactive Streamlit dashboard.

> Built as a mini ML project for an academic demonstration. The goal is a clear,
> honest, reproducible pipeline, not a production scouting product.

## Overview

A single match produces thousands of events. Comparing hundreds of players by
hand across many statistics is slow and subjective. This project treats each
player's season as a row of numbers and asks three questions:

| Question | ML task | Output |
| --- | --- | --- |
| How good was a player this season? | Regression | A continuous **performance score** |
| Is a player a high or low performer? | Classification | A **High / Low** label |
| Which players have similar styles? | Clustering | **Player profiles** (groups) |

On top of these it adds a **similarity finder** ("players like X") and a
**scouting filter**, and presents everything in a four-section app.

## Features

- **Performance prediction** — Linear Regression vs Random Forest.
- **High / Low classification** — Logistic Regression vs Decision Tree.
- **Rich feature system** — ~29 metrics grouped into football concepts (shooting, creation, passing,
  progression, carrying, defending, aerial), with per-90 rates, percentile transforms and position-aware
  normalisation. Proven against the 5-feature baseline by cross-validation (see Results).
- **Role-aware profiling** — K-Means on the style space (position not an input) → data-driven roles
  (e.g. Finisher, Progressor, Ball-playing/Defensive defender), named from measured category indices.
- **PCA visualisation** — interactive 2D "player map", coloured by role or by score.
- **Player similarity** — nearest players in the standardised ~29-feature style space, with a
  same-position mode and an explanation of which stats match / differ.
- **Fast search** — a Trie (prefix-tree) autocomplete index built once at load (see `search_index.py`).
- **Scouting** — filter by position, score, pass accuracy, tackles and profile.
- **Explainability** — feature importance, regression coefficients, a decision tree.
- **Cross-validation** — 5-fold CV alongside the single test-split results.
- **Player comparison** — any two players head-to-head (radar overlay, attribute table, similarity).
- **Scouting shortlist** — save players during a session, then review or AI-analyse them.
- **AI Scout (optional)** — natural-language search, player reports, comparisons and shortlist
  analysis via Google Gemini; strictly an interpretation layer over the real data (see below).
- **Streamlit dashboard** — a custom light, football-editorial UI (not default Streamlit)
  with a top nav, instant player search, an SVG score gauge, an interactive radar and PCA
  map (Plotly), clickable similar-player cards, a what-if scenario panel, and six sections:
  Players (scouting report), Analytics, Profiles, Scouting, Compare, AI Scout.

## Dataset

- **Source:** FBref statistics for the Big-5 European leagues (Premier League,
  La Liga, Serie A, Bundesliga, Ligue 1), via the public
  [`worldfootballR_data`](https://github.com/JaseZiv/worldfootballR_data) project.
- **Range downloaded:** seasons 2018-2025. [scripts/build_dataset.py](scripts/build_dataset.py)
  merges the standard, shooting, passing and defense tables into
  [data/players.csv](data/players.csv) (one row per player per season).
- **Used for the experiment:** the **2024 season only** (2023-24). This keeps one
  record per player, so the same player can never appear in both the training and
  test sets.
- **After cleaning:** **1,826 outfield players** (goalkeepers removed, at least
  450 minutes played, no missing values). The High/Low target is balanced ~50/50.

## ML methodology

- **Two representations.** A **5-feature baseline** (the original experiment, kept unchanged) and an
  **expanded ~29-feature** set from six FBref tables, grouped into football concepts. Per-90 rates are
  `count / minutes * 90`; profiling/similarity use **percentile ranks** (robust to scale/outliers) plus
  **within-position** percentiles. Separate feature spaces are used for performance, classification,
  profiling and similarity so the system isn't over-dependent on goals+assists.
  *Pressing metrics are intentionally absent — FBref no longer publishes player pressures.*
- **Baseline features (model inputs):** Shots/90, Passes/90, Pass accuracy %, Tackles/90,
  Interceptions/90.
- **Target (performance score):** `Goals/90 + Assists/90`. This is a
  project-defined score, **not** an official rating.
- **No target leakage:** Goals and Assists build the target, so they are
  **never** used as model inputs.
- **Classification label:** `High = score > median(score)`, else `Low`.
- **Split:** `train_test_split(test_size=0.2, random_state=42)`, stratified on the
  label for classification.
- **Scaling:** `StandardScaler` inside a `Pipeline` (fit on training data only)
  for Logistic Regression, K-Means and the similarity search. Tree-based models
  are left unscaled.
- **Metrics:** MAE / RMSE / R2 (regression); Accuracy / Precision / Recall / F1 +
  confusion matrix (classification); inertia (elbow) + silhouette (clustering).

## Results (actual, measured)

**Regression** (20% test set, and 5-fold CV on the training set):

| Model | CV R2 (mean +/- std) | Test R2 | MAE | RMSE |
| --- | --- | --- | --- | --- |
| Linear Regression | 0.561 +/- 0.040 | 0.559 | 0.111 | 0.153 |
| Random Forest | 0.552 +/- 0.028 | 0.608 | 0.104 | 0.144 |

**Classification:**

| Model | CV accuracy (mean +/- std) | Test accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.796 +/- 0.026 | 0.806 | 0.837 | 0.760 | 0.797 |
| Decision Tree | 0.777 +/- 0.026 | 0.798 | 0.830 | 0.749 | 0.787 |

**Baseline vs expanded classification** (5-fold CV on the training set, Logistic Regression):

| Representation | CV accuracy | CV F1 | CV ROC-AUC |
| --- | --- | --- | --- |
| Baseline (5 features) | 0.796 | 0.783 | 0.879 |
| Expanded-A (~25 style features, no xG) | 0.831 | 0.825 | 0.907 |
| Expanded-B (+ xG / xAG) | 0.840 | 0.832 | 0.920 |

The richer representation gives a genuine, cross-validated lift (not just a lucky test split).
Expanded-B adds xG/xAG, which are outcome-correlated, so its edge is expected and flagged.

**Role-aware clustering (expanded):** K = 4 data-driven roles on the ~29-feature style space —
*Finisher, Progressor, Defensive defender, Aerial defender* — named from each cluster's measured
category indices (position validates them but is not an input). Richer similarity is markedly more
plausible: e.g. Saka's nearest styles become Barcola / Martinelli / Riquelme (wingers), where the old
5-feature space returned unrelated names.

## Limitations

Stated honestly — these are discussion points, not flaws to hide:

- **Position effect.** Player position is a strong source of structure in the
  data. Some of the classification and clustering signal reflects positional
  differences (forwards shoot and score more) rather than an independent measure
  of overall player quality.
- **Overlapping clusters.** A silhouette of ~0.26 means the clusters overlap;
  K = 3 was chosen as a useful level of detail, not because the groups are
  cleanly separated.
- **Regressors are tied.** Random Forest's higher test R2 (0.608) is not
  confirmed by cross-validation (0.552 vs 0.561 for Linear Regression), so
  neither model is declared the winner.
- **In-sample scouting outputs.** Displayed player predictions demonstrate model
  inference on the available player dataset; they are **not** a held-out
  evaluation.
- **More features ≠ automatically better.** The expanded set helps here (shown by CV), but a richer
  representation is not guaranteed to classify better; it is limited by data coverage, metric
  availability, position labels, sample size, league/tactical/team context and role ambiguity.
- **Roles overlap.** K = 4 is a defensible level of detail; finer roles (e.g. distinct playmaker vs
  box-to-box midfield) are not cleanly separable in this data, so we don't invent them.
- **A statistical profile is not complete ability.** It describes measurable on-ball/defensive output,
  not everything that makes a footballer good.

## AI Scout (optional Gemini layer)

The app has an optional "AI Scout" powered by Google Gemini. **It is an interpretation /
orchestration layer, not a source of truth.**

- **ML and data are the source of truth.** ML generates the quantitative outputs (scores,
  predictions, clusters, similarity); Gemini only turns those into natural-language scouting
  assistance. It is instructed never to invent players or statistics.
- **Function calling.** For natural-language search, Gemini calls the app's own functions in
  [scouting.py](scouting.py) (`search_players`, `get_player`, `find_similar_players`,
  `scout_players`, `compare_players`), which read `models/players_scored.csv`. Gemini explains
  only what those functions return. Reports and comparisons are handed the real records directly.
- **Architecture:** `data → ML models → scouting.py functions → Gemini → user`.

**Setup** (the app works fully without this — AI features simply stay disabled):

1. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Provide it as `GEMINI_API_KEY` via **either** `.streamlit/secrets.toml` (copy
   [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example)) **or** a `.env` file
   (copy [.env.example](.env.example)) **or** a real environment variable. Optionally set
   `GEMINI_MODEL` (default `gemini-2.5-flash`).
3. On Streamlit Community Cloud, paste the key into the app's **Secrets** box.

**Security:** the key is read only from secrets/env — never hard-coded, printed, or committed.
`.env` and `.streamlit/secrets.toml` are git-ignored. **Cost:** Gemini is called only on explicit
user actions (ask / generate report / compare / analyse shortlist), results are cached per session,
and only the relevant player(s) or filtered candidates are sent — never the whole dataset. **Fallback:**
if the key is missing or a call fails, the app shows a notice and everything else keeps working.

## Installation

Requires Python 3.11+.

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2a. to RUN THE APP only:
pip install -r requirements.txt

# 2b. to also RUN THE NOTEBOOK / rebuild data:
pip install -r requirements-dev.txt
```

Runtime package versions are pinned in [requirements.txt](requirements.txt) to
match the saved models, so loading them never raises a version-mismatch warning.

## Running

**The app** (works offline from the committed data and models — nothing is
downloaded or retrained at runtime):

```bash
streamlit run app.py
```

**The notebook** (reproduces all models and regenerates `models/`):

```bash
jupyter notebook football_ml.ipynb      # then Restart & Run All
# or headless:
jupyter nbconvert --to notebook --execute football_ml.ipynb --inplace
```

## Project structure

```
Performance ML/
├── app.py                     # Streamlit dashboard (loads models/, never retrains)
├── scouting.py                # scouting functions over the saved data (UI + AI tools)
├── search_index.py            # Trie (prefix-tree) + hash-map player search index
├── ai_scout.py                # optional Gemini layer (interpretation only)
├── football_ml.ipynb          # full analysis: clean -> features -> models -> save
├── requirements.txt           # pinned runtime deps (app + Streamlit Cloud)
├── requirements-dev.txt       # + jupyter/nbconvert/pyreadr for the notebook & rebuild
├── .env.example               # template for GEMINI_API_KEY (copy to .env, git-ignored)
├── .streamlit/
│   ├── config.toml            # forces the light theme + primary colour
│   └── secrets.toml.example   # template for the key (copy to secrets.toml, git-ignored)
├── data/
│   └── players.csv            # merged FBref dataset (one row per player-season)
├── models/
│   ├── artifacts.joblib       # models, scaler, K-Means, PCA, result tables, cluster names
│   └── players_scored.csv     # every player with predictions, cluster and PCA coords
├── scripts/
│   └── build_dataset.py       # rebuild data/players.csv from the raw .rds tables
└── docs/
    ├── PROJECT_PLAN.md         # plan, timeline and progress log
    ├── football_ml.html        # rendered notebook (demo backup)
    └── figures/                # analytical charts + app screenshots (demo backup)
```

## Architecture

```
football_ml.ipynb  --(train)-->  models/artifacts.joblib + models/players_scored.csv
                                               |
                                        (load, no retrain)
                                               v
                            scouting.py  <-->  app.py  -->  Streamlit (6 sections)
                                 ^
                                 | (function calling, optional)
                             ai_scout.py  -->  Google Gemini
```

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (`data/` and `models/` are committed; the raw
   `.rds` files in `data/raw/` are git-ignored and not needed at runtime).
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `app.py` on the default branch.
3. Streamlit Cloud installs `requirements.txt` (pinned), loads the saved
   artifacts and starts — no training, no backend, no database.
4. (Optional) to enable AI Scout, paste `GEMINI_API_KEY` into the app's **Secrets** box.

## Future scope

Not implemented; natural next steps: more features (xG, progressive passes),
multiple seasons, position-aware modelling, Gradient Boosting / SVM, and
hyperparameter tuning with `GridSearchCV`.
