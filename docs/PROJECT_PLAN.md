# Football Player Performance Analysis: Mini ML Project Plan

## Progress log
- **Day 1 (2 Oct): DONE.** Dataset = FBref Big-5 player stats (2018-2025, via worldfootballR_data), merged by `scripts/build_dataset.py` into `data/players.csv` (has Player, Team, League, Pos, Minutes, Goals, Assists, Shots, Passes, PassAcc, Tackles, Interceptions). Notebook analyses season 2024 only (one row per player). `football_ml.ipynb` runs top to bottom with zero errors: 1,826 players after cleaning, 80/20 split, all 5 core models produce metrics.
  - Regression test R2: Linear 0.559, Random Forest 0.608. Classification test F1: Logistic 0.797, Tree 0.787. K-Means: K=3, silhouette 0.259.
  - Caveat to discuss in the demo: position strongly drives these results (forwards shoot more and score more), so part of the accuracy is "is this a forward?".

- **Day 2 (2 Oct, done early): Tier 2 core DONE.** Notebook now also has 5-fold CV, cluster naming from measured profiles (Attacking / Playmaking / Defensive), PCA plots (by cluster and by score), `similar_players()`, `scout()` filters, and saves `models/artifacts.joblib` + `models/players_scored.csv`. Zero errors on a fresh kernel run.
  - CV: Linear R2 0.561 +/- 0.040, RF R2 0.552 +/- 0.028 (RF test R2 0.608 was a lucky split: the two regressors are effectively tied). Logistic acc 0.796 +/- 0.026, Tree 0.777 +/- 0.026.
  - Not done (Tier 3, optional): Gradient Boosting, SVM, RF classifier, GridSearchCV.
  - Next: Streamlit app (4 tabs) reading `models/`, then rehearsal.

- **Day 2 (cont.): Streamlit app DONE.** `app.py` has 4 tabs (Player Card with what-if sliders, Model Analytics, Player Profiles, Scouting + similar players). Loads `models/` only, never retrains. Headless AppTest: no exceptions, interactions work; real server health check OK.
  - Remaining: clean-env test, `jupyter nbconvert --to html` backup + screenshots, demo rehearsal (<= 7-8 min), README. Tier 3 models optional.

- **Day 3 (2 Oct, done early): Finalisation DONE.** Added honesty cells to the notebook (position-effect limitation, ball-playing caveat, scouting disclaimer) and a sanity-check cell (passes). Pinned `requirements.txt` to the artifact-matching versions + split out `requirements-dev.txt`. Verified in a fresh venv: app loads with **no** version-mismatch warnings, notebook executes clean. Backups: `docs/football_ml.html` + `docs/figures/` (7 analytical charts + 4 app-tab screenshots, all visually verified). Wrote `README.md`, `.gitignore`. Project tree clean, no stray caches. `.venv_test` removed.
  - **Remaining (user actions):** look at the app in a browser (`streamlit run app.py`); rehearse the demo once; optional `git init` + push to GitHub for Streamlit Cloud deploy. Tier 3 models still optional/untouched.

- **UI redesign (2 Oct): DONE.** Rewrote `app.py` into a custom light "football-editorial" product (no dark theme): top-nav (ink/green), instant accent search (accent-normalised), hero scouting report with SVG score gauge + Plotly radar + percentile attribute bars, clickable similar-player cards, what-if scenario panel, "Player archetypes" blocks, interactive Plotly PCA "player map" (SVG scatter, hover), supporting Analytics, and a scouting grid of rich result cards. Added `.streamlit/config.toml` (forces light theme) and pinned `plotly==7.1.0`. Fixed two real bugs found via screenshots: active nav label invisible, and a global CSS rule that was stomping Streamlit's Material icon font (affected the live expander chevron, not just screenshots). Also fixed README.md (was UTF-16 → now UTF-8). Verified: all pages + interactions pass headless AppTest; pinned clean-venv load has no warnings; 4 fresh screenshots (`docs/figures/ui_*.png`) visually confirmed; old-UI screenshots removed. **Not yet committed/pushed** (last push was the pre-redesign version).

## Context
The PPT (`Football_Player_Analytics_Simplified_ML_Project.pptx.pptx`, 10 slides) defines a small ML learning project: take one player-stats dataset and apply **regression** (Linear Regression vs Random Forest), **classification** (Logistic Regression vs Decision Tree) and **clustering** (K-Means), using one shared pipeline: Data -> Cleaning -> Features -> 80/20 split -> Models -> Evaluation -> Insights. Slide 9 promises: per-player predictions + High/Low label + cluster, model comparison tables, confusion matrix, feature importance, cluster scatter, elbow curve, and an optional Streamlit demo.
Demo date: **5 Oct 2026** (today is 2 Oct, so 3 days). Scope is deliberately small: 5 algorithms, one notebook, one small app.

Decisions made: dataset = player-stats CSV (one row per player, e.g. a Kaggle FBref/European-leagues season file); demo = notebook + small Streamlit app.

## Scale-up (v2, revised): "Football Player Performance Analytics & Scouting System"
Principle: **scale the depth, not the chaos.** Keep the 5 slide models; make them useful. Story: Analyze -> Predict -> Classify -> Discover Profiles -> Find Similar Players -> Scout Players.

```
DATA -> CLEANING -> FEATURES (per-90)
   -> Regression (score) | Classification (High/Low) | Clustering (profiles)
   -> Evaluation (metrics, explainability, 5-fold CV)
   -> Scouting engine (similarity + filters)
   -> Streamlit app (4 tabs)
```

**Tier 1, MUST HAVE (done by end of Sat 3 Oct, then basic app Sun):** dataset, cleaning, per-90 features, the 5 slide models, metrics, core charts (incl. RF feature importance + LR coefficients), notebook, basic Streamlit player card.

**Tier 2, HIGH-VALUE (built once Tier 1 works):**
- **Player similarity:** nearest neighbours in scaled feature space, returns top-5 with a similarity % (e.g. cosine or 1/(1+distance), explained in the notebook).
- **PCA:** 2D scatter, toggle colour by cluster or by performance; explained-variance chart.
- **5-fold cross-validation** on all supervised models (per-fold + mean +/- std). Prioritised over adding algorithms.
- **Scouting filters:** position, min score, pass accuracy, tackles/90, cluster -> ranked table (Player, Score, Category, Cluster).
- **Richer Player Card:** name, position, team (if in CSV), score, category, profile name, the 5 per-90 stats.
- **Model comparison dashboard:** regression and classification tables, confusion matrices, importances, CV. No "best model" declared until results are in; let numbers decide.

**Tier 3, ONLY IF EVERYTHING ELSE IS DONE:** Gradient Boosting (reg), SVM + Random Forest (clf), `GridSearchCV` on RF, multi-season/multi-league CSV, position-vs-cluster crosstab.

**Explicitly out of scope:** deep learning/neural nets, computer vision/video, live football APIs, real-time prediction, LLM chatbot, web scraping, transfer-market prediction.

**Streamlit app, 4 tabs:** (1) Player Card, (2) Model Analytics, (3) Player Profiles (K-Means + PCA + cluster characteristics), (4) Scouting (similar players, filters, player comparison).

## First action after approval
Create `Performance ML/docs/` and save this plan there as `docs/PROJECT_PLAN.md` (verbatim copy of this file, including the timeline). Keep it updated (tick off checkpoints) as the project progresses.

## Deliverables (all in `Performance ML/`)
```
docs/PROJECT_PLAN.md        # this plan + timeline
```
```
data/players.csv            # downloaded stats CSV
football_ml.ipynb           # full analysis, the main demo artifact
app.py                      # Streamlit app: Player Card, Model Analytics, Player Profiles, Scouting
requirements.txt            # pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib, jupyter
models/                     # joblib-saved scaler + best models + cluster model (written by notebook, read by app)
```

## Design choices (follow the slides exactly)
- **Inputs (features):** Shots, Passes, Pass accuracy, Tackles, Interceptions, converted to **per-90** (counts / minutes * 90; pass accuracy stays a %).
- **Target (score):** project-defined formula from Goals + Assists per 90 (e.g. `score = goals_p90 + assists_p90`, optionally weighted). Goals/Assists are **NOT** inputs (slide 4, avoids target leakage). Note: with only these 5 defensive/passing features the R^2 will be modest. That is fine and is itself a talking point (honest results, "to be evaluated").
- **Class label:** `High = score > median(score)`, else Low (balanced classes, per slide 7).
- **Cleaning:** drop duplicates, handle NaNs, filter players with minutes < ~450 (so per-90 rates are stable).
- **Split:** `train_test_split(test_size=0.2, random_state=42)`; for classification add `stratify=y`.
- **Scaling:** `StandardScaler` fit on train only, used for Logistic Regression and K-Means; trees/forest unscaled (slide 5). Use a sklearn `Pipeline` to avoid leakage.
- **Metrics:** regression MAE/RMSE/R^2; classification Accuracy/Precision/Recall/F1 + confusion matrix; clustering inertia (elbow) + silhouette.
- **K-Means:** run K=2..8, pick K from elbow + silhouette (expect 3-4), then profile each cluster (mean per-90 stats) and give human names (e.g. "Playmaker", "Defensive", "Attacker") **after** seeing the numbers.

## Notebook structure (maps 1:1 to slides)
1. Title + problem (markdown)
2. Load data, `head()`, `info()`, `describe()`
3. Cleaning (duplicates, NaN, minutes filter), print rows before/after
4. Feature engineering: per-90 rates, build `score`, build `label`
5. EDA: correlation heatmap, score histogram
6. Train/test split + scaling
7. **Regression:** Linear vs Random Forest -> metrics table, predicted-vs-actual scatter, LR coefficients, RF feature importance bar chart
8. **Classification:** Logistic vs Decision Tree -> metrics table, two confusion-matrix heatmaps, `plot_tree` (max_depth=3)
9. **Clustering:** scale, elbow curve, silhouette per K, final K-Means, 2D PCA scatter (PCA only for plotting), cluster profile table
10. Final summary table (the "Expected Output" table of slide 9, now filled with real numbers) + conclusions + future scope
11. Save artifacts with `joblib.dump` to `models/`

## Streamlit app (`app.py`, ~60 lines)
- Loads `models/*.joblib` and `data/players.csv` (cached with `st.cache_data`/`st.cache_resource`).
- Two modes: pick a player from a dropdown, or adjust sliders for the 5 features.
- Shows the slide-9 card: **Predicted Performance (X.XX)**, **Category (High/Low)**, **Cluster (name)**, top feature importances chart.
- Run: `streamlit run app.py`.

## Timeline (3 days)
Hours are rough effort blocks (about 6-8 working hours/day), not fixed clock times. Each day ends with a **checkpoint** that must pass before moving on.

### Day 1: Fri 2 Oct (today): Foundation + Tier 1 end-to-end
| Block | Task | Output |
|---|---|---|
| 1 (~1h) | Create venv, `requirements.txt`, install deps, create folders (`data/`, `models/`) | env runs `import sklearn, streamlit` |
| 2 (~1h) | Find and download the multi-league player-stats CSV; inspect columns; write the column-mapping dict | `data/players.csv`, mapping confirmed |
| 3 (~1.5h) | Notebook sections 1-4: load, clean (dupes, NaN, minutes filter), per-90 features, `score` + High/Low label | clean DataFrame, row counts printed |
| 4 (~1h) | Section 5-6: EDA (heatmap, score histogram), 80/20 split, scaler | split + scaler ready |
| 5 (~2h) | Sections 7-9 first pass: LR + RF, LogReg + Tree, K-Means (elbow/silhouette) | all 5 core models produce metrics |
**Checkpoint (end of Day 1):** notebook runs top to bottom, 5 metrics tables filled with real numbers. If not, Day 2 morning is spent finishing this before any extras.

### Day 2: Sat 3 Oct: Polish core + Tier 2 extras
| Block | Task | Output |
|---|---|---|
| 1 (~1h) | Polish core charts: predicted-vs-actual, feature importance, confusion matrices, tree plot, cluster scatter; name the clusters from the profile table | presentable charts |
| 2 (~1h) | 5-fold cross-validation on all supervised models | CV per-fold + mean +/- std table |
| 3 (~1h) | PCA plot (colour by cluster / performance) | PCA charts |
| 4 (~1.5h) | Player similarity (nearest neighbours + similarity %) and scouting filter function | `similar_players(name, n=5)`, `scout(filters)` work |
| 5 (~1h) | Save everything with joblib (scaler, models, KMeans, results tables, cluster names, similarity data) | `models/` populated |
| 6 (optional, ~1.5h) | Tier 3 only if all above is done: Gradient Boosting, SVM, RF clf, GridSearchCV | extra rows in tables |
**Checkpoint (end of Day 2):** notebook `Restart & Run All` succeeds; `models/` has all artifacts. Anything unfinished in Tier 2/3 is cut here, not carried forward.

### Day 3: Sun 4 Oct: App + rehearsal + freeze
| Block | Task | Output |
|---|---|---|
| 1 (~1.5h) | `app.py` tab 1: Player Card (dropdown + sliders, richer card) | slide-9 card working |
| 2 (~2h) | Tabs 2-4: Model Analytics, Player Profiles (clusters + PCA), Scouting (filters + similar players + compare) | 4-tab app |
| 3 (~0.5h) | **Feature freeze (midday).** Clean-env test: fresh terminal, `streamlit run app.py`, notebook run-all | no errors |
| 4 (~1h) | Write conclusion + future-scope cells; add short markdown explanations above each section (for the audience) | notebook reads as a story |
| 5 (~1.5h) | Full demo dry run x2 with a timer (7-8 min); prepare Q&A answers; take backup screenshots; copy project to a pen drive / cloud | rehearsed + backed up |
**Checkpoint (end of Day 3):** everything runs offline, backup exists, demo timed.

### Day 4: Mon 5 Oct: Demo day
- Morning: 15-min smoke test (open notebook with outputs, launch Streamlit), no code changes.
- Before presenting: close other apps, increase font/zoom, open notebook and app in separate tabs.
- Fallback: screenshots + exported HTML of the notebook (`jupyter nbconvert --to html`) if anything fails live. Pre-open the notebook (outputs already run) and the Streamlit app; have a backup of the saved outputs/screenshots.

## Demo script (5-7 min)
Problem (30s) -> data & features, leakage point (1 min) -> regression results (1 min) -> classification + confusion matrix (1 min) -> clusters + elbow (1 min) -> live Streamlit player card (1 min) -> conclusion/future scope (30s).

## Risks and mitigations
- **Schedule risk from scale-up:** Tier 1 is finished first; Tier 2 items are independent, so any that slip are simply cut. Feature freeze Sunday midday.
- **CSV column names differ** by source: do one mapping dict at the top of the notebook; pick a CSV that has shots, passes, pass %, tackles, interceptions, minutes, goals, assists.
- **Low R^2:** expected with these features; present as honest result, don't tune to hide it. If it is near zero, add shots-on-target or key passes only if present in the CSV.
- **No internet on demo day:** everything runs locally from the saved CSV and models.
- **Python env:** `python-pptx` etc. are not installed; install from `requirements.txt` in a venv before starting.

## Verification
1. `pip install -r requirements.txt`, then `jupyter nbconvert --to notebook --execute football_ml.ipynb` runs with no errors.
2. Sanity checks in notebook: no NaNs after cleaning, goals/assists absent from feature matrix, scaler fit on train only, class balance ~50/50, silhouette in [-1,1].
3. Results tables show real numbers for all 5 models (no "to be evaluated" left).
4. `streamlit run app.py` opens, a selected player shows score, category and cluster; sliders change the output.
5. Full dry-run of the demo script once on Sunday.
