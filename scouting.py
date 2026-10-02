"""Scouting data-access layer (expanded representation).

Pure functions over the saved artifacts (models/players_scored.csv + artifacts.joblib). They return
plain JSON-serialisable Python so the same functions back both the Streamlit UI and the Gemini layer.

Similarity / profiling use the standardised percentile space saved as artifacts['sim_space'] (aligned to
the players_scored row order), so results exactly match the notebook. Nothing here trains or invents data.
"""
import functools
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS = Path(__file__).parent / "models"

# curated headline metrics for a compact player record (full set still available via the UI)
HEADLINE = {"xG_p90": "xg_per90", "Shots_p90": "shots_per90", "KeyPasses_p90": "key_passes_per90",
            "xAG_p90": "xag_per90", "ProgPasses_p90": "prog_passes_per90", "ProgCarries_p90": "prog_carries_per90",
            "Passes_p90": "passes_per90", "PassAcc": "pass_accuracy", "Tackles_p90": "tackles_per90",
            "Interceptions_p90": "interceptions_per90", "AerialWinPct": "aerial_win_pct"}


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()


@functools.lru_cache(maxsize=1)
def load():
    """Load once (memoised). Returns a dict bundle shared across UI + AI."""
    art = joblib.load(MODELS / "artifacts.joblib")
    df = pd.read_csv(MODELS / "players_scored.csv").reset_index(drop=True)
    df["norm"] = df.Player.map(_norm)
    prof = art["profiling_features"]
    sim = np.asarray(art["sim_space"], dtype="float32")              # standardised percentile space
    pct_overall = (df[prof].rank(pct=True) * 100).round(1)          # overall percentiles
    pct_pos = (df.groupby("Position")[prof].rank(pct=True) * 100).round(1)   # within-position
    return {"art": art, "df": df, "sim": sim, "prof": prof, "cats": art["category_cols"],
            "groups": art["groups"], "pct_overall": pct_overall, "pct_pos": pct_pos}


def _idx(name):
    b = load(); df = b["df"]; q = _norm(name)
    exact = df[df.norm == q]
    hit = exact if not exact.empty else df[df.norm.str.contains(q, regex=False)]
    return None if hit.empty else int(hit.sort_values("Score", ascending=False).index[0])


def _row_dict(i):
    b = load(); r = b["df"].iloc[i]
    rec = {"player": r.Player, "team": r.Team, "league": r.League, "position": r.Position,
           "performance_score": round(float(r.Score), 3), "model_prediction": round(float(r.Pred_Score), 3),
           "category": r.Pred_Category, "profile": r.Profile}
    rec.update({v: round(float(r[k]), 2) for k, v in HEADLINE.items() if k in r})
    rec["category_indices"] = {c.replace("idx_", "").lower(): round(float(r[c])) for c in b["cats"]}
    return rec


# ---------------------------------------------------------------- tool functions
def search_players(query: str, limit: int = 8) -> list:
    """Search players by (partial, accent-insensitive) name. Returns a list of player summaries."""
    b = load(); q = _norm(query)
    if not q:
        return []
    hit = b["df"][b["df"].norm.str.contains(q, regex=False)].sort_values("Score", ascending=False).head(int(limit))
    return [_row_dict(i) for i in hit.index]


def get_player(player_name: str) -> dict:
    """Full record for one player: measured per-90 stats, ML prediction, role profile, category indices (0-100),
    and overall + within-position percentiles for headline metrics."""
    b = load(); i = _idx(player_name)
    if i is None:
        return {"error": f"No player found matching '{player_name}'."}
    out = _row_dict(i)
    out["percentiles_overall"] = {HEADLINE[k]: int(b["pct_overall"].iloc[i][k]) for k in HEADLINE if k in b["pct_overall"].columns}
    out["percentiles_in_position"] = {HEADLINE[k]: int(b["pct_pos"].iloc[i][k]) for k in HEADLINE if k in b["pct_pos"].columns}
    return out


def find_similar_players(player_name: str, limit: int = 5, same_position: bool = False) -> list:
    """Players most similar in playing style (standardised percentile space). same_position restricts to the
    same broad position."""
    b = load(); i = _idx(player_name)
    if i is None:
        return [{"error": f"No player found matching '{player_name}'."}]
    sim, df = b["sim"], b["df"]
    d = np.linalg.norm(sim - sim[i], axis=1)
    order = [j for j in np.argsort(d) if j != i]
    if same_position:
        order = [j for j in order if df.Position.iloc[j] == df.Position.iloc[i]]
    out = []
    for j in order[:int(limit)]:
        rec = _row_dict(j)
        rec["similarity_pct"] = round(float(100 * (1 - d[j] / d.max())), 1)
        out.append(rec)
    return out


def similarity_reasons(name_a: str, name_b: str, n: int = 3) -> dict:
    """Why two players are (dis)similar: the profiling features where their overall percentiles are closest / furthest."""
    b = load(); ia, ib = _idx(name_a), _idx(name_b)
    if ia is None or ib is None:
        return {"shared": [], "different": []}
    po = b["pct_overall"]
    diff = (po.iloc[ia] - po.iloc[ib]).abs().sort_values()
    nice = lambda f: f.replace("_p90", "/90").replace("Acc", " accuracy")
    return {"shared": [nice(f) for f in diff.index[:n]], "different": [nice(f) for f in diff.index[-n:][::-1]]}


def scout_players(position: str = "Any", min_score: float = 0.0, min_pass_accuracy: float = 0.0,
                  min_tackles: float = 0.0, profile: str = "Any", min_xg: float = 0.0,
                  min_prog_carries: float = 0.0, limit: int = 12) -> list:
    """Filter players. position: Any/Forward/Midfielder/Defender. profile: one of the role profiles or Any.
    Thresholds are per-90 (min_xg, min_prog_carries) or % (min_pass_accuracy). Sorted by performance score."""
    b = load(); df = b["df"]
    q = df[(df.Score >= min_score) & (df.PassAcc >= min_pass_accuracy) & (df.Tackles_p90 >= min_tackles) &
           (df.xG_p90 >= min_xg) & (df.ProgCarries_p90 >= min_prog_carries)]
    if position and position != "Any":
        q = q[q.Position == position.title()]
    if profile and profile != "Any":
        q = q[q.Profile == profile]
    q = q.sort_values("Score", ascending=False).head(int(limit))
    return [_row_dict(i) for i in q.index]


def compare_players(player_a: str, player_b: str) -> dict:
    """Compare two players: records, category indices, similarity %, and the stats that most match / differ."""
    b = load(); ia, ib = _idx(player_a), _idx(player_b)
    if ia is None or ib is None:
        miss = player_a if ia is None else player_b
        return {"error": f"No player found matching '{miss}'."}
    sim = b["sim"]
    dist = float(np.linalg.norm(sim[ia] - sim[ib]))
    pct = round(100 * (1 - dist / float(np.linalg.norm(sim - sim[ia], axis=1).max())), 1)
    return {"player_a": get_player(player_a), "player_b": get_player(player_b),
            "similarity_pct": pct, "reasons": similarity_reasons(player_a, player_b)}
