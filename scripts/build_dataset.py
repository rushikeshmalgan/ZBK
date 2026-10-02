"""One-off: merge FBref Big-5 player tables (worldfootballR_data) into data/players.csv.

Raw .rds files live in data/raw/ (downloaded from
https://github.com/JaseZiv/worldfootballR_data/releases/tag/fb_big5_advanced_season_stats).
Needs `pyreadr` (only for this script, not for the notebook/app).
"""
from pathlib import Path

import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
KEY = ["Season_End_Year", "Squad", "Comp", "Url"]


def load(table, cols):
    df = list(pyreadr.read_r(RAW / f"big5_player_{table}.rds").values())[0]
    return df[KEY + cols].drop_duplicates(subset=KEY)


standard = load("standard", ["Player", "Pos", "Age", "Min_Playing", "Gls", "Ast"])
shooting = load("shooting", ["Sh_Standard"])
passing = load("passing", ["Att_Total", "Cmp_percent_Total"])
defense = load("defense", ["Tkl_Tackles", "Int"])

df = standard
for part in (shooting, passing, defense):
    df = df.merge(part, on=KEY, how="inner")

df = df.rename(columns={
    "Season_End_Year": "Season", "Squad": "Team", "Comp": "League",
    "Min_Playing": "Minutes", "Gls": "Goals", "Ast": "Assists",
    "Sh_Standard": "Shots", "Att_Total": "Passes", "Cmp_percent_Total": "PassAcc",
    "Tkl_Tackles": "Tackles", "Int": "Interceptions",
}).drop(columns=["Url"])

df = df[["Player", "Team", "League", "Season", "Pos", "Age", "Minutes", "Goals", "Assists",
         "Shots", "Passes", "PassAcc", "Tackles", "Interceptions"]]
df.to_csv(ROOT / "data" / "players.csv", index=False)
print(df.shape)
print(df.groupby("Season").size())
