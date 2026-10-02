"""One-off: merge FBref Big-5 player tables (worldfootballR_data) into data/players.csv.

Raw .rds files live in data/raw/ (downloaded from
https://github.com/JaseZiv/worldfootballR_data/releases/tag/fb_big5_advanced_season_stats).
Needs `pyreadr` (only for this script, not for the notebook/app).

Merges six tables into one row per player-season with a rich, role-descriptive feature set:
standard, shooting, passing, possession, defense, misc.

Note on availability: FBref no longer publishes player "pressures" (StatsBomb removed them in 2023),
so no pressing metrics are included. The original five baseline columns (Shots, Passes, PassAcc,
Tackles, Interceptions + Goals/Assists) are kept so the baseline experiment still runs unchanged.
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


standard = load("standard", ["Player", "Pos", "Age", "Min_Playing", "Gls", "Ast", "G_minus_PK"])
shooting = load("shooting", ["Sh_Standard", "SoT_Standard", "xG_Expected", "npxG_Expected"])
passing = load("passing", ["Att_Total", "Cmp_percent_Total", "Cmp_percent_Long", "KP",
                           "Final_Third", "PPA", "PrgP", "xAG"])
possession = load("possession", ["Touches_Touches", "Att Pen_Touches", "Att_Take", "Succ_Take",
                                 "Succ_percent_Take", "Carries_Carries", "PrgC_Carries",
                                 "Final_Third_Carries", "CPA_Carries", "Mis_Carries", "Dis_Carries",
                                 "PrgR_Receiving"])
defense = load("defense", ["Tkl_Tackles", "TklW_Tackles", "Blocks_Blocks", "Int", "Clr",
                           "Tkl_percent_Challenges"])
misc = load("misc", ["Fls", "Fld", "Won_Aerial", "Lost_Aerial", "Won_percent_Aerial", "Recov"])

df = standard
for part in (shooting, passing, possession, defense, misc):
    df = df.merge(part, on=KEY, how="inner")

df = df.rename(columns={
    "Season_End_Year": "Season", "Squad": "Team", "Comp": "League",
    "Min_Playing": "Minutes", "Gls": "Goals", "Ast": "Assists", "G_minus_PK": "NpGoals",
    # shooting
    "Sh_Standard": "Shots", "SoT_Standard": "ShotsOnTarget", "xG_Expected": "xG", "npxG_Expected": "npxG",
    # passing
    "Att_Total": "Passes", "Cmp_percent_Total": "PassAcc", "Cmp_percent_Long": "LongPassAcc",
    "KP": "KeyPasses", "Final_Third": "PassesFinalThird", "PPA": "PassesPenArea", "PrgP": "ProgPasses", "xAG": "xAG",
    # possession
    "Touches_Touches": "Touches", "Att Pen_Touches": "AttPenTouches", "Att_Take": "TakeOns",
    "Succ_Take": "TakeOnsWon", "Succ_percent_Take": "TakeOnPct", "Carries_Carries": "Carries",
    "PrgC_Carries": "ProgCarries", "Final_Third_Carries": "CarriesFinalThird", "CPA_Carries": "CarriesPenArea",
    "Mis_Carries": "Miscontrols", "Dis_Carries": "Dispossessed", "PrgR_Receiving": "ProgReceptions",
    # defense
    "Tkl_Tackles": "Tackles", "TklW_Tackles": "TacklesWon", "Blocks_Blocks": "Blocks", "Int": "Interceptions",
    "Clr": "Clearances", "Tkl_percent_Challenges": "TackleSuccessPct",
    # misc
    "Fls": "Fouls", "Fld": "Fouled", "Won_Aerial": "AerialsWon", "Lost_Aerial": "AerialsLost",
    "Won_percent_Aerial": "AerialWinPct", "Recov": "Recoveries",
}).drop(columns=["Url"])

order = ["Player", "Team", "League", "Season", "Pos", "Age", "Minutes",
         "Goals", "Assists", "NpGoals", "Shots", "ShotsOnTarget", "xG", "npxG",
         "Passes", "PassAcc", "LongPassAcc", "KeyPasses", "PassesFinalThird", "PassesPenArea", "ProgPasses", "xAG",
         "Touches", "AttPenTouches", "TakeOns", "TakeOnsWon", "TakeOnPct",
         "Carries", "ProgCarries", "CarriesFinalThird", "CarriesPenArea", "Miscontrols", "Dispossessed", "ProgReceptions",
         "Tackles", "TacklesWon", "Blocks", "Interceptions", "Clearances", "TackleSuccessPct",
         "Fouls", "Fouled", "AerialsWon", "AerialsLost", "AerialWinPct", "Recoveries"]
df = df[order]
df.to_csv(ROOT / "data" / "players.csv", index=False)
print(df.shape)
print("columns:", len(df.columns))
print(df.groupby("Season").size())
