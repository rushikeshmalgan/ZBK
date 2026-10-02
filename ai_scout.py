"""Optional Gemini "AI Scout" layer.

This is an interpretation / orchestration layer on top of the ML artifacts. It NEVER invents
players or statistics: for natural-language search it calls the scouting.py functions (Gemini
function-calling) and only explains what they return; for reports it is handed the real numbers.

The whole app works without this module: if the SDK or GEMINI_API_KEY is missing, available()
returns False and the UI shows an "AI Scout unavailable" notice instead of crashing.

Config (never hard-code the key):
  GEMINI_API_KEY   - required to enable AI features (env var or .streamlit/secrets.toml)
  GEMINI_MODEL     - optional, defaults to a cost-efficient Flash model
"""
import json
import os
from pathlib import Path

import scouting


def _load_dotenv():
    """Best-effort: load KEY=VALUE lines from a local .env into os.environ (no dependency)."""
    f = Path(__file__).parent / ".env"
    if not f.exists():
        return
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass


_load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"
TOOLS = [scouting.search_players, scouting.get_player, scouting.find_similar_players,
         scouting.scout_players, scouting.compare_players]

SYSTEM = (
    "You are an analytical football scouting assistant for a classical-ML scouting app. "
    "You do not invent player statistics. You do not invent players. You do not claim the "
    "machine-learning model proves a player is objectively better. You explain evidence from the "
    "supplied dataset only. All numerical claims must come from the supplied application data or tool "
    "results. Clearly distinguish: measured statistics, ML predictions, cluster/profile assignments, "
    "similarity scores, and your natural-language interpretation. The dataset is Big-5 European league "
    "outfield players, 2023-24 season, 5 per-90 features; the performance score = goals+assists per 90 "
    "(not an official rating) and position strongly influences the stats. If the data cannot answer a "
    "question, say so plainly. Be concise and use short headed sections."
)


def _secret(name):
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def _model_name():
    return _secret("GEMINI_MODEL") or DEFAULT_MODEL


def available() -> bool:
    """True only if the SDK is importable AND an API key is configured."""
    if not _secret("GEMINI_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except Exception:
        return False


def _client():
    from google import genai
    return genai.Client(api_key=_secret("GEMINI_API_KEY"))


def _generate(contents, use_tools=False, max_tokens=900):
    """Return (ok, text). Never raises; failures come back as (False, message).

    Tool path (natural-language search) uses the Chat API, the SDK's recommended way to run
    automatic function calling; plain report prompts use models.generate_content.
    """
    try:
        from google.genai import types
        client = _client()
        if use_tools:
            cfg = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.3,
                                              max_output_tokens=max_tokens, tools=TOOLS)
            chat = client.chats.create(model=_model_name(), config=cfg)
            resp = chat.send_message(contents)
        else:
            cfg = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.3,
                                              max_output_tokens=max_tokens)
            resp = client.models.generate_content(model=_model_name(), contents=contents, config=cfg)
        text = (resp.text or "").strip()
        return (True, text) if text else (False, "The AI returned an empty response. Try rephrasing.")
    except Exception as e:  # network, auth, quota, bad model name, ...
        return False, f"AI Scout error: {type(e).__name__}. Check the API key / model / connection."


# ---------------------------------------------------------------- public helpers
def natural_search(query: str):
    """Natural-language scouting. Gemini calls the scouting.py tools and explains the real results."""
    prompt = ("A user asked: \"" + query + "\"\n"
              "Use the available tools to fetch real candidates from the dataset, then explain the "
              "shortlist: who they are, the measured stats that match the request, and any caveats. "
              "List the specific players the tools returned.")
    return _generate(prompt, use_tools=True)


def player_report(player_name: str):
    """Structured scouting report for one player, from that player's real record."""
    data = scouting.get_player(player_name)
    if "error" in data:
        return False, data["error"]
    similar = scouting.find_similar_players(player_name, 5)
    prompt = ("Write a scouting report for this player using ONLY the data below. Use these headed "
              "sections: PLAYER OVERVIEW, STATISTICAL STRENGTHS, STATISTICAL WEAKNESSES, PLAYING PROFILE, "
              "SIMILAR PLAYERS, SCOUTING INTERPRETATION, DATA LIMITATIONS. Quote percentiles for strengths/"
              "weaknesses. Do not invent numbers.\n\n"
              f"PLAYER DATA:\n{json.dumps(data, indent=2)}\n\nSIMILAR PLAYERS:\n{json.dumps(similar, indent=2)}")
    return _generate(prompt, max_tokens=900)


def compare_report(player_a: str, player_b: str):
    """Natural-language comparison of two players from their real records."""
    data = scouting.compare_players(player_a, player_b)
    if "error" in data:
        return False, data["error"]
    prompt = ("Compare these two players using ONLY the data below. Cover measurable differences in the "
              "five per-90 stats and percentiles, their profiles and ML predicted scores, and what their "
              "similarity score implies. Do not declare one objectively 'better'. End with DATA LIMITATIONS.\n\n"
              f"{json.dumps(data, indent=2)}")
    return _generate(prompt, max_tokens=800)


def shortlist_report(player_names):
    """Analyse a shortlist from the players' real records."""
    players = [scouting.get_player(n) for n in player_names]
    players = [p for p in players if "error" not in p]
    if len(players) < 2:
        return False, "Add at least two valid players to the shortlist first."
    prompt = ("Analyse this shortlist using ONLY the data below. Use sections: COMMON CHARACTERISTICS, "
              "KEY DIFFERENCES, POSSIBLE COMPLEMENTARY PROFILES, DATA-BASED OBSERVATIONS. Do not rank the "
              "players with subjective claims; base everything on the supplied stats.\n\n"
              f"{json.dumps(players, indent=2)}")
    return _generate(prompt, max_tokens=900)
