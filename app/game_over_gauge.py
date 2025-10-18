#!/usr/bin/env python3
import os, sys, json, math, logging, time, re
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
from io import StringIO
from html import escape

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from jinja2 import Template

# Try the OpenAI SDK (DeepSeek is OpenAI-compatible)
try:
    from openai import OpenAI  # type: ignore
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# =========================
# Logging
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# =========================
# Constants & Defaults
# =========================
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = {
    "DGS10": {"desc": "US 10Y Treasury Yield (Nominal, %)", "units": "pct"},
    "DFII10": {"desc": "US 10Y TIPS Real Yield (%, CPI-adjusted)", "units": "pct"},
    "BAMLH0A0HYM2": {"desc": "US High Yield OAS (%, BofA ICE)", "units": "pct"},
    "VIXCLS": {"desc": "VIX (Implied Volatility)", "units": "idx"},
}
TP_SERIES_FRED = "THREEFYTP10"
TP_SERIES_PROXY = "T10Y2Y"
NYFED_TP_CSV = "https://www.newyorkfed.org/medialibrary/research/data_indicators/ACMTermPremium_Daily_Levels.csv"

DEFAULT_WEIGHTS = {
    "DGS10": 0.20,
    "DFII10": 0.25,
    "BAMLH0A0HYM2": 0.30,
    "VIXCLS": 0.15,
    "TERM_PREMIUM_10Y": 0.10,
}

NORMALIZERS = {
    "DGS10": (2.5, 7.0),
    "DFII10": (0.0, 3.0),
    "BAMLH0A0HYM2": (3.0, 9.0),
    "VIXCLS": (12.0, 45.0),
    "TERM_PREMIUM_10Y": (-1.0, 2.0),
    "TERM_PREMIUM_PROXY": (-1.0, 2.0),
}

TREND_LOOKBACK_DAYS = 5
TREND_PENALTIES = {
    "DGS10": (0.25, 3.0),
    "DFII10": (0.25, 4.0),
    "BAMLH0A0HYM2": (0.50, 5.0),
    "VIXCLS": (5.0, 4.0),
    "TERM_PREMIUM_10Y": (0.25, 2.0),
    "TERM_PREMIUM_PROXY": (0.25, 2.0),
}
MAX_TREND_PENALTY = 10.0

DEFAULT_OUTPUT_HTML = os.getenv("OUTPUT_HTML", "dashboard.html")
DEFAULT_OUTPUT_JSON = os.getenv("OUTPUT_JSON", "gauge.json")
DEFAULT_OUTPUT_YIELDS = os.getenv("OUTPUT_YIELDS_JSON", "yields.json")

USER_AGENT = {"User-Agent": "game-over-gauge/2.6 (+local)"}

# =========================
# Gauge bands
# =========================
BANDS = [
    {"lo": 0,  "hi": 20, "name": "Safe",         "color": "#16a34a"},
    {"lo": 20, "hi": 40, "name": "Cautious",     "color": "#84cc16"},
    {"lo": 40, "hi": 60, "name": "Risky",        "color": "#f59e0b"},
    {"lo": 60, "hi": 80, "name": "Nearly Crash", "color": "#f97316"},
    {"lo": 80, "hi": 100,"name": "Game Over",    "color": "#dc2626"},
]

# =========================
# Helpers
# =========================
def getenv_float(name: str, default: Optional[float]) -> Optional[float]:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def getenv_int(name: str, default: Optional[int]) -> Optional[int]:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def normalize(value: float, key: str) -> float:
    lo, hi = NORMALIZERS[key]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if hi == lo:
        return 0.0
    score = (value - lo) / (hi - lo)
    return clamp01(score) * 100.0

def fetch_fred_series(series_id: str, api_key: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": (datetime.utcnow() - timedelta(days=365*5)).strftime("%Y-%m-%d"),
    }
    r = requests.get(FRED_BASE, params=params, timeout=20, headers=USER_AGENT)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        raise RuntimeError(f"No observations for {series_id}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df

def latest_and_change(df: pd.DataFrame, days: int = TREND_LOOKBACK_DAYS) -> Tuple[float, Optional[float]]:
    if df.empty:
        return (float("nan"), None)
    latest_val = df.iloc[-1]["value"]
    cutoff_date = df.iloc[-1]["date"] - timedelta(days=days)
    past = df[df["date"] <= cutoff_date]
    if past.empty:
        return (latest_val, None)
    past_val = past.iloc[-1]["value"]
    return (latest_val, latest_val - past_val)

# =========================
# SVG geometry
# =========================
def degree_from_score(score: float) -> float:
    """Map 0..100 across the top semicircle (left→right): 0%->180°, 100%->0°."""
    return 180.0 - (score / 100.0) * 180.0

def needle_rotation_deg(score: float) -> float:
    """Needle relative to vertical: 0%->-90°, 50%->0°, 100%->+90°."""
    return (score / 100.0) * 180.0 - 90.0

def _polar(cx: float, cy: float, r: float, deg: float) -> Tuple[float, float]:
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)  # SVG Y axis downwards

def donut_segment_path(cx: float, cy: float, r_outer: float, r_inner: float,
                       start_deg: float, end_deg: float) -> str:
    ox1, oy1 = _polar(cx, cy, r_outer, start_deg)
    ox2, oy2 = _polar(cx, cy, r_outer, end_deg)
    ix2, iy2 = _polar(cx, cy, r_inner, end_deg)
    ix1, iy1 = _polar(cx, cy, r_inner, start_deg)
    delta = abs(end_deg - start_deg)
    large = 1 if delta > 180 else 0
    sweep_outer = 1 if end_deg < start_deg else 0
    sweep_inner = 1 - sweep_outer
    return (
        f"M {ox1:.3f},{oy1:.3f} "
        f"A {r_outer:.3f},{r_outer:.3f} 0 {large} {sweep_outer} {ox2:.3f},{oy2:.3f} "
        f"L {ix2:.3f},{iy2:.3f} "
        f"A {r_inner:.3f},{r_inner:.3f} 0 {large} {sweep_inner} {ix1:.3f},{iy1:.3f} Z"
    )

def arc_path(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> str:
    x1, y1 = _polar(cx, cy, r, start_deg)
    x2, y2 = _polar(cx, cy, r, end_deg)
    delta = abs(end_deg - start_deg)
    large = 1 if delta > 180 else 0
    sweep = 1 if end_deg < start_deg else 0
    return f"M {x1:.3f},{y1:.3f} A {r:.3f},{r:.3f} 0 {large} {sweep} {x2:.3f},{y2:.3f}"

def band_paths(cx: float, cy: float, r_outer: float, r_inner: float) -> List[Dict[str, str]]:
    out = []
    for b in BANDS:
        sd, ed = degree_from_score(b["lo"]), degree_from_score(b["hi"])
        d = donut_segment_path(cx, cy, r_outer, r_inner, sd, ed)
        out.append({"d": d, "color": b["color"], "name": b["name"], "lo": b["lo"], "hi": b["hi"], "sd": sd, "ed": ed})
    return out

def band_for(score: float) -> Dict[str, str]:
    for b in BANDS:
        if b["lo"] <= score < b["hi"]:
            return b
    return BANDS[-1]

# =========================
# Explanations (SDK + REST + fallback)
# =========================
def _fallback_summary(score: float, penalty: float) -> str:
    b = band_for(score)["name"]
    base = {
        "Safe": "Markets look calm. Stay the course. Keep position sizes sensible and avoid big, sudden bets.",
        "Cautious": "Some pressure is building. Stay diversified and trim oversized positions.",
        "Risky": "Risk is rising. Reduce speculative names, add cash/short-duration bonds, consider simple hedges.",
        "Nearly Crash": "Stress is high. Go defensive, prioritize liquidity and quality.",
        "Game Over": "Severe stress. Focus on capital preservation, liquidity, and counterparty safety."
    }[b]
    trend = "" if penalty <= 0.05 else f" Recent 5-day deterioration added a {penalty:.1f}-point penalty."
    return base + trend

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "20"))
DEEPSEEK_RETRIES = int(os.getenv("DEEPSEEK_RETRIES", "2"))
DEEPSEEK_REQUIRE = os.getenv("DEEPSEEK_REQUIRE", "true").lower() in ("1","true","yes","y")

def _sdk_generate_comment(api_key: str, score: float, band_name: str, penalty: float, components: List[Dict[str, object]]) -> str:
    comp_lines = []
    for c in components:
        try:
            comp_lines.append(f"- {c['name']}: score {c['score']:.1f}, latest {c['latest']}, 5D Δ {c['change']}")
        except Exception:
            pass
    comp_blob = "\n".join(comp_lines) if comp_lines else "(no details)"

    system = (
        "You are a market explainer. Speak to a broad audience in crisp, vivid language. "
        "Be accurate and calm. Use 2–3 short sentences. Explain the score’s meaning, key drivers, "
        "and a sensible action bias (diversify, trim risk, raise liquidity) without personal advice."
    )
    user = (
        f"Gauge score: {score:.1f}% ({band_name}). Trend penalty in last 5 days: {penalty:.1f} pts.\n"
        f"Components:\n{comp_blob}\n\n"
        "Write a brief plain-English status: 2–3 sentences, ~70 words max. Avoid jargon."
    )

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=DEEPSEEK_TIMEOUT)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.35,
        max_tokens=180,
        top_p=0.9,
        stream=False,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("DeepSeek SDK returned empty content")
    return " ".join(content.split())

def _rest_generate_comment(api_key: str, score: float, band_name: str, penalty: float, components: List[Dict[str, object]]) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT["User-Agent"],
    }
    comp_lines = []
    for c in components:
        try:
            comp_lines.append(f"- {c['name']}: score {c['score']:.1f}, latest {c['latest']}, 5D Δ {c['change']}")
        except Exception:
            pass
    comp_blob = "\n".join(comp_lines) if comp_lines else "(no details)"

    system = (
        "You are a market explainer. Speak to a broad audience in crisp, vivid language. "
        "Be accurate and calm. Use 2–3 short sentences. Explain the score’s meaning, key drivers, "
        "and a sensible action bias (diversify, trim risk, raise liquidity) without personal advice."
    )
    user = (
        f"Gauge score: {score:.1f}% ({band_name}). Trend penalty in last 5 days: {penalty:.1f} pts.\n"
        f"Components:\n{comp_blob}\n\n"
        "Write a brief plain-English status: 2–3 sentences, ~70 words max. Avoid jargon."
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":user},
        ],
        "temperature": 0.35,
        "max_tokens": 180,
        "top_p": 0.9,
    }
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    last_err = None
    for attempt in range(DEEPSEEK_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=DEEPSEEK_TIMEOUT)
            if resp.status_code >= 400:
                short = resp.text[:400] + ("..." if len(resp.text) > 400 else "")
                raise RuntimeError(f"HTTP {resp.status_code}: {short}")
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if content and isinstance(content, str):
                return " ".join(content.split())
            raise RuntimeError("DeepSeek REST returned empty content")
        except Exception as e:
            last_err = str(e)
            logging.warning("DeepSeek REST error (attempt %d/%d): %s", attempt+1, DEEPSEEK_RETRIES+1, last_err)
            if attempt < DEEPSEEK_RETRIES:
                time.sleep(1.5*(attempt+1))
            else:
                raise
    raise RuntimeError(last_err or "Unknown DeepSeek REST failure")

# =========================
# Explanation rendering (NEW)
# =========================
def explanation_to_html(raw: str) -> str:
    """
    Reflow messy newline-laden text into clean HTML paragraphs:
      - Convert CRLF to LF
      - Collapse spaces/tabs
      - Limit blank lines
      - Split by double newlines into paragraphs
      - Single newlines become spaces within a paragraph
      - HTML-escape content
    """
    s = str(raw or "")
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    if not s:
        return "<p></p>"
    paragraphs = [re.sub(r"\n+", " ", p).strip() for p in s.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]
    return "".join(f"<p>{escape(p)}</p>" for p in paragraphs)

# =========================
# HTML template (UPDATED)
# =========================
def html_template() -> Template:
    return Template(r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Game Over Gauge Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { 
      --needle-deg: {{ needle_deg }}deg;
      --navy: #0f1419;
      --dark-bg: #0a0e12;
      --card-bg: #131820;
      --accent: #00d4ff;
      --success: #00ff88;
      --border: #1a2332;
    }

    * { box-sizing: border-box; }

    body { 
      font-family: 'Inter', 'Helvetica Neue', system-ui, -apple-system, Segoe UI, Roboto, Arial;
      margin: 0;
      padding: 0;
      color: #e8e9eb;
      background: transparent;
    }

    .wrap { 
      max-width: 1180px;
      margin: 0 auto;
      padding: 0;
    }

    .top { 
      display: flex;
      gap: 28px;
      align-items: flex-start;
      flex-wrap: wrap;
    }

    .panel { 
      flex: 1 1 360px;
    }

    .gbox { 
      position: relative;
      width: 560px;
      max-width: 100%;
    }

    .gtitle { 
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 6px;
      color: #ffffff;
    }

    svg { 
      display: block;
      width: 100%;
      height: auto;
    }

    .needle { 
      transform-origin: 280px 240px;
      transform: rotate(var(--needle-deg));
    }

    .hub { 
      fill: #00d4ff;
    }

    .band-stroke { 
      stroke: #1a2332;
      stroke-width: 2;
    }

    .tick text { 
      font-size: 11px;
      fill: #8b95a5;
    }

    .bandlbl { 
      font-size: 12px;
      fill: #e8e9eb;
      opacity: 0.9;
      font-weight: 600;
    }

    .score { 
      font-size: 48px;
      font-weight: 900;
      line-height: 1;
      color: #ffffff;
      background: linear-gradient(135deg, var(--accent) 0%, var(--success) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .badge {
      -webkit-text-fill-color: var(--navy);
      color: var(--navy);
      -webkit-background-clip: initial;
      background-clip: initial;
      display: inline-block;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      margin-left: 12px;
      vertical-align: middle;
    }

    .badge.cautious { background: var(--success); }
    .badge.risky { background: #fbbf24; }
    .badge.crash { background: #ff6b6b; }

    .muted { 
      color: #8b95a5;
      font-size: 12px;
      margin-top: 8px;
    }

    /* UPDATED: clean paragraph rendering (no pre-wrap) */
    .explain { 
      margin-top: 16px;
      font-size: 14px;
      line-height: 1.7;
      color: #a8b5c4;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .explain p { margin: 0 0 10px 0; }
    .explain p:last-child { margin-bottom: 0; }

    .source { 
      margin-top: 12px;
      font-size: 12px;
      color: #6b7a8b;
    }

    .source strong { color: var(--accent); }

    h2 {
      font-size: 24px;
      font-weight: 900;
      margin: 40px 0 24px 0;
      color: #ffffff;
      letter-spacing: -0.5px;
    }

    table { 
      border-collapse: collapse;
      width: 100%;
      margin-top: 20px;
      background: rgba(19, 24, 32, 0.5);
      border: 1px solid var(--border);
      border-radius: 4px;
      overflow: hidden;
    }

    th, td { 
      border: 1px solid var(--border);
      padding: 12px 14px;
      font-size: 14px;
      text-align: left;
    }

    th { 
      background: rgba(0, 212, 255, 0.1);
      text-align: left;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.5px;
    }

    td { color: #a8b5c4; }

    tr:hover { background: rgba(0, 212, 255, 0.05); }

    tr td:first-child {
      font-weight: 600;
      color: #e8e9eb;
    }

    .meta { 
      color: #6b7a8b;
      font-size: 12px;
      margin-top: 16px;
      line-height: 1.6;
    }

    .sr-only { 
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0,0,0,0);
      white-space: nowrap;
      border: 0;
    }

    @media (max-width: 768px) {
      .top { flex-direction: column; }
      .gbox { width: 100%; }
      .score { font-size: 36px; }
      h2 { font-size: 20px; }
      table { font-size: 13px; }
      th, td { padding: 10px 12px; }
    }
  </style>
</head>
<body>
  <div class="wrap">

    <div class="top" role="group" aria-labelledby="gauge-title">
      <div class="gbox">
        <svg viewBox="0 0 560 280" role="img" aria-labelledby="gauge-title">
          <title id="gauge-title">Gauge at {{ total_score|round(1) }} percent</title>

          <defs>
            {% for p in band_paths %}
              <path id="{{ p.label_path_id }}" d="{{ p.label_path_d }}" />
            {% endfor %}
          </defs>

          {% for p in band_paths %}
            <path d="{{ p.d }}" fill="{{ p.color }}" class="band-stroke" />
          {% endfor %}

          {% for p in band_paths %}
            <text class="bandlbl">
              <textPath href="#{{ p.label_path_id }}" startOffset="50%" text-anchor="middle">{{ p.name }}</textPath>
            </text>
          {% endfor %}

          {% for mark in tick_marks %}
            <g class="tick">
              <line x1="{{ mark.x1 }}" y1="{{ mark.y1 }}" x2="{{ mark.x2 }}" y2="{{ mark.y2 }}" stroke="#374151" stroke-width="1"/>
              <text x="{{ mark.tx }}" y="{{ mark.ty }}" text-anchor="middle">{{ mark.label }}</text>
            </g>
          {% endfor %}

          <g class="needle">
            <line x1="280" y1="240" x2="280" y2="60" stroke="#00d4ff" stroke-width="4"/>
            <circle class="hub" cx="280" cy="240" r="8"/>
            <circle cx="280" cy="240" r="13" fill="none" stroke="#00d4ff" stroke-width="2" opacity="0.5"/>
          </g>
        </svg>

        <div class="sr-only" aria-live="polite">Gauge pointer at {{ total_score|round(1) }} percent.</div>
      </div>

      <div class="panel">
        <div class="score">
          {{ total_score|round(1) }}%
          <span class="badge {{ badge_class }}">{{ band_name }}</span>
        </div>
        <div class="muted">As of {{ timestamp }} (UTC). Trend penalty applied: {{ trend_penalty|round(1) }} pts.</div>
        <!-- UPDATED: explanation rendered as sanitized paragraphs -->
        <div class="explain">{{ explanation_html | safe }}</div>
        <div class="source">explanation_source: <strong>{{ explanation_source }}</strong></div>
      </div>
    </div>

    <h2>Risk Components Breakdown</h2>
    <table>
      <thead>
        <tr>
          <th>Indicator</th>
          <th>Latest</th>
          <th>5D Δ</th>
          <th>Score (0–100)</th>
          <th>Weight</th>
        </tr>
      </thead>
      <tbody>
      {% for row in rows %}
        <tr>
          <td>{{ row.name }}</td>
          <td>{{ row.latest }}</td>
          <td>{{ row.change }}</td>
          <td>{{ row.score|round(1) }}</td>
          <td>{{ (row.weight*100)|round(1) }}%</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>

    <p class="meta">
      * Scoring: normalized per indicator anchors (benign → stressed), weighted, then a capped trend penalty is added for sharp 5-day deteriorations.
    </p>
  </div>
</body>
</html>
    """)

# =========================
# Term premium helpers
# =========================
def _read_nyfed_tp_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:120]):
        if "date" in line.lower():
            header_idx = i; break
    if header_idx is None:
        raise RuntimeError("NY Fed TP CSV header not found")
    data = "\n".join(lines[header_idx:])
    for sep in (",",";"):
        try:
            df = pd.read_csv(StringIO(data), sep=sep, engine="python", on_bad_lines="skip", skip_blank_lines=True)
            date_col = next((c for c in df.columns if str(c).lower().strip()=="date"), None)
            if date_col is None: continue
            tp_col = None
            for c in df.columns:
                if str(c).upper().replace(" ","") in ("ACMTP10","ACMTP_10Y","TP10","ACMTP10_"): tp_col = c; break
            if tp_col is None:
                candidates=[c for c in df.columns if c!=date_col]
                best=None; best_ratio=-1.0
                for c in candidates:
                    ratio=pd.to_numeric(df[c], errors="coerce").notna().mean()
                    if ratio>best_ratio: best_ratio, best=ratio, c
                tp_col = best
            out = pd.DataFrame({
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "value": pd.to_numeric(df[tp_col], errors="coerce"),
            }).dropna().sort_values("date").reset_index(drop=True)
            if out.empty: continue
            return out
        except Exception:
            continue
    raise RuntimeError("NY Fed TP CSV parse failed")

def fetch_term_premium_any(fred_key: str) -> Tuple[pd.DataFrame, str]:
    try:
        df = fetch_fred_series(TP_SERIES_FRED, fred_key)
        return df, "ACM 10Y Term Premium (THREEFYTP10, %)"
    except Exception as e:
        logging.info(f"THREEFYTP10 unavailable: {e}")
    try:
        r = requests.get(NYFED_TP_CSV, timeout=25, headers=USER_AGENT, allow_redirects=True)
        r.raise_for_status()
        df_csv = _read_nyfed_tp_csv(r.text)
        return df_csv, "ACM 10Y Term Premium (NY Fed CSV, %)"
    except Exception as e:
        logging.info(f"NY Fed TP CSV unavailable: {e}")
    try:
        df = fetch_fred_series(TP_SERIES_PROXY, fred_key)
        return df, "Curve Proxy (10Y-2Y, %)"
    except Exception as e:
        logging.info(f"T10Y2Y proxy unavailable: {e}")
        raise RuntimeError("All term premium sources failed")

# =========================
# Yields catalog
# =========================
def _apr_from_env(var: str) -> Optional[int]:
    v = getenv_int(var, None)
    if v is None:
        return None
    return max(0, v)

def _safe_get(d: Dict, key: str, default=None):
    return d.get(key) if isinstance(d, dict) else default

def _load_yields_overrides(path: str) -> Optional[Dict]:
    if not path:
        return None
    if not os.path.exists(path):
        logging.warning("YIELDS_CONFIG path does not exist: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except Exception:
            logging.warning("YIELDS_CONFIG is not valid JSON and PyYAML not available.")
            return None
    except Exception as e:
        logging.warning("Failed to read YIELDS_CONFIG: %s", e)
        return None

def _default_yields_catalog(as_of_utc: str) -> Dict:
    def mk(name, product, access, chain, docs, kyc=True,
           min_initial_usd=None, min_instant_usd=None, min_non_instant_mint_usd=None,
           min_subsequent_usd=None, notes=None, apr_bps_env=None):
        apr_bps = _apr_from_env(apr_bps_env) if apr_bps_env else None
        out = {
            "name": name,
            "product": product,
            "access": access,
            "chain": chain if isinstance(chain, list) else [chain],
            "kyc_required": bool(kyc),
            "docs": docs,
        }
        if apr_bps is not None:
            out["apr_bps"] = apr_bps
        if min_initial_usd is not None:
            out["min_initial_usd"] = int(min_initial_usd)
        if min_instant_usd is not None:
            out["min_instant_usd"] = int(min_instant_usd)
        if min_non_instant_mint_usd is not None:
            out["min_non_instant_mint_usd"] = int(min_non_instant_mint_usd)
        if min_subsequent_usd is not None:
            out["min_subsequent_usd"] = int(min_subsequent_usd)
        if notes:
            out["notes"] = notes
        return out

    issuers = [
        mk(
            name="Ondo", product="USDY",
            access="Non-US retail (KYC + wallet whitelist)",
            chain=["Ethereum","Bridges/Solana/L2"],
            docs="https://docs.ondo.finance/general-access-products/usdy/faq/eligibility",
            kyc=True, min_initial_usd=500, apr_bps_env="APR_BPS_ONDO_USDY"
        ),
        mk(
            name="Ondo", product="OUSG",
            access="Qualified/accredited investors",
            chain=["Ethereum","Layer2"],
            docs="https://docs.ondo.finance/qualified-access-products/ousg/instant-limits",
            kyc=True, min_instant_usd=5000, min_non_instant_mint_usd=100000,
            apr_bps_env="APR_BPS_ONDO_OUSG"
        ),
        mk(
            name="OpenEden", product="TBILL",
            access="Global (KYC + wallet whitelist)",
            chain=["Ethereum"],
            docs="https://docs.openeden.com/tbill/faq",
            kyc=True, min_initial_usd=100000, min_subsequent_usd=1000,
            apr_bps_env="APR_BPS_OPENEDEN_TBILL"
        ),
        mk(
            name="Matrixdock", product="STBT",
            access="Accredited/Professional investors",
            chain=["Ethereum"],
            docs="https://forum.arbitrum.foundation/t/matrixdock-stbt-step-application/23584",
            kyc=True, min_initial_usd=100000,
            apr_bps_env="APR_BPS_MATRIXDOCK_STBT"
        ),
        mk(
            name="Franklin Templeton", product="BUIDL/FOBXX",
            access="Institutional/accredited",
            chain=["Public chain"],
            docs="https://www.franklintempleton.com/investments/options/money-market-funds/products/29386/SINGLCLASS/franklin-on-chain-u-s-government-money-fund/FOBXX",
            kyc=True, apr_bps_env="APR_BPS_FRANKLIN_BUIDL"
        ),
        mk(
            name="Maple Finance", product="Cash Management (UST bills)",
            access="DAOs/funds; US accredited",
            chain=["Ethereum"],
            docs="https://maple.finance/insights/maple-cash-management-opens-to-us-investors",
            kyc=True, apr_bps_env="APR_BPS_MAPLE_CASH",
            notes="Targets ~1M UST yield minus fees."
        ),
        mk(
            name="Superstate", product="USTB",
            access="Institutional-leaning",
            chain=["Ethereum"],
            docs="https://superstate.co/",
            kyc=True, apr_bps_env="APR_BPS_SUPERSTATE_USTB"
        ),
    ]
    return {"as_of_utc": as_of_utc, "issuers": issuers}

def _merge_yields(base: Dict, override: Dict) -> Dict:
    if not override or "issuers" not in override:
        return base
    by_key = {(i.get("name"), i.get("product")): i for i in base.get("issuers", [])}
    for item in override.get("issuers", []):
        key = (item.get("name"), item.get("product"))
        if key in by_key:
            by_key[key].update({k: v for k, v in item.items() if v is not None})
        else:
            by_key[key] = item
    merged = []
    seen = set()
    for i in base.get("issuers", []):
        key = (i.get("name"), i.get("product"))
        merged.append(by_key[key])
        seen.add(key)
    for key, val in by_key.items():
        if key not in seen:
            merged.append(val)
    return {"as_of_utc": base.get("as_of_utc"), "issuers": merged}

def build_yields_payload() -> Dict:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    base = _default_yields_catalog(ts)
    cfg_path = os.getenv("YIELDS_CONFIG", "").strip()
    if cfg_path:
        ovr = _load_yields_overrides(cfg_path)
        if ovr:
            try:
                merged = _merge_yields(base, ovr)
                logging.info("Applied YIELDS_CONFIG override: %s (issuers: %d)", cfg_path, len(merged.get("issuers", [])))
                return merged
            except Exception as e:
                logging.warning("Failed to merge YIELDS_CONFIG: %s", e)
    return base

def write_yields_json(path: str) -> Dict:
    payload = build_yields_payload()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info("Wrote yields catalog JSON: %s (issuers=%d)", path, len(payload.get("issuers", [])))
    except Exception as e:
        logging.error("Failed to write yields JSON: %s", e)
        raise
    return payload

# =========================
# Main
# =========================
def main():
    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if not fred_key:
        logging.error("FRED_API_KEY is not set. Create app/.env with your key.")
        sys.exit(1)

    # Optional weight overrides
    weights = dict(DEFAULT_WEIGHTS)
    for envname, key in [
        ("WEIGHT_NOMINAL_10Y", "DGS10"),
        ("WEIGHT_REAL_10Y", "DFII10"),
        ("WEIGHT_HY_OAS", "BAMLH0A0HYM2"),
        ("WEIGHT_VIX", "VIXCLS"),
        ("WEIGHT_TERM_PREMIUM", "TERM_PREMIUM_10Y"),
    ]:
        val = getenv_float(envname, None)
        if val is not None:
            weights[key] = max(0.0, min(1.0, val))

    def renorm(w: Dict[str, float], drop_key: str) -> Dict[str, float]:
        new = dict(w); new.pop(drop_key, None)
        s = sum(new.values())
        if s > 0:
            for k in new: new[k] = new[k] / s
        return new

    latest: Dict[str, float] = {}
    changes: Dict[str, Optional[float]] = {}
    scores: Dict[str, float] = {}
    rows: List[Dict[str, object]] = []

    # FRED indicators
    for sid in FRED_SERIES.keys():
        try:
            df = fetch_fred_series(sid, fred_key)
            lv, ch = latest_and_change(df, TREND_LOOKBACK_DAYS)
            latest[sid], changes[sid] = lv, ch
            scores[sid] = normalize(lv, sid)
        except Exception as e:
            logging.warning(f"Failed to fetch {sid}: {e}")
            latest[sid] = float("nan"); changes[sid] = None; scores[sid] = 0.0

    # Term premium (with fallbacks)
    tp_key = "TERM_PREMIUM_10Y"
    tp_label = "ACM 10Y Term Premium (%)"
    tp_ok = False
    try:
        tp_df, tp_label = fetch_term_premium_any(fred_key)
        lv, ch = latest_and_change(tp_df, TREND_LOOKBACK_DAYS)
        latest[tp_key], changes[tp_key] = lv, ch
        norm_key = "TERM_PREMIUM_PROXY" if "Proxy" in tp_label or "10Y-2Y" in tp_label else "TERM_PREMIUM_10Y"
        scores[tp_key] = normalize(lv, norm_key)
        tp_ok = True
    except Exception as e:
        logging.info(f"All term premium sources failed (degraded mode): {e}")
        latest[tp_key], changes[tp_key] = float("nan"), None
        scores[tp_key] = 0.0
        weights = renorm(weights, tp_key)

    # Weighted score + penalty
    total = sum(weights.get(k, 0.0) * scores.get(k, 0.0) for k in weights.keys())

    penalty = 0.0
    for k in ["DGS10", "DFII10", "BAMLH0A0HYM2", "VIXCLS", tp_key]:
        delta = changes.get(k)
        thr, pts = TREND_PENALTIES.get(k if k != tp_key or tp_ok else "TERM_PREMIUM_PROXY", (None, None))
        if delta is None or thr is None:
            continue
        if delta >= thr:
            penalty += pts
    penalty = min(penalty, MAX_TREND_PENALTY)

    total_score = max(0.0, min(100.0, total + penalty))
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Table rows
    nice_name = {
        "DGS10": "US 10Y Nominal (DGS10, %)",
        "DFII10": "US 10Y Real (DFII10, %)",
        "BAMLH0A0HYM2": "High Yield OAS (%, BofA ICE)",
        "VIXCLS": "VIX (index)",
        "TERM_PREMIUM_10Y": tp_label,
    }
    units_fmt = { "DGS10":"{:.2f}%", "DFII10":"{:.2f}%", "BAMLH0A0HYM2":"{:.2f}%", "VIXCLS":"{:.1f}", "TERM_PREMIUM_10Y":"{:.2f}%" }
    keys = ["DGS10","DFII10","BAMLH0A0HYM2","VIXCLS","TERM_PREMIUM_10Y"]
    for k in keys:
        w = weights.get(k,0.0); lv = latest.get(k,float("nan")); ch = changes.get(k,None)
        rows.append({
            "name": nice_name.get(k,k),
            "latest": ("n/a" if (lv is None or (isinstance(lv,float) and math.isnan(lv))) else units_fmt[k].format(lv)),
            "change": ("n/a" if (ch is None or (isinstance(ch,float) and math.isnan(ch))) else units_fmt[k].format(ch if k!="VIXCLS" else ch)),
            "score": scores.get(k,0.0), "weight": w
        })

    # Geometry for top semicircle
    cx, cy = 280.0, 240.0
    r_outer, r_inner = 230.0, 180.0
    r_center = (r_outer + r_inner) / 2.0
    paths = band_paths(cx, cy, r_outer, r_inner)
    for i, p in enumerate(paths):
        p["label_path_id"] = f"bandlbl_{i}"
        p["label_path_d"] = arc_path(cx, cy, r_center, p["sd"], p["ed"])

    # Ticks
    def tick_at(pct: float):
        deg=degree_from_score(pct)
        x1,y1=_polar(cx,cy,r_inner-4,deg)
        x2,y2=_polar(cx,cy,r_inner-22,deg)
        tx,ty=_polar(cx,cy,r_inner-34,deg)
        return {"x1":round(x1,1),"y1":round(y1,1),"x2":round(x2,1),"y2":round(y2,1),
                "tx":round(tx,1),"ty":round(ty,1),"label":f"{int(pct)}"}
    tick_marks=[tick_at(0), tick_at(50), tick_at(100)]

    # Band/explainer
    band = band_for(total_score)
    band_name = band["name"]; band_color = band["color"]

    # Badge class mapping for dark template
    if band_name in ("Safe", "Cautious"):
        badge_class = "cautious"
    elif band_name == "Risky":
        badge_class = "risky"
    else:
        badge_class = "crash"  # Nearly Crash / Game Over

    ds_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    explanation_source = "local"

    if not ds_key:
        logging.warning("DEEPSEEK_API_KEY not set; using local fallback text.")
        explanation_text = _fallback_summary(total_score, penalty)
    else:
        try:
            if HAS_OPENAI:
                explanation_text = _sdk_generate_comment(ds_key, total_score, band_name, penalty, rows)
                explanation_source = "deepseek-sdk"
            else:
                if DEEPSEEK_REQUIRE:
                    raise RuntimeError("openai SDK not installed but DEEPSEEK_REQUIRE=true; run: pip install openai")
                logging.warning("openai SDK not installed; using REST fallback.")
                explanation_text = _rest_generate_comment(ds_key, total_score, band_name, penalty, rows)
                explanation_source = "deepseek-rest"
        except Exception as e:
            logging.warning("DeepSeek failed: %s", e)
            if DEEPSEEK_REQUIRE:
                logging.error("DEEPSEEK_REQUIRE=true; aborting build because DeepSeek failed.")
                raise
            explanation_text = _fallback_summary(total_score, penalty)
            explanation_source = "local"

    # NEW: pre-render sanitized paragraph HTML
    explanation_html = explanation_to_html(explanation_text)

    out_obj = {
        "timestamp_utc": ts,
        "total_score_percent": round(total_score, 1),
        "trend_penalty": round(penalty, 2),
        "band": band_name,
        "components": rows,
        "explanation": explanation_text,
        "explanation_source": explanation_source,
    }
    print(json.dumps(out_obj, indent=2))

    # Write gauge JSON and HTML
    with open(DEFAULT_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    html = html_template().render(
        total_score=total_score,
        trend_penalty=penalty,
        timestamp=ts, rows=rows,
        needle_deg=needle_rotation_deg(total_score),
        band_paths=paths, tick_marks=tick_marks,
        band_name=band_name, band_color=band_color,
        explanation_html=explanation_html,  # <- updated
        explanation_source=explanation_source,
        badge_class=badge_class,
    )
    with open(DEFAULT_OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    logging.info(
        "Gauge: %.1f%% (%s) | explainer=%s | JSON=%s | HTML=%s",
        total_score, band_name, explanation_source, DEFAULT_OUTPUT_JSON, DEFAULT_OUTPUT_HTML
    )

    # Also write yields catalog JSON every run
    try:
        write_yields_json(DEFAULT_OUTPUT_YIELDS)
    except Exception:
        logging.exception("yields.json generation failed but continuing build.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(2)
