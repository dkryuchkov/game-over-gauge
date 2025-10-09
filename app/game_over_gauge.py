#!/usr/bin/env python3
import os, sys, json, math, logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from io import StringIO

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from jinja2 import Template

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
# Term premium sources (in priority order)
TP_SERIES_FRED = "THREEFYTP10"     # ACM 10y term premium (FRED-hosted)
TP_SERIES_PROXY = "T10Y2Y"         # 10y-2y Treasury spread (proxy if TP fails)
NYFED_TP_CSV = "https://www.newyorkfed.org/medialibrary/research/data_indicators/ACMTermPremium_Daily_Levels.csv"

# Default weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    "DGS10": 0.20,         # Nominal 10y
    "DFII10": 0.25,        # Real 10y
    "BAMLH0A0HYM2": 0.30,  # HY OAS
    "VIXCLS": 0.15,        # VIX
    "TERM_PREMIUM_10Y": 0.10,  # Term premium (true or proxy)
}

# Normalization anchors (min->0 score; max->100 score)
NORMALIZERS = {
    "DGS10": (2.5, 7.0),          # 2.5% benign ... 7% severe
    "DFII10": (0.0, 3.0),         # 0% benign ... 3% severe (real yields)
    "BAMLH0A0HYM2": (3.0, 9.0),   # 3% benign ... 9% crisis-like
    "VIXCLS": (12.0, 45.0),       # 12 calm ... 45 stressed
    # For true ACM 10y TP and proxy curve: similar stress mapping
    "TERM_PREMIUM_10Y": (-1.0, 2.0),     # -1% benign … +2% stressed
    "TERM_PREMIUM_PROXY": (-1.0, 2.0),
}

# Trend penalty params
TREND_LOOKBACK_DAYS = 5
TREND_PENALTIES = {
    "DGS10": (0.25, 3.0),          # +25 bps in 5d => +3pts
    "DFII10": (0.25, 4.0),         # +25 bps in 5d => +4pts
    "BAMLH0A0HYM2": (0.50, 5.0),   # +50 bps in 5d => +5pts
    "VIXCLS": (5.0, 4.0),          # +5 vol pts in 5d => +4pts
    "TERM_PREMIUM_10Y": (0.25, 2.0),
    "TERM_PREMIUM_PROXY": (0.25, 2.0),
}
MAX_TREND_PENALTY = 10.0

# Output
DEFAULT_OUTPUT_HTML = os.getenv("OUTPUT_HTML", "dashboard.html")
DEFAULT_OUTPUT_JSON = os.getenv("OUTPUT_JSON", "gauge.json")

USER_AGENT = {"User-Agent": "game-over-gauge/1.4 (+local)"}

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

# --- NY Fed CSV (kept as a quiet secondary fallback) ---
def _read_nyfed_tp_csv(text: str) -> pd.DataFrame:
    # Try to identify header line with 'date'
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:120]):
        if "date" in line.lower():
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("NY Fed TP CSV header not found")
    data = "\n".join(lines[header_idx:])
    # try comma first, then semicolon
    for sep in (",", ";"):
        try:
            df = pd.read_csv(StringIO(data), sep=sep, engine="python", on_bad_lines="skip", skip_blank_lines=True)
            # pick date column
            date_col = None
            for c in df.columns:
                if str(c).lower().strip() == "date":
                    date_col = c
                    break
            if date_col is None:
                continue
            # pick TP column (ACMTP10) or most-numeric
            tp_col = None
            for c in df.columns:
                if str(c).upper().replace(" ", "") in ("ACMTP10", "ACMTP_10Y", "TP10", "ACMTP10_"):
                    tp_col = c
                    break
            if tp_col is None:
                # choose the column with highest numeric density
                candidates = [c for c in df.columns if c != date_col]
                best, best_ratio = None, -1.0
                for c in candidates:
                    ratio = pd.to_numeric(df[c], errors="coerce").notna().mean()
                    if ratio > best_ratio:
                        best_ratio, best = ratio, c
                tp_col = best
            out = pd.DataFrame({
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "value": pd.to_numeric(df[tp_col], errors="coerce"),
            }).dropna().sort_values("date").reset_index(drop=True)
            if out.empty:
                continue
            return out
        except Exception:
            continue
    raise RuntimeError("NY Fed TP CSV parse failed")

def fetch_term_premium_any(fred_key: str) -> Tuple[pd.DataFrame, str]:
    """
    Try FRED THREEFYTP10 (preferred), then NY Fed CSV, then FRED T10Y2Y proxy.
    Returns (df, label) where label is for the UI row.
    """
    # 1) FRED THREEFYTP10
    try:
        df = fetch_fred_series(TP_SERIES_FRED, fred_key)
        return df, "ACM 10Y Term Premium (THREEFYTP10, %)"
    except Exception as e:
        logging.info(f"THREEFYTP10 unavailable: {e}")

    # 2) NY Fed CSV (quiet)
    try:
        r = requests.get(NYFED_TP_CSV, timeout=25, headers=USER_AGENT, allow_redirects=True)
        r.raise_for_status()
        df_csv = _read_nyfed_tp_csv(r.text)
        return df_csv, "ACM 10Y Term Premium (NY Fed CSV, %)"
    except Exception as e:
        logging.info(f"NY Fed TP CSV unavailable: {e}")

    # 3) FRED T10Y2Y proxy
    try:
        df = fetch_fred_series(TP_SERIES_PROXY, fred_key)
        return df.rename(columns={"value": "value"}), "Curve Proxy (10Y-2Y, %)"
    except Exception as e:
        logging.info(f"T10Y2Y proxy unavailable: {e}")
        raise RuntimeError("All term premium sources failed")

def html_template() -> Template:
    return Template(r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Game Over Gauge</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; color: #111; }
    .gauge-wrap { display:flex; align-items:center; gap:24px; flex-wrap:wrap; }
    .gauge {
      width: 220px; height: 110px; position: relative; overflow: hidden; border-top-left-radius: 220px; border-top-right-radius: 220px;
      background: linear-gradient(90deg, #16a34a 0%, #f59e0b 50%, #dc2626 100%);
    }
    .gauge:after {
      content:""; position:absolute; left:50%; bottom:-10px; transform:translateX(-50%);
      width: 200px; height: 200px; background: #fff; border-radius: 50%;
      box-shadow: 0 -2px 6px rgba(0,0,0,0.1) inset;
    }
    .needle {
      position:absolute; left:50%; bottom:0; width:2px; height:110px; background:#111;
      transform-origin: bottom center;
    }
    .score { font-size: 28px; font-weight: 700; }
    table { border-collapse: collapse; width: 100%; margin-top: 16px; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 14px; }
    th { background:#f5f5f5; text-align:left; }
    .meta { color:#555; font-size: 12px; margin-top:8px; }
  </style>
</head>
<body>
  <h1>Game Over Gauge</h1>
  <div class="gauge-wrap">
    <div class="gauge">
      <div class="needle" style="transform: rotate({{ needle_deg }}deg)"></div>
    </div>
    <div>
      <div class="score">{{ total_score|round(1) }}%</div>
      <div class="meta">As of {{ timestamp }} (UTC)</div>
      <div class="meta">Trend penalty applied: {{ trend_penalty|round(1) }} pts</div>
    </div>
  </div>

  <h2>Components</h2>
  <table>
    <thead>
      <tr><th>Indicator</th><th>Latest</th><th>5D Δ</th><th>Score (0–100)</th><th>Weight</th></tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row.name }}</td>
        <td>{{ row.latest }}</td>
        <td>{{ row.change }}</td>
        <td>{{ row.score|round(1) }}</td>
        <td>{{ (row.weight*100)|round(0) }}%</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <p class="meta">
    * Scoring: normalized per indicator anchors (benign → stressed), weighted, then a capped trend penalty is added for sharp 5-day deteriorations.
  </p>
</body>
</html>
    """)

def degree_from_score(score: float) -> float:
    # 0% => -90deg; 100% => +90deg
    return (score / 100.0) * 180.0 - 90.0

def main():
    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if not fred_key:
        logging.error("FRED_API_KEY is not set. Create app/.env with your key.")
        sys.exit(1)

    # Load optional weights
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
        new = dict(w)
        new.pop(drop_key, None)
        s = sum(new.values())
        if s > 0:
            for k in new:
                new[k] = new[k] / s
        return new

    latest: Dict[str, float] = {}
    changes: Dict[str, Optional[float]] = {}
    scores: Dict[str, float] = {}
    rows = []

    # FRED indicators
    for sid in FRED_SERIES.keys():
        try:
            df = fetch_fred_series(sid, fred_key)
            lv, ch = latest_and_change(df, TREND_LOOKBACK_DAYS)
            latest[sid], changes[sid] = lv, ch
            scores[sid] = normalize(lv, sid)
        except Exception as e:
            logging.warning(f"Failed to fetch {sid}: {e}")
            latest[sid] = float("nan")
            changes[sid] = None
            scores[sid] = 0.0

    # Term premium (auto with fallbacks)
    tp_key = "TERM_PREMIUM_10Y"
    tp_label = "ACM 10Y Term Premium (%)"
    tp_ok = False
    try:
        tp_df, tp_label = fetch_term_premium_any(fred_key)
        lv, ch = latest_and_change(tp_df, TREND_LOOKBACK_DAYS)
        latest[tp_key], changes[tp_key] = lv, ch
        # If label is proxy, use proxy normalizer key
        norm_key = "TERM_PREMIUM_PROXY" if "Proxy" in tp_label or "10Y-2Y" in tp_label else "TERM_PREMIUM_10Y"
        scores[tp_key] = normalize(lv, norm_key)
        tp_ok = True
    except Exception as e:
        logging.info(f"All term premium sources failed (degraded mode): {e}")
        latest[tp_key], changes[tp_key] = float("nan"), None
        scores[tp_key] = 0.0
        weights = renorm(weights, tp_key)

    # Weighted score
    total = 0.0
    for k, w in weights.items():
        total += w * scores.get(k, 0.0)

    # Trend penalty
    penalty = 0.0
    for k in ["DGS10", "DFII10", "BAMLH0A0HYM2", "VIXCLS", tp_key]:
        delta = changes.get(k)
        thr_pts = TREND_PENALTIES.get(k if k != tp_key or tp_ok else "TERM_PREMIUM_PROXY", (None, None))
        thr, pts = thr_pts
        if delta is None or thr is None:
            continue
        if delta >= thr:
            penalty += pts
    penalty = min(penalty, MAX_TREND_PENALTY)

    total_score = max(0.0, min(100.0, total + penalty))
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    nice_name = {
        "DGS10": "US 10Y Nominal (DGS10, %)",
        "DFII10": "US 10Y Real (DFII10, %)",
        "BAMLH0A0HYM2": "High Yield OAS (%, BofA ICE)",
        "VIXCLS": "VIX (index)",
        "TERM_PREMIUM_10Y": tp_label,
    }
    units_fmt = {
        "DGS10": "{:.2f}%",
        "DFII10": "{:.2f}%",
        "BAMLH0A0HYM2": "{:.2f}%",
        "VIXCLS": "{:.1f}",
        "TERM_PREMIUM_10Y": "{:.2f}%",
    }

    keys = ["DGS10", "DFII10", "BAMLH0A0HYM2", "VIXCLS", "TERM_PREMIUM_10Y"]
    for k in keys:
        w = weights.get(k, 0.0)
        lv = latest.get(k, float("nan"))
        ch = changes.get(k, None)
        rows.append({
            "name": nice_name.get(k, k),
            "latest": ("n/a" if (lv is None or (isinstance(lv,float) and math.isnan(lv))) else units_fmt[k].format(lv)),
            "change": ("n/a" if (ch is None or (isinstance(ch,float) and math.isnan(ch))) else units_fmt[k].format(ch if k!="VIXCLS" else ch)),
            "score": scores.get(k, 0.0),
            "weight": w
        })

    out_obj = {
        "timestamp_utc": ts,
        "total_score_percent": round(total_score, 1),
        "trend_penalty": round(penalty, 2),
        "components": rows
    }
    print(json.dumps(out_obj, indent=2))

    out_json = os.getenv("OUTPUT_JSON", DEFAULT_OUTPUT_JSON)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    tmpl = html_template()
    html = tmpl.render(
        total_score=total_score,
        trend_penalty=penalty,
        timestamp=ts,
        rows=rows,
        needle_deg=degree_from_score(total_score)
    )
    out_html = os.getenv("OUTPUT_HTML", DEFAULT_OUTPUT_HTML)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    logging.info(f"Gauge: {total_score:.1f}% | JSON: {out_json} | HTML: {out_html}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(2)
