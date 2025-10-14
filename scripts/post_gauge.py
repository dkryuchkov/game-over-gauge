#!/usr/bin/env python3
import os, json, time
from datetime import datetime, timezone
import requests

SITE_URL = os.getenv("SITE_URL", "https://game-over-gauge.netlify.app").rstrip("/")

def log(msg): print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}")

def get_gauge():
    url = f"{SITE_URL}/gauge.json"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def gen_caption(gauge):
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        # safe fallback
        return f"{gauge['total_score_percent']}% → {gauge['band']}. Diversify, avoid outsized bets, keep liquidity handy."

    comps = "\n".join(
        f"- {c['name']}: score {round(c['score'],1)}, latest {c['latest']}, 5D Δ {c['change']}"
        for c in gauge.get("components", [])
    )
    prompt = (
        f"Gauge score: {gauge['total_score_percent']}% ({gauge['band']}).\n"
        f"Timestamp UTC: {gauge['timestamp_utc']}\n"
        f"Components:\n{comps}\n\n"
        "Write 2–3 short sentences (<=70 words), vivid and calm, explaining what it means and a sensible action bias "
        "(diversify, trim risk, maintain liquidity). No emojis, no disclaimers."
    )
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a market explainer; concise, vivid, non-alarmist."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.35,
                "max_tokens": 180
            },
            timeout=25
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return " ".join(text.split())
    except Exception as e:
        log(f"DeepSeek failed ({e}); using fallback.")
        return f"{gauge['total_score_percent']}% → {gauge['band']}. Diversify, avoid outsized bets, keep liquidity handy."

def utm(url_base, source, medium="social", campaign="daily"):
    from urllib.parse import urlencode
    return f"{url_base}/?{urlencode({'utm_source':source,'utm_medium':medium,'utm_campaign':campaign})}"

# --------- Posters (each safely no-op if secrets missing) ----------

def post_mastodon(caption, page_url):
    inst = os.getenv("MASTODON_INSTANCE","").rstrip("/")
    token = os.getenv("MASTODON_TOKEN","").strip()
    if not inst or not token: return
    try:
        msg = f"Game Over Gauge: {caption} {utm(page_url, 'mastodon')} #markets #macro #risk #GameOverGauge"
        r = requests.post(
            f"{inst}/api/v1/statuses",
            headers={"Authorization": f"Bearer {token}"},
            json={"status": msg, "visibility": "public"},
            timeout=20
        )
        r.raise_for_status()
        log("✅ Mastodon posted.")
    except Exception as e:
        log(f"❌ Mastodon: {e}")

def post_bluesky(caption, page_url):
    handle = os.getenv("BSKY_HANDLE","").strip()
    app_pw = os.getenv("BSKY_APP_PASSWORD","").strip()
    if not handle or not app_pw: return
    try:
        sess = requests.post(
            "https://bsky.social/xrpc/com.atproto.server.createSession",
            json={"identifier": handle, "password": app_pw},
            timeout=20
        ).json()
        msg = f"Game Over Gauge: {caption} {utm(page_url, 'bluesky')}"
        r = requests.post(
            "https://bsky.social/xrpc/com.atproto.repo.createRecord",
            headers={"Authorization": f"Bearer {sess['accessJwt']}"},
            json={
                "repo": sess["did"],
                "collection": "app.bsky.feed.post",
                "record": {
                    "$type": "app.bsky.feed.post",
                    "text": msg,
                    "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                },
            },
            timeout=20
        )
        r.raise_for_status()
        log("✅ Bluesky posted.")
    except Exception as e:
        log(f"❌ Bluesky: {e}")

def post_x(caption, page_url):
    api_key = os.getenv("X_API_KEY","").strip()
    token   = os.getenv("X_API_TOKEN","").strip()
    if not api_key or not token: return
    headers = {
        "Authorization": f"Bearer {token}",
        "X-API-Key": api_key,
        "User-Agent": "gog-poster/1.0",
    }
    try:
        msg = f"Game Over Gauge: {caption} {utm(page_url, 'x')} #markets #macro #risk #GameOverGauge"
        r = requests.post(
            "https://api.twitter.com/2/tweets",
            headers=headers,
            json={"text": msg},
            timeout=20
        )
        # Some tenants return 201; some 403 if posting is restricted. Raise on hard errors.
        if r.status_code not in (200, 201, 202, 403):
            r.raise_for_status()
        if r.status_code == 403:
            log("⚠️  X responded 403 (posting not permitted with this token).")
        else:
            log("✅ X/Twitter posted.")
    except Exception as e:
        log(f"❌ X/Twitter: {e}")

def post_linkedin(caption, page_url):
    token = os.getenv("LI_ACCESS_TOKEN","").strip()
    urn   = os.getenv("LI_PERSON_URN","").strip()
    if not token or not urn: return
    try:
        msg = f"Game Over Gauge: {caption}\n\nRead more: {utm(page_url, 'linkedin')}"
        r = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "author": urn,
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": msg},
                        "shareMediaCategory": "NONE",
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            },
            timeout=20
        )
        # LinkedIn may return 401/403 if token lacks scope or expired.
        r.raise_for_status()
        log("✅ LinkedIn posted.")
    except Exception as e:
        log(f"❌ LinkedIn: {e}")

def post_reddit(caption, page_url):
    cid  = os.getenv("REDDIT_CLIENT_ID","").strip()
    csec = os.getenv("REDDIT_CLIENT_SECRET","").strip()
    user = os.getenv("REDDIT_USERNAME","").strip()
    pwd  = os.getenv("REDDIT_PASSWORD","").strip()
    sub  = os.getenv("REDDIT_SUBREDDIT","").strip() or "investing"
    if not all([cid,csec,user,pwd]): return
    try:
        tok = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, csec),
            data={"grant_type":"password","username":user,"password":pwd},
            headers={"User-Agent":"gog-poster/1.0"},
            timeout=20
        ).json()["access_token"]

        title = f"Game Over Gauge – {gauge['total_score_percent']}% ({gauge['band']})"
        body  = f"{caption}\n\n{utm(page_url, 'reddit')}"

        r = requests.post(
            "https://oauth.reddit.com/api/submit",
            headers={"Authorization": f"bearer {tok}", "User-Agent":"gog-poster/1.0"},
            data={"sr": sub, "kind":"self", "title": title, "text": body},
            timeout=20
        )
        r.raise_for_status()
        log(f"✅ Reddit posted to r/{sub}.")
    except Exception as e:
        log(f"❌ Reddit: {e}")

if __name__ == "__main__":
    log("Fetching gauge…")
    gauge = get_gauge()
    caption = gen_caption(gauge)
    page_url = SITE_URL  # landing page

    log(f"Caption: {caption}")

    post_mastodon(caption, page_url)
    post_bluesky(caption, page_url)
    post_x(caption, page_url)
    post_linkedin(caption, page_url)
    post_reddit(caption, page_url)

    log("Done.")
