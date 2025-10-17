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

def post_bluesky(gauge, caption, page_url):
    handle = getenv_nonempty("BSKY_HANDLE")
    app_pw = getenv_nonempty("BSKY_APP_PASSWORD")
    if not handle or not app_pw:
        warn(f"Skip Bluesky: BSKY_HANDLE/BSKY_APP_PASSWORD missing "
             f"(handle={'set' if handle else 'missing'}, app_pw={'set' if app_pw else 'missing'}).")
        return
    try:
        log("Bluesky createSession …")
        s = requests.post(
            "https://bsky.social/xrpc/com.atproto.server.createSession",
            json={"identifier": handle, "password": app_pw},
            timeout=20
        )
        if s.status_code >= 400:
            err(f"Bluesky session HTTP {s.status_code}: {_brief_response_text(s)}")
            s.raise_for_status()
        sess = s.json()

        # Bluesky text hard limit ≈ 300 chars; stay conservative (280)
        base = f"Game Over Gauge: {caption} {utm(page_url, 'bluesky')}"
        text = (base[:277] + "…") if len(base) > 280 else base

        body = {
            "repo": sess["did"],
            "collection": "app.bsky.feed.post",
            "record": {
                "$type": "app.bsky.feed.post",
                "text": text,
                "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                "langs": ["en"],  # helps validation & discovery
            },
        }
        log("Bluesky createRecord …")
        r = requests.post(
            "https://bsky.social/xrpc/com.atproto.repo.createRecord",
            headers={"Authorization": f"Bearer {sess['accessJwt']}"},
            json=body,
            timeout=20
        )
        if r.status_code == 400:
            # Log full-ish body to diagnose (e.g., TooLongText, BadToken)
            warn(f"Bluesky 400: {_brief_response_text(r)}")
        if r.status_code >= 400:
            err(f"Bluesky post HTTP {r.status_code}: {_brief_response_text(r)}")
            r.raise_for_status()
        ok("Bluesky posted.")
    except Exception as e:
        err(f"Bluesky error: {e}")

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

def post_reddit(gauge, caption, page_url):
    cid  = getenv_nonempty("REDDIT_CLIENT_ID")
    csec = getenv_nonempty("REDDIT_CLIENT_SECRET")
    user = getenv_nonempty("REDDIT_USERNAME")
    pwd  = getenv_nonempty("REDDIT_PASSWORD")
    sub  = getenv_nonempty("REDDIT_SUBREDDIT") or "investing"
    if not all([cid, csec, user, pwd]):
        warn("Skip Reddit: one or more secrets missing "
             f"(client_id={'set' if cid else 'missing'}, secret={'set' if csec else 'missing'}, "
             f"user={'set' if user else 'missing'}, password={'set' if pwd else 'missing'}).")
        return
    try:
        log("Reddit: fetching access token …")
        tok_r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, csec),  # HTTP Basic with client_id:client_secret
            data={"grant_type": "password", "username": user, "password": pwd},
            headers={"User-Agent": "gog-poster/1.0"},
            timeout=20
        )
        if tok_r.status_code >= 400:
            err(f"Reddit token HTTP {tok_r.status_code}: {_brief_response_text(tok_r)}")
            return  # don’t crash; just skip reddit
        tok_json = tok_r.json()
        access_token = tok_json.get("access_token")
        if not access_token:
            err(f"Reddit token missing access_token: {tok_json}")
            return

        title = f"Game Over Gauge – {gauge['total_score_percent']}% ({gauge['band']})"
        body  = f"{caption}\n\n{utm(page_url, 'reddit')}"

        log(f"Reddit: posting to r/{sub} …")
        r = requests.post(
            "https://oauth.reddit.com/api/submit",
            headers={"Authorization": f"bearer {access_token}", "User-Agent":"gog-poster/1.0"},
            data={"sr": sub, "kind": "self", "title": title, "text": body},
            timeout=20
        )
        if r.status_code >= 400:
            err(f"Reddit post HTTP {r.status_code}: {_brief_response_text(r)}")
            return
        ok(f"Reddit posted to r/{sub}.")
    except Exception as e:
        err(f"Reddit error: {e}")

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
