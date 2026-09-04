"""O-Ton Stack — Import-Pipeline: Bundestag-Mediathek-RSS -> MP4 -> Parakeet -> GLM -> pgvector

Usage:
  python import_pipeline.py [--limit N] [--only <title-substring>]
"""
import os, re, sys, json, hashlib, subprocess, tempfile
from datetime import datetime, date, time

import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import init_db, insert_speech
from llm import llm_chat, embed_text

RSS_URL = "https://webtv.bundestag.de/player/macros/bttv/podcast/video/plenar.xml"
ASR_URL = "http://127.0.0.1:5092/v1/audio/transcriptions"
WORK = "/opt/oton/work"


def fetch_rss():
    with urllib.request.urlopen(RSS_URL, timeout=30) as r:
        data = r.read().decode("utf-8", errors="replace")
    items = re.findall(r"<item>(.*?)</item>", data, re.S)
    out = []
    for it in items:
        title = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", it, re.S) or re.search(r"<title>(.*?)</title>", it, re.S)
        guid = re.search(r"<guid>(.*?)</guid>", it, re.S)
        pub = re.search(r"<pubDate>(.*?)</pubDate>", it, re.S)
        if not title or not guid:
            continue
        out.append({"title": title.group(1).strip(), "media": guid.group(1).strip(), "pub": pub.group(1).strip() if pub else ""})
    return out


def parse_title(title):
    # "Redebeitrag von Omid Nouripour (Bundestagsvizepräsident/B90/Grüne) am 10.07.2026 um 17:01 Uhr (25)"
    m = re.match(r"Redebeitrag von (.+?)\s*\(([^)]*)\)\s*am (\d{2}\.\d{2}\.\d{4}) um (\d{2}):(\d{2})", title)
    if not m:
        return None
    name, party, d, hh, mm = m.groups()
    try:
        d = datetime.strptime(d, "%d.%m.%Y").date()
    except ValueError:
        d = date.today()
    return {"speaker": name, "party": party.strip(), "date": d, "time": time(int(hh), int(mm))}


def transcribe_audio(wav):
    """Parakeet ASR-Server (achteronic/parakeet, ONNX-int8) -> Liste {start, end, text}"""
    p = subprocess.run(
        ["curl", "-s", "--max-time", "1800", "-F", f"file=@{wav}",
         "-F", "model=parakeet", "-F", "response_format=json", ASR_URL],
        capture_output=True, text=True, timeout=1800,
    )
    try:
        d = json.loads(p.stdout)
    except Exception:
        print("WARN: ASR-Response unparsebar:", p.stdout[:200])
        return []
    text = (d.get("text") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": 0.0, "text": text}]


def main():
    init_db()
    limit = 6
    only = None
    for i, a in enumerate(sys.argv):
        if a == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])
        if a == "--only" and i + 1 < len(sys.argv):
            only = sys.argv[i + 1]

    items = fetch_rss()
    if only:
        items = [x for x in items if only.lower() in x["title"].lower()]
    items = items[:limit]

    for it in items:
        meta = parse_title(it["title"])
        if not meta:
            print("SKIP (Titel nicht parsebar):", it["title"][:90])
            continue
        # Dedupe: bereits importierte Medien überspringen
        import psycopg
        from db import get_conn
        with get_conn() as c:
            exists = c.execute("SELECT 1 FROM speeches WHERE media_url=%s", (it["media"],)).fetchone()
        if exists:
            print("SKIP (schon importiert):", meta["speaker"], meta["date"])
            continue
        # Existiert schon?
        os.makedirs(WORK, exist_ok=True)
        mp4 = os.path.join(WORK, "rede.mp4")
        try:
            urllib.request.urlretrieve(it["media"], mp4)
        except Exception as e:
            print("DL-Fehler:", it["media"][:60], e)
            continue
        h = hashlib.sha256(open(mp4, "rb").read()).hexdigest()[:16]
        os.rename(mp4, os.path.join(WORK, f"{h}.mp4"))
        mp4 = os.path.join(WORK, f"{h}.mp4")
        wav = os.path.join(WORK, f"{h}.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-ar", "16000", "-ac", "1", wav], check=True)

        print(f"TRANSKRIBIERT: {meta['speaker']} ({meta['party']}) {meta['date']}")
        segs = transcribe_audio(wav)
        full = " ".join(s["text"] for s in segs)
        if not full.strip():
            print("  -> leer, SKIP")
            continue

        summary = llm_chat(
            "Du fasst eine deutsche Bundestagsrede neutral zusammen. "
            "Antworte: 1 Absatz Kernaussagen (bis 6 Zeilen), dann 'Themen: ', dann bis zu 5 Stichworte. "
            "Keine Wertung, keine Haltung.",
            full[:24000],
        )
        vec = embed_text(summary[:8000])
        dur = 0.0
        if segs:
            dur = segs[-1]["end"]
        sp = {
            "speaker": meta["speaker"], "party": meta["party"], "title": it["title"][:300],
            "speech_date": meta["date"], "speech_time": meta["time"], "media_url": it["media"],
            "audio_hash": h, "duration_s": dur, "transcript": full, "summary": summary,
            "embedding": "[" + ",".join(str(x) for x in vec) + "]", "segments": segs,
        }
        sid = insert_speech(sp)
        print(f"  OK id={sid} | {len(segs)} Segmente | Summary {len(summary)} Zeichen")
        os.remove(mp4)
        os.remove(wav)


if __name__ == "__main__":
    main()
