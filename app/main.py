"""O-Ton Stack — FastAPI Backend"""
import os, json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import get_conn
from llm import llm_chat, embed_text
from db import init_db

init_db()

BASE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="O-Ton Stack", docs_url="/api/docs")


class Query(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE, "templates", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/speeches")
def speeches(limit: int = 30, page: int = 1, date: str = None, speaker: str = None,
             party: str = None, q: str = None):
    where, args = [], []
    if date:
        where.append("speech_date = %s"); args.append(date)
    if speaker:
        where.append("speaker ILIKE %s"); args.append(f"%{speaker}%")
    if party:
        where.append("party ILIKE %s"); args.append(f"%{party}%")
    if q:
        where.append("(summary ILIKE %s OR transcript ILIKE %s)"); args += [f"%{q}%", f"%{q}%"]
    w = "WHERE " + " AND ".join(where) if where else ""
    offset = (page - 1) * limit
    with get_conn() as c:
        total = c.execute(f"SELECT count(*) FROM speeches {w}", args).fetchone()["count"]
        rows = c.execute(
            f"""SELECT id, speaker, party, title, speech_date, speech_time, summary, created_at
                FROM speeches {w} ORDER BY speech_date DESC, speech_time DESC LIMIT %s OFFSET %s""",
            args + [limit, offset],
        ).fetchall()
    return {"speeches": [dict(r) for r in rows], "total": total, "page": page, "limit": limit}


@app.get("/api/dates")
def dates():
    with get_conn() as c:
        rows = c.execute(
            "SELECT DISTINCT speech_date FROM speeches WHERE speech_date IS NOT NULL "
            "ORDER BY speech_date DESC LIMIT 60"
        ).fetchall()
    return {"dates": [str(r["speech_date"]) for r in rows]}


@app.get("/api/speeches/{sid}")
def speech_detail(sid: int):
    with get_conn() as c:
        row = c.execute("SELECT * FROM speeches WHERE id=%s", (sid,)).fetchone()
        if not row:
            raise HTTPException(404, "not found")
        segs = c.execute(
            "SELECT start_s, end_s, text FROM segments WHERE speech_id=%s ORDER BY start_s",
            (sid,),
        ).fetchall()
    d = dict(row)
    d["segments"] = [dict(s, start=float(s["start_s"]), end=float(s["end_s"])) for s in segs]
    return d


@app.post("/api/search")
def search(q: Query):
    vec = embed_text(q.question)
    with get_conn() as c:
        rows = c.execute(
            """SELECT id, speaker, party, summary, title,
                      1 - (embedding <=> %s::vector) AS score
               FROM speeches WHERE embedding IS NOT NULL
               ORDER BY embedding <=> %s::vector LIMIT 5""",
            (str(vec), str(vec)),
        ).fetchall()
    if not rows:
        return {"answer": "Noch keine Reden indexiert — Import läuft. Kurz später nochmal probieren.", "hits": []}
    hits = [dict(r) for r in rows]
    ctx = "\n\n".join(f"[Redner: {h['speaker']} ({h['party']})]\n{h['summary']}" for h in hits)
    answer = llm_chat(
        "Du beantwortest Fragen zu Bundestagsreden neutral und knapp auf Deutsch. "
        "Nutze nur die untenstehenden Zusammenfassungen. Antworte 3-6 Sätze mit Quellenangabe (Redner).\n\n" + ctx,
        q.question,
    )
    return {"answer": answer, "hits": [{k: h[k] for k in ("id", "speaker", "party", "score")} for h in hits]}


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")
