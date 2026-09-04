"""O-Ton Stack — DB-Schicht (PostgreSQL + pgvector + Mistral-Embeddings)"""
import os
import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("OTON_DB_DSN", "postgresql://oton:oton123@localhost:5432/oton_app")

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS speeches (
  id SERIAL PRIMARY KEY,
  speaker TEXT NOT NULL,
  party TEXT DEFAULT '',
  title TEXT,
  speech_date DATE,
  speech_time TIME,
  media_url TEXT,
  audio_hash TEXT,
  duration_s FLOAT,
  transcript TEXT,
  summary TEXT,
  embedding VECTOR(1024),
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS segments (
  id SERIAL PRIMARY KEY,
  speech_id INT REFERENCES speeches(id) ON DELETE CASCADE,
  idx INT,
  start_s FLOAT,
  end_s FLOAT,
  text TEXT
);
CREATE INDEX IF NOT EXISTS ix_speeches_vec ON speeches USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS ix_speeches_date ON speeches (speech_date DESC);
"""


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row)


def init_db():
    with get_conn() as c:
        c.execute(SCHEMA)
        c.commit()


def insert_speech(sp):
    with get_conn() as c:
        r = c.execute(
            """INSERT INTO speeches (speaker, party, title, speech_date, speech_time,
                                     media_url, audio_hash, duration_s, transcript, summary, embedding)
               VALUES (%(speaker)s, %(party)s, %(title)s, %(speech_date)s, %(speech_time)s,
                       %(media_url)s, %(audio_hash)s, %(duration_s)s, %(transcript)s, %(summary)s, %(embedding)s)
               RETURNING id""",
            sp,
        ).fetchone()
        c.execute(
            "DELETE FROM segments WHERE speech_id=%s", (r["id"],)
        )
        for i, s in enumerate(sp.get("segments", [])):
            c.execute(
                "INSERT INTO segments (speech_id, idx, start_s, end_s, text) VALUES (%s,%s,%s,%s,%s)",
                (r["id"], i, s["start"], s["end"], s["text"]),
            )
        c.commit()
        return r["id"]
