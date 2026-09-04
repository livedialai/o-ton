# O-Ton Stack: One-Machine-App — Parakeet ASR + Mistral Small 24B LLM + pgvector RAG

> **Idee:** Alles läuft auf **einem** Server (Leaseweb VPS, EU). Audiodaten werden **lokal** transkribiert (NVIDIA Parakeet), die LLM-Aufgaben (Zusammenfassungen, RAG-Antworten) übernimmt ein **offenes Modell** (**Mistral Small 3 / 24B**, Apache 2.0) über die DeepInfra-API — **$0.05 in / $0.08 out je 1M**, flach und billig. Retrieval und Embeddings liegen in **PostgreSQL + pgvector**. Kein S3, kein Cloud-Transkribierdienst: **Local Data Sovereignty** ❤️

---

## 1. Architektur (eine Maschine)

```
┌─────────────────────────── Leaseweb VPS (4 vCPU / 6 GB / 100 GB NVMe) ───────────────────────────┐
│                                                                                                  │
│  [Audio] ──▶ ffmpeg (16 kHz mono) ──▶ Parakeet-TDT-0.6B-v3 (lokal, CPU, GGUF) ──▶ Rohtext        │
│                                            │                                                     │
│                                            ▼                                                     │
│                                   PostgreSQL 16 + pgvector    ◀─── Embeddings (API, billig)      │
│                                     (Reden, Segmente, Vektoren, Cache, Usage)                     │
│                                            ▲                                                     │
│  [Website/Bot] ──▶ FastAPI-Backend ──▶ Mistral Small 24B (DeepInfra) ──▶ Zusammenfassung / RAG   │
│                           └──────── umschreibt Skripte/Tasks (cron)                              │
│                                                                                                  │
│  Backups: 15 GB Infomaniak kDrive (rclone) — keine Cloud-Fremdkosten                             │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **ASR läuft komplett lokal** → Audiomaterial verlässt den Server nie (DSGVO-freundlich).
- **LLM via API** (DeepSeek V4 Flash) → kein GPU-Server nötig, 6 GB RAM reichen.
- **Vektordatenbank = PostgreSQL** → kein separates Vector-DB-System, Wartung = 0.
- **Embeddings**: kleines, billiges Modell über die API (z. B. DeepSeek/Einkaufspreis) oder lokal via `sentence-transformers` (siehe Option B).

---

## 2. Server-Setup (Leaseweb VPS, Ubuntu 24.04, Root)

```bash
# 1) System aktualisieren
apt update && apt upgrade -y
apt install -y python3 python3-venv ffmpeg postgresql postgresql-contrib git curl

# 2) pgvector (der Vektor-Modus in PostgreSQL)
apt install -y postgresql-16-pgvector   # Debian/Ubuntu-Paket
# falls nicht vorhanden: Source-Build siehe https://github.com/pgvector/pgvector

# 3) Datenbank + Rolle anlegen
sudo -u postgres psql -c "CREATE USER oton WITH PASSWORD 'starkes-passwort';"
sudo -u postgres psql -c "CREATE DATABASE oton_app OWNER oton;"
sudo -u postgres psql -d oton_app -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 4) Ram-Swap wenn nötig (6 GB RAM reichen, aber Sicherheitsnetz)
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## 3. ASR-Transkription: NVIDIA Parakeet TDT 0.6B v3 (lokal!)

Das Modell hat **600 Mio. Parameter**, erkennt automatisch 25 europäische Sprachen
(inkl. Deutsch), setzt automatisch **Punkt/Komma/Großschreibung**, liefert
**Segment- und Wort-Zeitstempel** und läuft auf einem 4-vCPU-Server flott auf der CPU.

**Quelle:** <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> (CC-BY-4.0)

```bash
# 1) NeMo-Speech.cpp (offizielle CPU-Engine für Parakeet-GGUF)
git clone --recursive https://github.com/NVIDIA/NeMo-Speech.cpp
cd NeMo-Speech.cpp && bash scripts/setup.sh

# 2) Modell als GGUF (quantisiert, CPU-freundlich) herunterladen — einmalig ca. 700 MB
hf download nvidia/parakeet-tdt-0.6b-v3 \
  parakeet-tdt-0.6b-v3.q8_0.gguf \
  --local-dir /opt/oton/models

# 3) Audio vorbereiten (Whisper/Parakeet-Standard: 16 kHz mono)
ffmpeg -y -v error -i input.mp4 -ar 16000 -ac 1 /tmp/audio.wav

# 4) Transkription (lokal, kostenlos)
nemo-speech transcribe /tmp/audio.wav \
  --model /opt/oton/models/parakeet-tdt-0.6b-v3.q8_0.gguf

# 5) Roh-Audio nach erfolgreicher Transkription löschen (wird NICHT archiviert)
rm /tmp/audio.wav input.mp4
```

**Alternativen:** NeMo (`nemo_toolkit[asr]`) oder Transformers
(`pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")`)
— mehr RAM (ca. 2 GB+, GPU optional), dafür volle Kontrolle.

---

## 4. LLM: Mistral Small 3 / 24B via DeepInfra (offen, billig, flach)

- Modell: `mistralai/Mistral-Small-24B-Instruct-2501` (Apache 2.0, 32k Kontext, multilingual inkl. Deutsch, 24B — konkurriert mit 70B-Modellen, 3× schneller)
- **Preise DeepInfra (flach, keine Peak-Zeiten, Stand 09/2026):**
  | Tier | Input / 1M Tokens | Output / 1M Tokens |
  |---|---|---|
  | **Standard** *(verwendete Basis)* | **$0.05** | **$0.08** |
  | Flex (0,8×, nur für async Batch-Jobs) | $0.04 | $0.064 |
- Gegenüber Qwen3.8 Flash (Global $0.113/$0.382): **Output 4,8× billiger**, Input 2,3× billiger — und das Modell ist offen (Apache 2.0), kein Vendor-Lock-in.
- **Preis Tag und Nacht gleich** — ideal für Cron-Jobs. 6-GB-VPS ist nur der Client.

```python
# app/llm.py — Minimalbeispiel
from openai import OpenAI  # pip install openai

client = OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],          # Key siehe README Abschnitt 7
    base_url="https://api.deepinfra.com/v1/openai",
)

def zusammenfassen(rede_text: str) -> str:
    resp = client.chat.completions.create(
        model="mistralai/Mistral-Small-24B-Instruct-2501",
        messages=[
            {"role": "system", "content": "Du fasst Bundestagsreden neutral und knapp zusammen."},
            {"role": "user", "content": rede_text[:30000]},
        ],
        max_tokens=1200,
    )
    return resp.choices[0].message.content
```

---

## 5. RAG: PostgreSQL + pgvector

```sql
-- Tabellen
CREATE TABLE reden (
  id SERIAL PRIMARY KEY,
  session_date DATE,
  speaker TEXT,
  text TEXT,                  -- Rohtranskript
  summary TEXT,               -- LLM-Zusammenfassung (im Cache, einmalig erzeugt)
  embedding VECTOR(1024),     -- bzw. 768/1536 je nach Embedding-Modell
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON reden USING hnsw (embedding vector_cosine_ops);
```

```python
# app/rag.py
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg://oton:pass@localhost/oton_app")

def suchen(query_embedding, k=5):
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT id, text, 1 - (embedding <=> :v) AS score "
            "FROM reden ORDER BY embedding <=> :v LIMIT :k"
        ), {"v": str(query_embedding), "k": k}).fetchall()
    return rows
```

**Embeddings: Mistral Embed** (`mistral-embed`, 1024-dim, EU, Zero Data Retention)
- **Preis: $0.10 / 1M Tokens** (Batch: $0.05), passt perfekt zu `VECTOR(1024)` oben.
- **Embeddet werden nur die Zusammenfassungen** (`summary`), NICHT die Rohtexte — die Vektorsuche findet dann die passende Zusammenfassung.
- Kosten: 500 Zusammenfassungen/Monat ≈ 300k Tokens (mit 1,5× Chunk-Overhead) → **≈ 3 Cent €/Monat**
- SQL: `INSERT ... embedding :=>'[...]'` — pgvector nimmt die 1024-dim Mistral-Vektoren direkt.

```python
# embeddings über Mistral (API-Key wie bei der Transkription)
from mistralai import Mistral
m = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
resp = m.embeddings.create(model="mistral-embed", inputs=[chunk])
v = resp.data[0].embedding        # 1024 floats
```

**Fallback (0 €, lokal):** `pip install sentence-transformers` + MiniLM-L12 (~120 MB) — nur falls die API mal ausfällt.

**Cache-Prinzip (= Geld sparen):** Zusammenfassungen werden **einmalig** beim Import erzeugt, in `reden.summary` gespeichert und danach nur noch ausgelesen. Kein erneuter LLM-Aufruf pro Website-Besuch. Bei Bedarf per cron neu generieren — der Qwen-Preis ist rund um die Uhr flach.

---

## 6. Kostenrechnung (großzügig, EUR, Stand 09/2026)

### Fixkosten (monatlich/jährlich)
| Position | €/Monat | €/Jahr |
|---|---|---|
| Leaseweb VPS 4 vCPU / 6 GB / 100 GB NVMe / 30 TB | 4,49 | 53,88 |
| Infomaniak Domain + 15 GB Cloud (jährlich) | 0,30 | 3,60 |
| **Fixkosten gesamt** | **4,79** | **57,48** |

### Datenbasis (echte Zahlen)
Das Plenum des Deutschen Bundestags debattiert im langen Durchschnitt **≈ 38–45 h/Monat** (3 Plenartage pro Sitzungswoche, sitzungsfreie Wochen ohne Plenum).
→ Bei ~6 min pro Rede sind das **≈ 400–430 Reden/Monat** — großzügig wird mit **500 Reden/Monat** gerechnet.

### LLM-Kosten (Mistral Small 24B via DeepInfra: $0.05 Input / $0.08 Output je 1M)
**Grundlast — Zusammenfassungen (einmalig erzeugt, im Cache):** Direkte Umrechnung der Sprache in Tokens:
42 h Audio/Monat ≈ 2.520 min × ~130 Wörter/min ≈ **330.000 Wörter ≈ 0,46 M Tokens** (deutsch ≈ 1,4 Tokens/Wort)
+ System-Prompt-Overhead ≈ **0,53 M Tokens Input** + ~0,22 M Output → **≈ 0,04 €/Monat** (4 Cent!)
Selbst mit 3× Sicherheitsfaktor: < 0,15 €/Monat.

**Nutzung — RAG-Suchanfragen (großzügig: 10.000 Besucher-Anfragen/Monat):**
Die **Vektorsuche selbst kostet 0 €** (pgvector läuft auf dem VPS, enthalten in den Fixkosten).
Bezahlt wird nur, was an die LLM-API geht: Kontext + Antwort = 20,0 M Input / 6,0 M Output → **≈ 1,36 €/Monat**

**LLM gesamt: ≈ 1,40 €/Monat = ca. 1,50 $ — unter dem 5-$-Budget** ✓
(Mit Flex-Tier 0,8×: ≈ 1,12 €/Monat)

**Embeddings (Mistral Embed, $0.10/1M) — nur die Zusammenfassungen, nur für den Index:**
Beim Import werden **nur die 500 Zusammenfassungen/Monat embeddet** (~300k Tokens → **≈ 3 Cent €/Monat**).
Die Suche selbst braucht nur ein Mini-Embedding pro Anfrage (~25 Tokens) — praktisch 0 Cent.

### Speicherplan (100 GB NVMe, ~50 h Audio/Monat)
**Roh-Audio wird nach der Transkription gelöscht** — gespeichert wird nur der Text:
| Was | Größe/Monat | 100 GB reichen für |
|---|---|---|
| Transkript-Text (390k Wörter) | ~2,5 MB | unbegrenzt |
| PostgreSQL + pgvector (inkl. 1024-dim Vektoren) | <1 GB/Jahr | unbegrenzt |
| Backup (pg_dump komprimiert) | ~0,5 GB/Jahr (nach Infomaniak 15 GB) | Jahrzehnte |

→ 100 GB NVMe sind damit massiv überdimensioniert — auch das kleinere Modell des Leaseweb-VPS (z. B. „L") reicht locker. Die 30 TB Traffic sind genauso unkritisch (nur Text-API-Verkehr).

### Gesamtkosten
| Szenario | €/Monat | €/Jahr |
|---|---|---|
| **A) Großzügig (alle Reden + 10.000 RAG-Queries)** | **≈ 6,20 €** | **≈ 74,40 €** |
| **B) nur Zusammenfassungen (500 Reden, kein Query-Verkehr)** | **≈ 4,85 €** | **≈ 58,20 €** |
| ASR (Parakeet lokal) | **0,00 €** | **0,00 €** |
| Embeddings (Mistral Embed) | **0,03 €** | **0,36 €** |
| Cloud-Speicher | **0,00 €** (Infomaniak 15 GB für Backups) | **0,00 €** |

### Einmalige Aufbaukosten (Backkatalog → RAG-Index)
Embedding (Mistral, inkl. 1,5× Chunk-Overhead) + rückwirkende Zusammenfassungen (Mistral Small):
| Backkatalog (nur Zusammenfassungen embedden) | Einmalig gesamt | mit Flex-Tier (0,8×) |
|---|---|---|
| 1.000 Reden | **≈ 0,17 €** | 0,09 € |
| 5.000 Reden | **≈ 0,85 €** | 0,43 € |
| 10.000 Reden | **≈ 1,69 €** | 0,85 € |
pgvector-Index, PostgreSQL und Parakeet-Transkription: **0 €** (auf dem VPS).

**Ergebnis:** Die komplette App — lokale Transkription, RAG-Chat, Zusammenfassungen — läuft für
**unter 6,50 €/Monat gesamt (davon LLM ≈ 1,50 $/Monat)**. Mit 100 GB NVMe und 30 TB Traffic ist der
Audio-Rohbestand kein Problem, der Cache-Trick hält die LLM-Kosten klein.

---

## 7. Betrieb & Sicherheit

- **Backups:** nightly `pg_dump` → `/opt/oton/backup/` → rclone zu Infomaniak kDrive (15 GB reichen locker für DB + Configs).
- **Secrets:** `.env` lokal, nie ins Repo. `DEEPINFRA_API_KEY` (Mistral Small 24B), `MISTRAL_API_KEY` (Embeddings/Transkription) + DB-Passwort.
- **TLS:** Caddy oder nginx + Let's Encrypt vor der FastAPI (`:8000`).
- **Monitoring:** `pm2` (fastcab-worker-Trick) oder systemd für die Worker + `journalctl`.
- **DSGVO:** Audiomaterial und Transkripte bleiben auf dem EU-VPS; nur Texte gehen (nach Bedarf) an die LLM-API (DeepInfra/Mistral) — üblicher API-Verarbeitungsfall, kein Training auf deinen Daten.

---

## 8. Roadmap (Umsetzungs-Schritte)

- [ ] Leaseweb VPS + Ubuntu 24.04 provisionieren
- [ ] PostgreSQL 16 + pgvector (Abschnitt 2)
- [ ] Parakeet GGUF + NeMo-Speech.cpp (Abschnitt 3) — Testwav transkribieren
- [ ] Qwen (DashScope)-API-Key + `app/llm.py` (Abschnitt 4)
- [ ] Import-Pipeline: Bundestag-Audio → ffmpeg → Parakeet → PostgreSQL (Abschnitt 5)
- [ ] Embeddings (Option A oder B) + `suchen()` fürs RAG
- [ ] FastAPI-Endpunkte: `/summary` (Cache!), `/rag`, `/transcribe`
- [ ] Cron: Import + Zusammenfassungen (Qwen-Preis ist rund um die Uhr gleich — Zeitplan frei wählbar)
- [ ] Caddy-TLS + Backups (rclone → Infomaniak)
