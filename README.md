# NotebookLM Clone

A self-hosted, NotebookLM-style research assistant. Add your own sources, ask questions, and get answers **only from those sources**, with citations you can verify, plus a **token and cost breakdown for every embedding and every answer**.

> Inspired by Google's NotebookLM. This is an independent project and is not affiliated with Google.

## Features

**Sources** (up to 50 per notebook)
- Upload files: PDF, DOCX, PPTX, TXT, MD, CSV, JSON, HTML
- Images (PNG, JPG, WEBP): text is read with the chat model's vision
- Websites and YouTube links (YouTube uses the video transcript)
- Pasted text
- Per-source checkbox, select all, delete, and a card showing chunks, tokens and cost

**Chat**
- Answers grounded only in the selected sources
- Numbered citations. Hover a number for a preview of the supporting sentence, click it to open the source in the left panel with that sentence highlighted
- Citations are verified against the real text. A citation the excerpt does not support is dropped or moved to the excerpt that does
- Save to note, copy, thumbs up/down, and suggested follow-up questions
- Retrieval details for every answer (which chunks were found and their scores)

**Studio**
- Summary, Study guide, Briefing doc, FAQ, Timeline
- Interactive Quiz and flip-style Flashcards
- Data table
- Notes you write yourself (**+ Add note**)

**Cost tracking**
- Every answer shows Embed, Input, Output and Total tokens and cost, plus time and model
- Session cost with question count, and a notebook total that survives deleting sources
- Every source and every Studio output shows its own cost
- Prices live in `pricing.json`, so changing models is a one-line edit

## How it works

```
Upload -> extract text -> chunk (1200 chars, 150 overlap) -> OpenAI embeddings -> ChromaDB
Question -> embed -> top 40 by vector search -> local hybrid rerank (vector + keyword) -> top 10
        -> chat model answers with [n] citations
        -> citation check against the excerpt text -> exact supporting sentences saved for highlighting
```

- **Backend:** FastAPI
- **Metadata, chats, notes, cost ledger:** SQLite
- **Vectors:** ChromaDB (vectors are created through OpenAI so real token usage is available)
- **Frontend:** one static HTML file, no build step
- **Embedding model:** `text-embedding-3-small`
- **Chat model:** set in `pricing.json` (`default_chat_model`) or with `CHAT_MODEL` in `.env`

## Quick start (Docker)

You need Docker with the Compose plugin and an OpenAI API key with credit.

```bash
git clone https://github.com/tashfeenejaz/NotebookLM_Clone.git
cd NotebookLM_Clone
echo "OPENAI_API_KEY=sk-your-key" > .env
docker compose up -d
```

Open **http://localhost:8000**. The first start takes a minute or two while the image builds.

Helper script:

```bash
chmod +x run.sh
./run.sh            # start
./run.sh stop       # stop (data is kept)
./run.sh logs       # live logs
./run.sh status     # running / healthy?
./run.sh update     # rebuild after changing .py files
./run.sh reset      # delete ALL data
```

Your notebooks live in the `notebook-data` Docker volume, so they survive stops, restarts and rebuilds. `docker compose down -v` deletes them.

## Run without Docker

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key" > .env
uvicorn main:app --reload
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `OPENAI_API_KEY` | `.env` | required |
| `CHAT_MODEL` | `.env` | `default_chat_model` in `pricing.json` |
| `TOP_K` (chunks sent to the model) | `.env` | `10` |
| `CANDIDATES` (chunks fetched before reranking) | `.env` | `40` |
| Prices per 1M tokens | `pricing.json` | see file |

To use another chat model, set `CHAT_MODEL` and make sure the exact model name has an entry under `chat` in `pricing.json`. Check current prices on the OpenAI pricing page; the numbers in this repo may be out of date. Which models you can use depends on your OpenAI account.

## Project structure

```
main.py              FastAPI routes
rag.py               retrieval, rerank, citation check, answers, Studio, costs
ingest.py            text extraction for every source type, chunking
db.py                SQLite tables and cost ledger
static/index.html    the whole UI
pricing.json         model prices
Dockerfile, docker-compose.yml
```

## Limitations

- Scanned PDFs have no text layer and are not OCR'd (images you upload as files are read with vision)
- Single user, no login. Do not expose it to the internet as is
- No Audio/Video Overview, Slide Deck, Infographic, Mind Map, web search or Google Drive import
- Citation checking uses word overlap. If the model paraphrases heavily, a correct citation can occasionally be dropped
- Changing the chunking or embedding model means re-adding your sources

## Security

Never commit `.env`; it is already in `.gitignore` and `.dockerignore`. If a key is ever exposed, revoke it in the OpenAI dashboard and create a new one.
