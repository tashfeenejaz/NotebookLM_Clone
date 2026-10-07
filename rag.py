import os, json, time, base64, re, math
from collections import Counter
import chromadb
from fastapi import HTTPException
import db, ingest

PRICING = json.load(open('pricing.json'))
CHAT_MODEL = os.getenv('CHAT_MODEL', PRICING['default_chat_model'])
EMB_MODEL = PRICING['embedding']['model']
TOP_K = int(os.getenv('TOP_K', 10))
CAND = int(os.getenv('CANDIDATES', 40))
os.makedirs('data/text', exist_ok=True)
# We create the vectors ourselves via OpenAI (to get real token usage); Chroma only stores and searches them.
col = chromadb.PersistentClient(path='data/chroma').get_or_create_collection(
    'chunks_v2', metadata={'hnsw:space': 'cosine'}, embedding_function=None)

SYSTEM = ("You are a research assistant. Answer ONLY using the numbered source excerpts provided. "
          "Put a citation right after each claim, before the full stop, like [2] or [2][5]. Cite ONLY the excerpt that directly "
          "states that claim; never cite an excerpt that merely mentions the topic. If the excerpts do not contain the answer, "
          "say so plainly and do not guess. Be clear and well organised; use short paragraphs or bullet points. "
          "Reply in English. After your answer, add a final line that starts with 'Follow-ups:' followed by three short "
          "follow-up questions the sources could answer, separated by ' || '.")

STOP = set('the and for with from that this these those which who whom what when where how not than then such can may might '
           'should would could will shall has have had also into over under between among about any all each other their there '
           'its they them his her our your you are was were been being per via within without including used using use based '
           'one two more most some any is of to in on at as by be it we or an if no so do'.split())

def toks(s):
    out = []
    for w in re.findall(r'[a-z0-9]+(?:[.\-][0-9]+)?', s.lower()):
        if w in STOP or len(w) < 2: continue
        if len(w) > 3 and w.endswith('s') and not w.endswith('ss'): w = w[:-1]
        out.append(w)
    return out

def make_idf(sets):
    n = len(sets); df = Counter(w for s in sets for w in s)
    return lambda w: math.log((n + 1) / (df.get(w, 0) + 0.5))

def wrecall(claim, have, idf):
    tot = sum(idf(w) for w in claim)
    return sum(idf(w) for w in claim if w in have) / tot if tot > 0 else 0.0

def sent_spans(text):
    spans, i = [], 0
    for m in re.finditer(r'(?<=[.!?])\s+|\n+', text):
        if m.start() > i: spans.append((i, m.start()))
        i = m.end()
    if i < len(text): spans.append((i, len(text)))
    return spans

def best_span(claim, text, idf, cache):
    """Best window of up to 3 sentences inside `text` for this claim -> (score, start, end)."""
    if text not in cache: cache[text] = [(a, b, set(toks(text[a:b]))) for a, b in sent_spans(text)]
    ss, best = cache[text], (0.0, 0, 0)
    for a in range(len(ss)):
        have = set()
        for b in range(a, min(a + 3, len(ss))):
            have |= ss[b][2]; s, e = ss[a][0], ss[b][1]
            if e - s > 700: break
            sc = wrecall(claim, have, idf)
            if sc > best[0] + 1e-9 or (abs(sc - best[0]) <= 1e-9 and sc > 0 and e - s < best[2] - best[1]): best = (sc, s, e)
    return best

def attribute(answer, cites):
    """Verify each [n] against the real excerpt text. Unsupported citations are dropped or moved to the excerpt that
    actually states the claim, and the exact supporting sentences are recorded for highlighting."""
    answer = re.sub(r'([.!?])((?:\s*\[\d+\])+)', r'\2\1', answer)
    idf, cache = make_idf([set(toks(c['text'])) for c in cites]), {}
    spans, ev, out = {c['n']: [] for c in cites}, {c['n']: [] for c in cites}, []
    for line in answer.split('\n'):
        new = []
        for sent in re.split(r'(?<=[.!?])\s+', line):
            marks = [int(x) for x in re.findall(r'\[(\d+)\]', sent)]
            text = re.sub(r'\s*\[\d+\]', '', sent).strip()
            claim = set(toks(text))
            if not marks or len(claim) < 2: new.append(sent); continue
            sc = {c['n']: best_span(claim, c['text'], idf, cache) for c in cites}
            keep = [k for k in dict.fromkeys(marks) if k in sc and sc[k][0] >= 0.35]
            top = max(sc, key=lambda k: sc[k][0])
            if top not in keep and ((not keep and sc[top][0] >= 0.45) or (keep and sc[top][0] >= 0.5 and sc[top][0] >= max(sc[k][0] for k in keep) + 0.2)):
                keep.append(top)
            for k in keep:
                s, e = sc[k][1], sc[k][2]
                spans[k].append([cites[k - 1]['start'] + s, e - s]); ev[k].append(cites[k - 1]['text'][s:e])
            tail = re.search(r'[.!?]\s*$', text)
            new.append((text[:tail.start()] if tail else text) + ''.join(f'[{k}]' for k in sorted(keep)) + (text[tail.start():] if tail else ''))
        out.append(' '.join(new))
    for c in cites:
        c['spans'] = [list(x) for x in dict.fromkeys(map(tuple, spans[c['n']]))]; c['evidence'] = '\n…\n'.join(dict.fromkeys(ev[c['n']]))
    return '\n'.join(out)

_client = None
def client():
    global _client
    if not _client:
        from openai import OpenAI
        if not os.getenv('OPENAI_API_KEY'): raise HTTPException(500, 'OPENAI_API_KEY is missing in your .env file')
        _client = OpenAI()
    return _client

def usd(tin, tout):
    p = PRICING['chat'].get(CHAT_MODEL)
    if not p: raise HTTPException(500, f"Add a price for '{CHAT_MODEL}' under 'chat' in pricing.json")
    return tin * p['input_per_1M'] / 1e6, tout * p['output_per_1M'] / 1e6

def embed(texts):
    vecs, tokens = [], 0
    for i in range(0, len(texts), 96):
        r = client().embeddings.create(model=EMB_MODEL, input=texts[i:i + 96])
        vecs += [d.embedding for d in r.data]; tokens += r.usage.total_tokens
    return vecs, tokens

def _ocr(acc):
    def f(data, mime):
        b64 = base64.b64encode(data).decode()
        r = client().chat.completions.create(model=CHAT_MODEL, messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Transcribe all text in this image exactly as written. If it has little or no text, describe it in detail. Output plain text only.'},
            {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}}]}])
        acc['in'] += r.usage.prompt_tokens; acc['out'] += r.usage.completion_tokens
        return r.choices[0].message.content or ''
    return f

def add_source(nb, **inp):
    acc = {'in': 0, 'out': 0}
    name, typ, pages = ingest.extract(ocr=_ocr(acc), **inp)
    chunks, norm = ingest.chunk(pages)
    if not chunks: raise ValueError('No readable text found in this source (scanned PDFs need OCR).')
    vecs, tokens = embed([c[1] for c in chunks])
    ic, oc = usd(acc['in'], acc['out']) if (acc['in'] or acc['out']) else (0.0, 0.0)
    cost = tokens * PRICING['embedding']['per_1M'] / 1e6 + ic + oc
    tok = tokens + acc['in'] + acc['out']
    sid = db.run('insert into sources(nb,name,type,chars,tokens,cost,chunks) values(?,?,?,?,?,?,?)',
                 (nb, name, typ, sum(len(c[1]) for c in chunks), tok, cost, len(chunks)))
    json.dump(norm, open(f'data/text/{sid}.json', 'w', encoding='utf-8'))
    for i in range(0, len(chunks), 200):
        part = chunks[i:i + 200]
        col.add(ids=[f'{sid}-{i + j}' for j in range(len(part))], documents=[c[1] for c in part], embeddings=vecs[i:i + 200],
                metadatas=[{'src': sid, 'nb': nb, 'page': c[0] or 0, 'start': c[2], 'len': len(c[1]), 'name': name} for c in part])
    db.ledger(nb, 'embed', tok, cost)
    return db.row('select * from sources where id=?', (sid,))

def split_followups(t):
    m = re.search(r'\n*\s*Follow-ups?:\s*(.+)$', t, re.S | re.I)
    if not m: return t.strip(), []
    qs = [re.sub(r'^\d+[.)]\s*', '', q).strip(' -•*"') for q in re.split(r'\s*\|\|\s*|\n+', m.group(1))]
    return t[:m.start()].strip(), [q for q in qs if q][:3]

def ask(nb, question):
    t0 = time.time(); usd(0, 0)
    ids = [r['id'] for r in db.rows('select id from sources where nb=? and enabled=1', (nb,))]
    if not ids: raise HTTPException(400, 'Add and select at least one source first')
    qv, eq = embed([question])
    res = col.query(query_embeddings=qv, n_results=CAND, where={'$and': [{'nb': nb}, {'src': {'$in': ids}}]})
    docs, metas, dists = res['documents'][0], res['metadatas'][0], res['distances'][0]
    if not docs: raise HTTPException(400, 'No indexed text found for the selected sources. Please add them again.')
    # local hybrid rerank (free): vector similarity + keyword match weighted by rarity, over the top candidates
    sets = [set(toks(d)) for d in docs]; idf = make_idf(sets); qt = set(toks(question))
    sims = [1 - x for x in dists]; lo, hi = min(sims), max(sims)
    final = [0.55 * (s - lo) / (hi - lo + 1e-9) + 0.45 * wrecall(qt, st, idf) for s, st in zip(sims, sets)]
    order = sorted(range(len(docs)), key=lambda i: -final[i])[:TOP_K]
    citations = [{'n': j + 1, 'source': metas[i]['name'], 'src': metas[i]['src'], 'page': metas[i]['page'] or None,
                  'start': metas[i]['start'], 'len': metas[i]['len'], 'text': docs[i], 'score': round(final[i], 3)} for j, i in enumerate(order)]
    context = '\n\n---\n\n'.join(f"[{c['n']}] ({c['source']}{', p.' + str(c['page']) if c['page'] else ''})\n{c['text']}" for c in citations)
    history = list(reversed(db.rows('select role,content from messages where nb=? order by id desc limit 6', (nb,))))
    r = client().chat.completions.create(model=CHAT_MODEL, messages=[
        {'role': 'system', 'content': SYSTEM}, *history, {'role': 'user', 'content': f'Sources:\n\n{context}\n\nQuestion: {question}'}])
    answer, followups = split_followups(r.choices[0].message.content or 'No answer returned.')
    answer = attribute(answer, citations)
    tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
    ec = eq * PRICING['embedding']['per_1M'] / 1e6
    ic, oc = usd(tin, tout)
    cost = {'embedTokens': eq, 'inTokens': tin, 'outTokens': tout, 'embedCost': ec, 'inCost': ic, 'outCost': oc,
            'chatCost': ic + oc, 'total': ec + ic + oc, 'seconds': round(time.time() - t0, 1), 'candidates': len(docs), 'top': len(order)}
    now = int(time.time() * 1000)
    db.run('insert into messages(nb,role,content,created) values(?,?,?,?)', (nb, 'user', question, now))
    db.run('insert into messages(nb,role,content,meta,created) values(?,?,?,?,?)',
           (nb, 'assistant', answer, json.dumps({'citations': citations, 'cost': cost, 'model': CHAT_MODEL, 'followups': followups}), now))
    db.ledger(nb, 'chat', eq + tin + tout, cost['total'])

# ---------------- Studio ----------------
STUDIO = {
    'summary': ('Summary', False, 'Write a clear summary of the sources: a short overview paragraph, then the key points as bullet points. Use Markdown.'),
    'study_guide': ('Study guide', False, 'Create a study guide: key topics, 8 short-answer questions, an answer key, and a glossary of key terms. Use Markdown headings.'),
    'briefing': ('Briefing doc', False, 'Create a briefing document: title, executive summary, main themes, key facts and figures, and conclusions. Use Markdown headings.'),
    'faq': ('FAQ', False, 'Write 8-10 frequently asked questions with concise answers drawn from the sources. Use Markdown with the questions in bold.'),
    'timeline': ('Timeline', False, 'Create a chronological timeline of the events, dates and developments in the sources, one bullet per entry with the date in bold. If the sources contain no dates or sequence, say so.'),
    'quiz': ('Quiz', True, 'Create an 8-question multiple-choice quiz with 4 options each. Return JSON only: {"questions":[{"question":"...","options":["...","...","...","..."],"answer_index":0,"explanation":"..."}]}'),
    'flashcards': ('Flashcards', True, 'Create 12 flashcards of key concepts. Return JSON only: {"cards":[{"front":"...","back":"..."}]}'),
    'data_table': ('Data table', True, 'Extract the most important structured information into a table. Return JSON only: {"columns":["..."],"rows":[["..."]]} with 3-6 columns and up to 15 rows.'),
}

def gather(nb, budget=60000):
    ids = [r['id'] for r in db.rows('select id from sources where nb=? and enabled=1', (nb,))]
    if not ids: raise HTTPException(400, 'Add and select at least one source first')
    res = col.get(where={'$and': [{'nb': nb}, {'src': {'$in': ids}}]}, include=['documents', 'metadatas'])
    items = sorted(zip(res['documents'], res['metadatas']), key=lambda x: (x[1]['src'], x[1]['page'], x[1]['start']))
    total = sum(len(d) for d, _ in items)
    if total > budget: items = items[::-(-total // budget)]  # evenly sample long notebooks
    return '\n\n'.join(f"[{m['name']}]\n{d}" for d, m in items)

def studio(nb, kind):
    t0 = time.time()
    if kind not in STUDIO: raise HTTPException(400, 'Unknown Studio type')
    label, is_json, instr = STUDIO[kind]
    usd(0, 0); ctx = gather(nb)
    r = client().chat.completions.create(model=CHAT_MODEL, messages=[
        {'role': 'system', 'content': 'You create study materials strictly from the provided source excerpts. Do not use outside knowledge. Write in English unless the sources are in another language.'},
        {'role': 'user', 'content': f'Sources:\n\n{ctx}\n\nTask: {instr}'}],
        **({'response_format': {'type': 'json_object'}} if is_json else {}))
    text = r.choices[0].message.content or ''
    if is_json:
        m = re.search(r'\{.*\}', text, re.S)
        try: json.loads(m.group(0)); text = m.group(0)
        except Exception: raise HTTPException(500, 'The model returned an invalid format. Please try again.')
    tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
    ic, oc = usd(tin, tout)
    cost = {'inTokens': tin, 'outTokens': tout, 'inCost': ic, 'outCost': oc, 'total': ic + oc, 'seconds': round(time.time() - t0, 1), 'model': CHAT_MODEL}
    nid = db.run('insert into notes(nb,content,created,kind,title,meta) values(?,?,?,?,?,?)',
                 (nb, text, int(time.time() * 1000), kind, label, json.dumps(cost)))
    db.ledger(nb, 'chat', tin + tout, cost['total'])
    return nid
