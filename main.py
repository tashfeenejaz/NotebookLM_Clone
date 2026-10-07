import json, os, time
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import db, rag

LIMIT = 50
db.init()
app = FastAPI(title='Notebook Clone')

@app.exception_handler(HTTPException)
async def _he(_, e): return JSONResponse({'error': str(e.detail)}, status_code=e.status_code)
@app.exception_handler(Exception)
async def _ex(_, e): return JSONResponse({'error': str(e)}, status_code=500)

class NbIn(BaseModel): title: Optional[str] = None
class EnIn(BaseModel): enabled: bool
class QIn(BaseModel): question: str
class FbIn(BaseModel): value: int
class StIn(BaseModel): type: str
class NoteIn(BaseModel):
    content: str = ''
    title: Optional[str] = None
    kind: str = 'note'

def _rm_text(sid):
    p = f'data/text/{sid}.json'
    if os.path.exists(p): os.remove(p)

def _note(n):
    n['meta'] = json.loads(n['meta']) if n.get('meta') else None; return n

@app.get('/api/config')
def config(): return {'chatModel': rag.CHAT_MODEL, 'embedModel': rag.EMB_MODEL, 'limit': LIMIT}

@app.get('/api/notebooks')
def nbs(): return db.rows('select * from notebooks order by id desc')

@app.post('/api/notebooks')
def new_nb(b: NbIn):
    i = db.run('insert into notebooks(title,created) values(?,?)', (b.title or 'Untitled notebook', int(time.time() * 1000)))
    return db.row('select * from notebooks where id=?', (i,))

@app.patch('/api/notebooks/{nb}')
def rename(nb: int, b: NbIn): db.run('update notebooks set title=? where id=?', (b.title, nb)); return {'ok': True}

@app.delete('/api/notebooks/{nb}')
def del_nb(nb: int):
    for s in db.rows('select id from sources where nb=?', (nb,)): _rm_text(s['id'])
    rag.col.delete(where={'nb': nb}); db.run('delete from notebooks where id=?', (nb,)); return {'ok': True}

@app.get('/api/notebooks/{nb}')
def get_nb(nb: int):
    msgs = db.rows('select * from messages where nb=? order by id', (nb,))
    for m in msgs: m['meta'] = json.loads(m['meta']) if m['meta'] else None
    return {'sources': db.rows('select * from sources where nb=? order by id', (nb,)), 'messages': msgs,
            'notes': [_note(n) for n in db.rows('select * from notes where nb=? order by id desc', (nb,))], 'totals': db.totals(nb)}

@app.post('/api/notebooks/{nb}/sources')
async def add_sources(nb: int, files: List[UploadFile] = File(default=[]), url: Optional[str] = Form(None),
                      text: Optional[str] = Form(None), title: Optional[str] = Form(None)):
    jobs = [{'file': (f.filename, await f.read())} for f in files]
    jobs += [{'url': u.strip()} for u in (url or '').splitlines() if u.strip()]
    if text and text.strip(): jobs.append({'text': text, 'title': title})
    n = db.row('select count(*) c from sources where nb=?', (nb,))['c']
    added, errors = [], []
    for j in jobs:
        label = j['file'][0] if 'file' in j else j.get('url') or 'Pasted text'
        if n >= LIMIT: errors.append(f'Source limit reached ({LIMIT} per notebook)'); break
        try: added.append(await run_in_threadpool(rag.add_source, nb, **j)); n += 1
        except Exception as e: errors.append(f'{label}: {getattr(e, "detail", e)}')
    return {'added': added, 'errors': errors, 'totals': db.totals(nb)}

@app.get('/api/sources/{sid}/text')
def src_text(sid: int):
    p = f'data/text/{sid}.json'
    if not os.path.exists(p): raise HTTPException(404, 'The text of this source is not available. Please add it again.')
    return {'pages': json.load(open(p, encoding='utf-8'))}

@app.patch('/api/sources/{sid}')
def toggle(sid: int, b: EnIn): db.run('update sources set enabled=? where id=?', (int(b.enabled), sid)); return {'ok': True}

@app.delete('/api/sources/{sid}')
def del_src(sid: int):
    _rm_text(sid); rag.col.delete(where={'src': sid}); db.run('delete from sources where id=?', (sid,)); return {'ok': True}

@app.post('/api/notebooks/{nb}/chat')
def chat(nb: int, b: QIn):
    rag.ask(nb, b.question); return {'totals': db.totals(nb)}

@app.post('/api/messages/{mid}/feedback')
def feedback(mid: int, b: FbIn): db.run('update messages set feedback=? where id=?', (b.value, mid)); return {'ok': True}

@app.delete('/api/notebooks/{nb}/messages')
def clear(nb: int): db.run('delete from messages where nb=?', (nb,)); return {'ok': True}

@app.post('/api/notebooks/{nb}/studio')
def make_studio(nb: int, b: StIn):
    nid = rag.studio(nb, b.type)
    return {'note': _note(db.row('select * from notes where id=?', (nid,))), 'totals': db.totals(nb)}

@app.post('/api/notebooks/{nb}/notes')
def add_note(nb: int, b: NoteIn):
    i = db.run('insert into notes(nb,content,created,kind,title) values(?,?,?,?,?)', (nb, b.content, int(time.time() * 1000), b.kind, b.title))
    return _note(db.row('select * from notes where id=?', (i,)))

@app.patch('/api/notes/{nid}')
def edit_note(nid: int, b: NoteIn): db.run('update notes set content=?, title=? where id=?', (b.content, b.title, nid)); return {'ok': True}

@app.delete('/api/notes/{nid}')
def del_note(nid: int): db.run('delete from notes where id=?', (nid,)); return {'ok': True}

base = Path(__file__).parent
web = next((d for d in (base / 'static', base) if (d / 'index.html').exists()), base / 'static')
app.mount('/', StaticFiles(directory=web, html=True), name='web')
