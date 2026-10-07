import sqlite3, os, time
os.makedirs('data', exist_ok=True)
DB = 'data/app.db'

def _c():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.execute('PRAGMA foreign_keys=ON'); return c

def run(sql, args=()):
    c = _c()
    try:
        cur = c.execute(sql, args); c.commit(); return cur.lastrowid
    finally: c.close()

def rows(sql, args=()):
    c = _c()
    try: return [dict(r) for r in c.execute(sql, args).fetchall()]
    finally: c.close()

def row(sql, args=()):
    r = rows(sql, args); return r[0] if r else None

def init():
    c = _c()
    c.executescript('''
    create table if not exists notebooks(id integer primary key, title text, created integer);
    create table if not exists sources(id integer primary key, nb integer references notebooks(id) on delete cascade, name text, type text, enabled integer default 1, chars integer, tokens integer, cost real);
    create table if not exists messages(id integer primary key, nb integer references notebooks(id) on delete cascade, role text, content text, meta text, created integer);
    create table if not exists ledger(id integer primary key, nb integer references notebooks(id) on delete cascade, kind text, tokens integer, cost real, created integer);
    create table if not exists notes(id integer primary key, nb integer references notebooks(id) on delete cascade, content text, created integer);
    ''')
    for sql in ('alter table sources add column chunks integer',
                "alter table notes add column kind text default 'note'",
                'alter table notes add column title text',
                'alter table notes add column meta text',
                'alter table messages add column feedback integer default 0'):
        try: c.execute(sql)
        except sqlite3.OperationalError: pass
    c.commit(); c.close()

def ledger(nb, kind, tokens, cost):
    run('insert into ledger(nb,kind,tokens,cost,created) values(?,?,?,?,?)', (nb, kind, tokens, cost, int(time.time() * 1000)))

def totals(nb):
    def s(k):
        r = row('select coalesce(sum(cost),0) c, coalesce(sum(tokens),0) t from ledger where nb=? and kind=?', (nb, k)); return r['c'], r['t']
    ec, et = s('embed'); cc, ct = s('chat')
    return {'embedCost': ec, 'embedTokens': et, 'chatCost': cc, 'chatTokens': ct, 'total': ec + cc}
