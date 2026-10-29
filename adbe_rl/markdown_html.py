"""Minimal Markdown -> standalone HTML for the report (headings, lists, tables, code, images)."""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

CSS = '''
:root{--bg:#edf2f7;--page:#fff;--ink:#172337;--muted:#5b6f87;--accent:#0f4a73;--rule:#d9e2ec;--head:#e8f1f8;--zebra:#f8fafc;--code:#14263b}
@media (prefers-color-scheme:dark){:root{--bg:#10151c;--page:#1a2029;--ink:#e6ebf2;--muted:#9aa8ba;--accent:#8cc4f0;--rule:#2f3a48;--head:#243041;--zebra:#1f2632;--code:#0b1220}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 Arial,sans-serif}
main{max-width:1050px;margin:32px auto;background:var(--page);padding:42px 50px;border-radius:14px}
h1{font-size:31px;line-height:1.3;color:var(--accent)}h2{margin-top:40px;font-size:24px;border-bottom:2px solid var(--rule);padding-bottom:8px;color:var(--accent)}
h3{font-size:19px}p{margin:15px 0}li{margin-bottom:8px}a{color:var(--accent);overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0}th,td{border:1px solid var(--rule);padding:9px 12px;text-align:left}
th{background:var(--head)}tr:nth-child(even){background:var(--zebra)}.table{overflow:auto}
pre{background:var(--code);color:#eef6ff;padding:20px;border-radius:8px;overflow:auto;font-size:13px;line-height:1.6}code{font-size:.9em}
figure{margin:24px 0}img{width:100%;height:auto;background:#fff;border-radius:6px}figcaption{text-align:center;font-size:13px;color:var(--muted)}
@media(max-width:700px){main{padding:22px 16px;margin:0;border-radius:0}h1{font-size:25px}}
@media print{body{background:#fff;color:#000}main{margin:0;padding:0;max-width:none}h2,h3{break-after:avoid}tr,figure{break-inside:avoid}pre{white-space:pre-wrap}table{font-size:11px}}
'''


def inline(text: str) -> str:
    s = html.escape(text)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<![\w*])\*([^*]+)\*(?![\w*])', r'<em>\1</em>', s)
    return re.sub(r'(https://[^\s<]+)', r'<a href="\1">\1</a>', s)


def table_row(cells: list[str], tag: str) -> str:
    return '<tr>' + ''.join(f'<{tag}>{inline(c.strip())}</{tag}>' for c in cells) + '</tr>'


def to_html(markdown: str, root: Path, title: str) -> str:
    lines, out, i = markdown.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith('```'):
            block, i = [], i + 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i])
                i += 1
            out.append('<pre><code>' + html.escape('\n'.join(block)) + '</code></pre>')
            i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = lines[i].strip().strip('|').split('|')
                if not all(re.fullmatch(r'\s*:?-+:?\s*', c) for c in cells):
                    rows.append(cells)
                i += 1
            body = ''.join(table_row(row, 'th' if j == 0 else 'td') for j, row in enumerate(rows))
            out.append(f'<div class="table"><table>{body}</table></div>')
        elif image := re.fullmatch(r'!\[([^]]+)\]\(([^)]+)\)', line):
            alt, path = image.groups()
            data = base64.b64encode((root / path).read_bytes()).decode()
            out.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'
                       f'<figcaption>{html.escape(alt)}</figcaption></figure>')
            i += 1
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            out.append(f'<h{level}>{inline(line[level:].strip())}</h{level}>')
            i += 1
        elif re.match(r'(- |\d+\. )', line):
            tag = 'ul' if line.startswith('- ') else 'ol'
            items = []
            while i < len(lines) and re.match(r'(- |\d+\. )', lines[i]):
                items.append('<li>' + inline(re.sub(r'^(- |\d+\. )', '', lines[i])) + '</li>')
                i += 1
            out.append(f'<{tag}>{"".join(items)}</{tag}>')
        else:
            paragraph = []
            while i < len(lines) and lines[i].strip() and not re.match(r'(#|\||```|- |\d+\. |!\[)', lines[i]):
                paragraph.append(inline(lines[i].rstrip()))
                i += 1
            out.append('<p>' + '<br>'.join(paragraph) + '</p>')
    return (f'<!doctype html><html lang="vi"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{CSS}</style></head><body><main>{"".join(out)}</main></body></html>')
