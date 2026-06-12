/* ============================================================
   AI SCHOOL · lesson reader
   Resolves ?path=phases/<phase>/<lesson> (or ?p=phaseId:idx) to a
   lesson, fetches its markdown from the curriculum repo, renders it
   in the 8-bit reader, and wires mark-complete + prev/next. The
   markdown→HTML pass is a compact line-based renderer (headings,
   lists, tables, code fences, blockquotes, inline marks).
   ============================================================ */
(function () {
  'use strict';
  const { store } = window.AIS;
  const $ = (s) => document.querySelector(s);


  /* ---------- flatten lessons + derive paths ---------- */
  const flat = [];
  PHASES.forEach((p, pi) => p.lessons.forEach((l, li) => {
    let path = '';
    if (l.url) { const m = l.url.match(/(phases\/[^/?#]+\/[^/?#]+)/); if (m) path = m[1]; }
    flat.push({ pi, li, phaseId: p.id, phaseName: p.name, name: l.name, type: l.type, lang: l.lang, url: l.url || '', status: l.status, path });
  }));

  /* ---------- resolve the requested lesson ---------- */
  const params = new URLSearchParams(location.search);
  const qPath = params.get('path');
  const qP = params.get('p'); // phaseId:idx fallback
  let cur = null;
  if (qPath) cur = flat.find((f) => f.path === qPath) || flat.find((f) => f.url.indexOf(qPath) !== -1);
  if (!cur && qP) { const [pid, idx] = qP.split(':').map(Number); cur = flat.find((f) => f.phaseId === pid && f.li === idx); }
  if (!cur && (qPath || qP)) cur = null;

  if (!cur) { showError('No lesson selected', 'Open a lesson from the Course or Catalog, or press ⌘K to search.'); return; }

  const flatIdx = flat.indexOf(cur);

  buildSidebar();
  initSidebarToggle();
  initScrollProgress();
  loadLesson(cur);

  /* ---------- sidebar (current phase) ---------- */
  function buildSidebar() {
    const phase = PHASES[cur.pi];
    const host = $('#sidebar');
    const frag = document.createDocumentFragment();

    const hd = document.createElement('div');
    hd.className = 'lz-sb-phase';
    hd.textContent = 'Phase ' + String(phase.id).padStart(2, '0') + ' · ' + phase.name;
    frag.appendChild(hd);

    phase.lessons.forEach((l, li) => {
      const f = flat.find((x) => x.pi === cur.pi && x.li === li);
      const a = document.createElement('a');
      a.className = 'lz-sb-link' + (li === cur.li ? ' is-active' : '');
      if (f.path) a.href = 'lesson.html?path=' + encodeURIComponent(f.path);
      else if (l.url) a.href = l.url;
      const dot = document.createElement('span');
      dot.className = 'lz-sb-dot' + (store.isDone(phase.id, li) ? ' is-done' : '');
      a.appendChild(dot);
      a.appendChild(document.createTextNode(l.name));
      frag.appendChild(a);
    });

    const nav = document.createElement('div');
    nav.className = 'lz-sb-nav';
    if (cur.pi > 0) nav.appendChild(phaseLink('← ' + PHASES[cur.pi - 1].name, cur.pi - 1));
    if (cur.pi < PHASES.length - 1) nav.appendChild(phaseLink(PHASES[cur.pi + 1].name + ' →', cur.pi + 1));
    const all = document.createElement('a'); all.href = 'course.html'; all.textContent = 'All phases'; nav.appendChild(all);
    frag.appendChild(nav);

    host.replaceChildren(frag);
  }
  function phaseLink(label, pi) {
    const first = flat.find((x) => x.pi === pi && x.path);
    const a = document.createElement('a');
    a.href = first ? 'lesson.html?path=' + encodeURIComponent(first.path) : 'course.html';
    a.textContent = label;
    return a;
  }

  /* ---------- fetch + render ---------- */
  async function loadLesson(lesson) {
    if (!lesson.url) { renderArticle(lesson, '# ' + lesson.name + '\n\nThis lesson has no published material yet.'); return; }
    let base = lesson.url
      .replace('://github.com/', '://raw.githubusercontent.com/')
      .replace('/tree/main/', '/main/').replace('/blob/main/', '/main/');
    if (!/\/$/.test(base)) base += '/';
    const candidates = ['README.md', 'readme.md', 'docs/en.md', 'index.md'];
    for (const c of candidates) {
      try {
        const res = await fetch(base + c);
        if (res.ok) { const md = await res.text(); renderArticle(lesson, md); return; }
      } catch (_) { /* try next */ }
    }
    // graceful fallback: link out to GitHub
    showError(lesson.name, 'The lesson text could not be loaded here.', lesson.url);
  }

  function renderArticle(lesson, md) {
    const art = $('#article');
    const phase = PHASES[lesson.pi];
    const done = store.isDone(phase.id, lesson.li);

    const head = document.createElement('div');
    head.className = 'lz-eyebrow';
    head.textContent = 'Phase ' + String(phase.id).padStart(2, '0') + ' · ' + phase.name + ' · Lesson ' + (lesson.li + 1) + ' of ' + phase.lessons.length;

    const meta = document.createElement('div');
    meta.className = 'lz-meta';
    if (lesson.type) { const t = document.createElement('span'); t.className = 'lz-tag'; t.setAttribute('data-t', lesson.type); t.textContent = lesson.type; meta.appendChild(t); }
    if (lesson.lang && lesson.lang !== '—') { const lg = document.createElement('span'); lg.className = 'lz-tag'; lg.textContent = lesson.lang; meta.appendChild(lg); }
    if (lesson.url) { const gh = document.createElement('a'); gh.className = 'lz-tag'; gh.href = lesson.url; gh.target = '_blank'; gh.rel = 'noopener'; gh.textContent = 'GitHub ↗'; meta.appendChild(gh); }
    if (lesson.path) {
      const cp = document.createElement('button');
      cp.className = 'lz-tag lz-copy-path';
      cp.title = 'Copy path for Claude Code';
      cp.textContent = '$ copy path';
      cp.addEventListener('click', () => {
        const prompt = `I'm working on this lesson: ${lesson.path}\nRead ${lesson.path}/docs/en.md and help me work through it.`;
        navigator.clipboard.writeText(prompt).then(() => {
          cp.textContent = '✓ copied';
          setTimeout(() => { cp.textContent = '$ copy path'; }, 1500);
        });
      });
      meta.appendChild(cp);
    }
    const btn = document.createElement('button');
    btn.className = 'lz-done-btn' + (done ? ' is-done' : '');
    btn.textContent = done ? '✓ Completed' : 'Mark complete';
    btn.addEventListener('click', () => {
      const nowDone = store.toggle(phase.id, lesson.li);
      btn.classList.toggle('is-done', nowDone);
      btn.textContent = nowDone ? '✓ Completed' : 'Mark complete';
      const dot = $('#sidebar .lz-sb-link.is-active .lz-sb-dot');
      if (dot) dot.classList.toggle('is-done', nowDone);
    });
    meta.appendChild(btn);

    const body = document.createElement('div');
    body.className = 'lz-body';
    body.innerHTML = mdToHtml(md);

    art.replaceChildren(head, meta, body, buildNav(lesson));
    window.scrollTo(0, 0);
  }

  function buildNav(lesson) {
    const nav = document.createElement('div');
    nav.className = 'lz-nav';
    const prev = flat[flatIdx - 1], next = flat[flatIdx + 1];
    if (prev) nav.appendChild(navBtn('prev', 'Previous', prev));
    if (next) nav.appendChild(navBtn('next', 'Next', next));
    return nav;
  }
  function navBtn(cls, label, f) {
    const a = document.createElement('a');
    a.className = cls;
    a.href = f.path ? 'lesson.html?path=' + encodeURIComponent(f.path) : (f.url || 'course.html');
    const l = document.createElement('span'); l.className = 'lz-nav-l'; l.textContent = label;
    const t = document.createElement('span'); t.className = 'lz-nav-t'; t.textContent = f.name;
    a.appendChild(l); a.appendChild(t);
    return a;
  }

  /* ---------- markdown → HTML (compact) ---------- */
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  function inline(text) {
    const codes = [];
    text = text.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
    text = esc(text);
    text = text.replace(/!\[([^\]]*)\]\(([^)\s]+)[^)]*\)/g, '<img src="$2" alt="$1">');
    text = text.replace(/\[([^\]]+)\]\(([^)\s]+)[^)]*\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/__([^_]+)__/g, '<strong>$1</strong>');
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    text = text.replace(/(^|[^A-Za-z0-9])_([^_]+)_/g, '$1<em>$2</em>');
    text = text.replace(/\u0000(\d+)\u0000/g, (_, i) => '<code>' + esc(codes[i]) + '</code>');
    return text;
  }
  function splitRow(line) { return line.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim()); }

  function mdToHtml(md) {
    md = md.replace(/\r\n?/g, '\n').replace(/^---\n[\s\S]*?\n---\n/, ''); // strip YAML front-matter
    const lines = md.split('\n'); const out = []; let i = 0;
    const isBlockStart = (s) => /^(#{1,6}\s|```|>\s?|\s*[-*+]\s+|\s*\d+\.\s+)/.test(s) || /^\s*([-*_])\1\1+\s*$/.test(s);
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^```\s*([\w+-]+)?/);
      if (fence) {
        const lang = fence[1] || ''; const buf = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        if (lang === 'mermaid') {
          out.push('<div class="lz-diagram"><span class="lz-code-lang">diagram</span><span class="lz-diagram-note">View on GitHub ↗</span></div>');
        } else {
          out.push('<pre>' + (lang ? '<span class="lz-code-lang">' + esc(lang) + '</span>' : '') + '<code>' + esc(buf.join('\n')) + '</code></pre>');
        }
        continue;
      }
      if (/^\s*$/.test(line)) { i++; continue; }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { const lv = h[1].length; out.push('<h' + lv + '>' + inline(h[2].replace(/#+\s*$/, '').trim()) + '</h' + lv + '>'); i++; continue; }
      if (/^\s*([-*_])\1\1+\s*$/.test(line)) { out.push('<hr>'); i++; continue; }
      if (/^>\s?/.test(line)) { const bq = []; while (i < lines.length && /^>\s?/.test(lines[i])) { bq.push(lines[i].replace(/^>\s?/, '')); i++; } out.push('<blockquote>' + inline(bq.join(' ')) + '</blockquote>'); continue; }
      if (/\|/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(lines[i + 1]) && /\|/.test(lines[i + 1])) {
        const header = splitRow(line); i += 2; const rows = [];
        while (i < lines.length && /\|/.test(lines[i]) && lines[i].trim() !== '') { rows.push(splitRow(lines[i])); i++; }
        let t = '<table><thead><tr>' + header.map((c) => '<th>' + inline(c) + '</th>').join('') + '</tr></thead><tbody>';
        t += rows.map((r) => '<tr>' + r.map((c) => '<td>' + inline(c) + '</td>').join('') + '</tr>').join('');
        out.push(t + '</tbody></table>'); continue;
      }
      if (/^\s*[-*+]\s+/.test(line)) { const it = []; while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { it.push(lines[i].replace(/^\s*[-*+]\s+/, '')); i++; } out.push('<ul>' + it.map((x) => '<li>' + inline(x) + '</li>').join('') + '</ul>'); continue; }
      if (/^\s*\d+\.\s+/.test(line)) { const it = []; while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { it.push(lines[i].replace(/^\s*\d+\.\s+/, '')); i++; } out.push('<ol>' + it.map((x) => '<li>' + inline(x) + '</li>').join('') + '</ol>'); continue; }
      const para = []; while (i < lines.length && !/^\s*$/.test(lines[i]) && !isBlockStart(lines[i])) { para.push(lines[i]); i++; }
      // single newlines inside a block are authored as hard breaks (e.g. the
      // **Type:** / **Languages:** / **Time:** front-matter), so join with <br>
      out.push('<p>' + para.map((ln) => inline(ln)).join('<br>') + '</p>');
    }
    return out.join('\n');
  }

  /* ---------- chrome ---------- */
  function showError(title, msg, link) {
    const art = $('#article');
    const wrap = document.createElement('div'); wrap.className = 'lz-error';
    const h = document.createElement('h2'); h.textContent = title;
    const p = document.createElement('p'); p.textContent = msg;
    wrap.appendChild(h); wrap.appendChild(p);
    if (link) { const a = document.createElement('a'); a.className = 'btn btn--primary'; a.href = link; a.target = '_blank'; a.rel = 'noopener'; a.textContent = 'Open on GitHub ↗'; wrap.appendChild(a); }
    else { const a = document.createElement('a'); a.className = 'btn'; a.href = 'course.html'; a.textContent = '← Back to Course'; wrap.appendChild(a); }
    art.replaceChildren(wrap);
  }
  function initSidebarToggle() {
    const t = $('#sidebarToggle'), sb = $('#sidebar');
    if (t) t.addEventListener('click', () => sb.classList.toggle('is-open'));
  }
  function initScrollProgress() {
    const bar = $('#scrollProgress');
    if (!bar) return;
    const onScroll = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + '%';
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }
})();
