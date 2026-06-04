/* ============================================================
   AI SCHOOL · catalog (the searchable lesson index)
   Every lesson, flattened into one fast filterable list. This
   is the "look something up" view — search box + type/status
   chips + a sortable manifest. Status reflects YOUR progress.
   (The visual dependency view is the Roadmap page.)
   ============================================================ */
(async function () {
  'use strict';
  const { store, game, ui } = window.AIS;
  const { el } = ui;
  const $ = (s) => document.querySelector(s);
  await store.init();


  // GitHub lesson URL → in-site reader link (lesson.html?path=…), or null
  function readerHref(url) {
    if (!url) return null;
    const m = url.match(/(phases\/[^/?#]+\/[^/?#]+)/);
    return m ? 'lesson.html?path=' + encodeURIComponent(m[1]) : null;
  }

  /* flatten every LIVE lesson into a searchable row — hide planned (not yet available) */
  const rows = [];
  PHASES.forEach((p) => p.lessons.forEach((l, i) => {
    if (l.status !== 'complete') return;  // hide lessons not yet live in curriculum
    // Phase 19: only show the main capstone entries (P-prefixed combines), same as course view
    if (p.id === 19 && !(l.combines && /^P/.test(l.combines))) return;
    rows.push({
      phase: p.id, phaseName: p.name, idx: i,
      name: l.name, type: l.type, lang: l.lang || '—',
      live: true, url: l.url || ''
    });
  }));
  const userStatus = (r) => store.isDone(r.phase, r.idx) ? 'cleared' : 'open';

  let sortCol = 'phase', sortDir = 1, query = '', fType = 'all', fStatus = 'all', fPhase = '';

  /* chapter dropdown */
  $('#fPhase').append(...PHASES.map((p) =>
    el('option', { value: String(p.id) }, String(p.id).padStart(2, '0') + ' · ' + p.name)));

  /* chip groups (segmented filters) — more game-UI than dropdowns */
  function wireChips(sel, set) {
    const group = $(sel);
    group.addEventListener('click', (e) => {
      const b = e.target.closest('.chip'); if (!b) return;
      group.querySelectorAll('.chip').forEach((c) => c.classList.toggle('is-on', c === b));
      set(b.dataset.v);
      render();
    });
  }
  wireChips('#typeChips', (v) => (fType = v));
  wireChips('#statusChips', (v) => (fStatus = v));

  function render() {
    let out = rows.filter((r) => {
      if (fType !== 'all' && r.type !== fType) return false;
      if (fStatus !== 'all' && userStatus(r) !== fStatus) return false;
      if (fPhase && String(r.phase) !== fPhase) return false;
      if (query) return (r.name + ' ' + r.phaseName + ' ' + r.type + ' ' + r.lang).toLowerCase().includes(query.toLowerCase());
      return true;
    });
    out.sort((a, b) => {
      const get = (r) => sortCol === 'phase' ? r.phase : sortCol === 'status' ? userStatus(r) : String(r[sortCol]).toLowerCase();
      const va = get(a), vb = get(b);
      return (va < vb ? -1 : va > vb ? 1 : 0) * sortDir;
    });

    $('#count').textContent = out.length === rows.length
      ? `${rows.length} lessons`
      : `${out.length} of ${rows.length} lessons`;
    const body = $('#body');
    if (!out.length) { body.replaceChildren(el('tr', null, el('td', { colspan: '5', class: 'cat-empty' }, 'No lessons match — try clearing a filter.'))); return; }
    body.replaceChildren(...out.map((r) => {
      const st = userStatus(r);
      const href = readerHref(r.url) || r.url || null;
      const row = el('tr', { class: href ? 'is-link' : '' }, [
        el('td', null, el('span', { class: 'cat-phase' }, String(r.phase).padStart(2, '0'))),
        el('td', null, el('span', { class: 'cat-name' }, r.name)),
        el('td', null, el('span', { class: 'cat-tag', 'data-t': r.type }, r.type)),
        el('td', null, el('span', { class: 'cat-lang' }, r.lang)),
        el('td', null, el('span', { class: 'cat-stat', 'data-s': st }, st))
      ]);
      if (href) row.addEventListener('click', () => window.open(href, '_blank'));
      return row;
    }));
  }

  document.querySelectorAll('th[data-sort]').forEach((th) => th.addEventListener('click', () => {
    const col = th.getAttribute('data-sort');
    if (sortCol === col) sortDir *= -1; else { sortCol = col; sortDir = 1; }
    document.querySelectorAll('th .arr').forEach((a) => (a.textContent = ''));
    th.querySelector('.arr').textContent = sortDir === 1 ? ' ▲' : ' ▼';
    render();
  }));
  $('#search').addEventListener('input', (e) => { query = e.target.value.trim(); render(); });
  $('#fPhase').addEventListener('change', (e) => { fPhase = e.target.value; render(); });

  render();
})();
