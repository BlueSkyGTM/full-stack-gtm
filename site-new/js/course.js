/* ============================================================
   AI SCHOOL · course (player card + curriculum accordion)
   Player card: level, rank, XP bar, career stats.
   No avatar, no streak, no reset button.
   ============================================================ */
(function () {
  'use strict';
  const { store, game, ui } = window.AIS;
  const { el } = ui;
  const $ = (s) => document.querySelector(s);

  store.seedIfEmpty(PHASES);

  /* ---------- player card ---------- */
  function renderCard() {
    const s = game.derive(PHASES, store.load());

    /* left panel: level + rank + XP */
    $('#id').replaceChildren(
      el('div', { class: 'card__lv pix' }, 'LV.' + String(s.level).padStart(2, '0')),
      el('div', { class: 'card__rank' }, s.rank),
      el('div', { class: 'card__name' }, 'AI Engineering from Scratch'),
      el('div', { class: 'xp', style: 'margin: 16px 0 6px' }, el('i', { style: `--p:${s.pctIntoLevel}%` })),
      el('div', { class: 'xp__txt', style: 'text-align:center' },
        `${s.intoLevel} / ${s.levelSpan} XP → LV.${String(s.level + 1).padStart(2, '0')}`)
    );

    /* right panel: career stats */
    const rows = [
      ['Lessons cleared',  s.lessonsDone,    s.lessonsDone / s.lessonsTotal],
      ['Chapters cleared', s.phasesCleared,  s.phasesCleared / s.phasesTotal],
      ['Languages',        `${s.languages}/${s.languagesTotal}`, s.languages / s.languagesTotal],
      ['Build artifacts',  s.buildArtifacts, Math.min(1, s.buildArtifacts / 60)]
    ];
    $('#stats').replaceChildren(
      ...rows.map(([l, v, p]) => el('div', { class: 'stat' }, [
        el('span', { class: 'stat__l' }, l),
        el('span', { class: 'bar' }, el('i', { style: `--p:${Math.round(p * 100)}%` })),
        el('span', { class: 'stat__v' }, String(v))
      ]))
    );
  }

  // GitHub lesson URL → in-site reader link (lesson.html?path=…), or null
  function readerHref(url) {
    if (!url) return null;
    const m = url.match(/(phases\/[^/?#]+\/[^/?#]+)/);
    return m ? 'lesson.html?path=' + encodeURIComponent(m[1]) : null;
  }

  /* ---------- curriculum accordion ---------- */
  const expanded = new Set();
  let filter = 'all';

  const phaseStat = (ph) => {
    let done = 0;
    ph.lessons.forEach((_, i) => { if (store.isDone(ph.id, i)) done++; });
    const total = ph.lessons.length;
    return { done, total, pct: total ? Math.round((done / total) * 100) : 0 };
  };

  const passes = (st) =>
    filter === 'all' ||
    (filter === 'cleared'   && st.pct === 100) ||
    (filter === 'progress'  && st.pct > 0 && st.pct < 100) ||
    (filter === 'untouched' && st.pct === 0);

  function lessonRow(ph, ls, i) {
    const done = store.isDone(ph.id, i);
    return el('div', { class: 'lesson' + (done ? ' is-done' : '') }, [
      el('button', { class: 'lesson__chk' + (done ? ' is-done' : ''), 'aria-label': 'toggle',
        onClick: () => { store.toggle(ph.id, i); renderCard(); render(); } }, done ? '✓' : ''),
      el('div', { class: 'lesson__main' }, [
        readerHref(ls.url)
          ? el('a', { class: 'lesson__nm', href: readerHref(ls.url) }, ls.name)
          : el('div', { class: 'lesson__nm' }, ls.name),
        el('div', { class: 'lesson__meta' }, [
          el('span', { class: 'lesson__type', 'data-t': ls.type }, ls.type),
          el('span', null, ls.lang || '—')
        ])
      ])
    ]);
  }

  function phaseBlock(ph) {
    const st = phaseStat(ph);
    const open = expanded.has(ph.id);
    const head = el('button', { class: 'toc__phase', 'aria-expanded': open ? 'true' : 'false',
      onClick: () => { open ? expanded.delete(ph.id) : expanded.add(ph.id); render(); } }, [
      el('span', { class: 'toc__num' }, String(ph.id).padStart(2, '0')),
      el('span', null, [
        el('span', { class: 'toc__nm' + (st.pct === 100 ? ' is-done' : '') }, ph.name),
        ph.id === 19 ? el('span', { class: 'toc__badge' }, 'capstone') : null
      ].filter(Boolean)),
      el('span', { class: 'xp toc__count mini', style: 'height:8px' }, el('i', { style: `--p:${st.pct}%` })),
      el('span', { class: 'toc__count' }, `${st.done}/${st.total}`),
      el('span', { class: 'toc__chev' }, '▸')
    ]);
    const frag = [head];
    if (open) frag.push(el('div', { class: 'toc__lessons' }, ph.lessons.map((ls, i) => lessonRow(ph, ls, i))));
    return frag;
  }

  function render() {
    const list = PHASES.filter((ph) => passes(phaseStat(ph)));
    const host = $('#toc');
    host.replaceChildren();
    list.forEach((ph) => phaseBlock(ph).forEach((n) => host.append(n)));
    const s = game.derive(PHASES, store.load());
    $('#count').textContent = `${list.length} of ${PHASES.length} chapters · ${s.lessonsDone}/${s.lessonsTotal} lessons cleared`;
  }

  $('#filter').addEventListener('change', (e) => { filter = e.target.value; render(); });
  renderCard();
  render();
})();
