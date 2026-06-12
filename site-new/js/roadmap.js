/* ============================================================
   AI SCHOOL · roadmap (chapter overview grid)
   20 chapters arranged in a 4-column grid in curriculum order.
   Click any chapter to see what it teaches, its prerequisites,
   and what it unlocks. No bezier graph — just a clean grid.
   ============================================================ */
(function () {
  'use strict';
  const { store, game, ui } = window.AIS;
  const { el } = ui;
  const $ = (s) => document.querySelector(s);

  const stats  = game.derive(PHASES, store.load());
  const byId   = (id) => stats.phaseStats.find((p) => p.id === id);
  const phById = (id) => PHASES.find((p) => p.id === id);

  const PREREQS = {
    0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [3], 6: [3], 7: [3], 8: [3, 7],
    9: [3], 10: [7], 11: [10], 12: [4, 6, 11], 13: [11], 14: [11, 13],
    15: [14], 16: [15], 17: [11], 18: [10], 19: [14, 15, 16, 17]
  };
  const UNLOCKS = {};
  Object.entries(PREREQS).forEach(([id, reqs]) =>
    reqs.forEach((r) => (UNLOCKS[r] = UNLOCKS[r] || []).push(+id)));

  /* ---------- info card ---------- */
  function showCard(id) {
    const ph = phById(id);
    const ps = byId(id);
    if (!ph) return;
    const prereqs  = (PREREQS[id]  || []).map((pid) => { const p = phById(pid);  return p ? `Ch.${String(pid).padStart(2,'0')} ${p.name}`  : null; }).filter(Boolean);
    const unlocks  = (UNLOCKS[id]  || []).map((uid) => { const p = phById(uid);  return p ? `Ch.${String(uid).padStart(2,'0')} ${p.name}`  : null; }).filter(Boolean);

    $('#cardBody').replaceChildren(
      el('div', { class: 'drawer__num' },
        `Chapter ${String(id).padStart(2,'0')} · ${ph.lessons.length} lessons · ${ps.pct}% complete`),
      el('h2', { class: 'drawer__title' }, ph.name),
      ph.desc ? el('p', { class: 'drawer__desc' }, ph.desc) : null,
      prereqs.length
        ? el('div', { class: 'drawer__rec' }, [
            el('span', { class: 'drawer__rec-l' }, 'Needs'),
            el('span', null, prereqs.join(' · '))
          ])
        : el('div', { class: 'drawer__rec' }, [
            el('span', { class: 'drawer__rec-l' }, 'Starting point'),
            el('span', { class: 'drawer__rec-note' }, 'No prerequisites — jump in anytime.')
          ]),
      unlocks.length
        ? el('div', { class: 'drawer__rec' }, [
            el('span', { class: 'drawer__rec-l' }, 'Unlocks'),
            el('span', null, unlocks.join(' · '))
          ])
        : null
    );
    $('#chapterCard').hidden = false;
  }
  function hideCard() { $('#chapterCard').hidden = true; }

  /* ---------- grid ---------- */
  let selected = null;

  function renderGrid() {
    const grid = el('div', { class: 'rm-grid' });
    PHASES.forEach((ph) => {
      const ps  = byId(ph.id);
      const cls = ['rm-tile',
        ps.pct === 100 ? 'is-done' : ps.pct > 0 ? 'is-active' : 'is-locked',
        ph.id === 19 ? 'rm-tile--boss' : ''
      ].join(' ');

      const tile = el('button', {
        class: cls, 'data-id': ph.id,
        'aria-label': `Chapter ${ph.id}: ${ph.name}`,
        onClick: () => {
          selected = (selected === ph.id) ? null : ph.id;
          $('#tree').querySelectorAll('.rm-tile').forEach((t) =>
            t.classList.toggle('rm-tile--sel', selected != null && +t.dataset.id === selected));
          $('#clear').hidden = selected == null;
          selected != null ? showCard(selected) : hideCard();
        }
      }, [
        el('span', { class: 'rm-tile__num' }, String(ph.id).padStart(2, '0')),
        el('span', { class: 'rm-tile__nm'  }, ph.name),
        el('div',  { class: 'xp rm-tile__bar' }, el('i', { style: `--p:${ps.pct}%` })),
        el('span', { class: 'rm-tile__pct' }, ps.pct === 100 ? '✓' : ps.pct + '%')
      ]);
      tile.dataset.id = ph.id;
      grid.append(tile);
    });
    $('#tree').replaceChildren(grid);
  }

  renderGrid();
  $('#clear').addEventListener('click', () => {
    selected = null;
    $('#clear').hidden = true;
    $('#tree').querySelectorAll('.rm-tile--sel').forEach((t) => t.classList.remove('rm-tile--sel'));
    hideCard();
  });
  $('#cardClose').addEventListener('click', () => {
    selected = null;
    $('#clear').hidden = true;
    $('#tree').querySelectorAll('.rm-tile--sel').forEach((t) => t.classList.remove('rm-tile--sel'));
    hideCard();
  });
})();
