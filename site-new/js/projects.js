/* ============================================================
   AI SCHOOL · projects (capstone builds across all chapters)
   Collects capstone lessons from every chapter — both the
   dedicated capstone lesson that closes each chapter and all
   Phase 19 final builds. Sorted: shipped first, then by chapter.
   ============================================================ */
(function () {
  'use strict';
  const { store, ui } = window.AIS;
  const { el } = ui;
  const $ = (s) => document.querySelector(s);

  store.seedIfEmpty(PHASES);

  /* collect capstones from all chapters */
  const allCapstones = [];
  PHASES.forEach((ph) => {
    ph.lessons.forEach((l, i) => {
      if (
        l.type === 'Capstone' ||
        (l.url  && l.url.toLowerCase().includes('capstone')) ||
        (l.name && l.name.toLowerCase().includes('apstone'))
      ) {
        allCapstones.push({ ph, l, i, done: store.isDone(ph.id, i) });
      }
    });
  });

  /* shipped first, then by chapter id */
  allCapstones.sort((a, b) => (b.done - a.done) || (a.ph.id - b.ph.id));

  const shipped = allCapstones.filter((c) => c.done).length;
  $('#capcount').textContent = `${shipped} / ${allCapstones.length} shipped`;

  const tiles = allCapstones.map(({ ph, l, i, done }) => {
    const cls = 'proj ' + (done ? 'is-shipped' : 'is-locked');
    const foot = el('div', { class: 'proj__foot' }, [
      el('span', { class: 'proj__tag' }, `Ch. ${String(ph.id).padStart(2, '0')}`),
      done && l.url
        ? el('a', { class: 'proj__link', href: l.url, target: '_blank', rel: 'noopener' }, 'View ↗')
        : el('span', { class: 'proj__st' }, done ? 'Shipped' : 'Complete lesson in Course ↗')
    ]);
    return el('div', { class: cls }, [
      el('div', { class: 'slot proj__art' }, done ? 'build 16×9' : '🔒 locked'),
      el('div', { class: 'proj__body' }, [
        el('div', { class: 'proj__name' }, l.name),
        el('div', { class: 'proj__desc' }, l.summary || ph.name),
        foot
      ])
    ]);
  });

  $('#builds').replaceChildren(...tiles);
})();
