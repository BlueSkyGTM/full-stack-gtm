/* ============================================================
   AI SCHOOL · projects (capstone builds across all chapters)
   Shows ALL capstones — shipped ones open, locked ones with
   real content but a lock indicator. Sorted: shipped first.
   ============================================================ */
(async function () {
  'use strict';
  await window.AIS.store.init();
  const { store, ui } = window.AIS;
  const { el } = ui;
  const $ = (s) => document.querySelector(s);

  /* Phase 19 is the capstone chapter — show only the 17 main capstones (P-prefixed combines) */
  const phase19 = PHASES.find((p) => p.id === 19) || { lessons: [] };
  const allCapstones = phase19.lessons
    .map((l, i) => ({ ph: phase19, l, i, done: store.isDone(phase19.id, i) }))
    .filter(({ l }) => l.combines && /^P/.test(l.combines));

  /* shipped first, then curriculum order */
  allCapstones.sort((a, b) => (b.done - a.done) || (a.i - b.i));

  const shippedCount = allCapstones.filter((c) => c.done).length;
  $('#capcount').textContent = shippedCount === 0
    ? `${allCapstones.length} projects · none shipped yet`
    : `${shippedCount} of ${allCapstones.length} shipped`;

  const tiles = allCapstones.map(({ ph, l, i, done }) => {
    const chTag = `Ch. ${String(ph.id).padStart(2, '0')}`;
    const foot  = el('div', { class: 'proj__foot' }, [
      el('span', { class: 'proj__tag' }, chTag),
      done && l.url
        ? el('a',    { class: 'proj__link', href: l.url, target: '_blank', rel: 'noopener' }, 'View ↗')
        : done
          ? el('span', { class: 'proj__st is-shipped' }, 'Shipped')
          : el('a',    { class: 'proj__link proj__link--locked', href: 'course.html' }, '[ locked ] — Go to Course')
    ]);

    return el('div', { class: 'proj ' + (done ? 'is-shipped' : 'is-locked') }, [
      el('div', { class: 'proj__body' }, [
        el('div', { class: 'proj__name' }, l.name),
        el('div', { class: 'proj__desc' }, l.summary || ph.name),
        foot
      ])
    ]);
  });

  $('#builds').replaceChildren(...tiles);
})();
