/* ============================================================
   AI SCHOOL · projects (capstone builds across all chapters)
   Shows SHIPPED capstones only. Locked ones collapse into a
   single summary tile so the page doesn't look like a graveyard.
   Tick a capstone lesson complete in Course to unlock it here.
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

  const shippedList = allCapstones.filter((c) => c.done);
  const lockedCount = allCapstones.length - shippedList.length;

  $('#capcount').textContent = `${shippedList.length} / ${allCapstones.length} shipped`;

  /* render only shipped cards */
  const tiles = shippedList.map(({ ph, l }) => {
    const foot = el('div', { class: 'proj__foot' }, [
      el('span', { class: 'proj__tag' }, `Ch. ${String(ph.id).padStart(2, '0')}`),
      l.url
        ? el('a', { class: 'proj__link', href: l.url, target: '_blank', rel: 'noopener' }, 'View ↗')
        : el('span', { class: 'proj__st' }, 'Shipped')
    ]);
    return el('div', { class: 'proj is-shipped' }, [
      el('div', { class: 'slot proj__art' }, 'build 16×9'),
      el('div', { class: 'proj__body' }, [
        el('div', { class: 'proj__name' }, l.name),
        el('div', { class: 'proj__desc' }, l.summary || ph.name),
        foot
      ])
    ]);
  });

  /* single summary tile for all locked builds */
  if (lockedCount > 0) {
    tiles.push(el('div', { class: 'proj proj--locked-summary' }, [
      el('div', { class: 'proj__body proj__body--center' }, [
        el('div', { class: 'proj__locked-num' }, String(lockedCount)),
        el('div', { class: 'proj__name' }, 'builds locked'),
        el('div', { class: 'proj__desc' }, 'Complete capstone lessons in Course to unlock them here.'),
        el('div', { class: 'proj__foot' }, [
          el('a', { class: 'proj__link', href: 'course.html' }, 'Go to Course ↗')
        ])
      ])
    ]));
  }

  /* empty state */
  if (tiles.length === 0) {
    tiles.push(el('div', { class: 'proj proj--locked-summary' }, [
      el('div', { class: 'proj__body proj__body--center' }, [
        el('div', { class: 'proj__name' }, 'No builds shipped yet'),
        el('div', { class: 'proj__desc' }, 'Complete capstone lessons in Course. They appear here when done.'),
        el('div', { class: 'proj__foot' }, [
          el('a', { class: 'proj__link', href: 'course.html' }, 'Go to Course ↗')
        ])
      ])
    ]));
  }

  $('#builds').replaceChildren(...tiles);
})();
