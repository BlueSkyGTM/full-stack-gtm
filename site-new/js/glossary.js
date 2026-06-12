/* ============================================================
   AI SCHOOL · glossary (click a card to flip say → means)
   ============================================================ */
(function () {
  'use strict';
  const { ui } = window.AIS;
  const { el } = ui;
  const $ = (s) => document.querySelector(s);

  let query = '';

  function card(t) {
    const c = el('button', { class: 'gloss-card' }, [
      el('span', { class: 'gloss-hint' }, 'click →'),
      el('div', { class: 'gloss-term' }, t.term),
      el('div', { class: 'gloss-face-say' }, [
        el('div', { class: 'gloss-lbl' }, 'What people say'),
        el('div', { class: 'gloss-say' }, '“' + t.says + '”')
      ]),
      el('div', { class: 'gloss-mean' }, [
        el('div', { class: 'gloss-lbl', style: 'color:var(--terra)' }, 'What it actually means'),
        document.createTextNode(t.means)
      ])
    ]);
    c.addEventListener('click', () => {
      const flipped = c.classList.toggle('is-flipped');
      c.querySelector('.gloss-hint').textContent = flipped ? '← say' : 'click →';
    });
    return c;
  }

  function render() {
    const terms = !query ? GLOSSARY : GLOSSARY.filter((t) =>
      (t.term + ' ' + t.says + ' ' + t.means).toLowerCase().includes(query.toLowerCase()));
    $('#count').textContent = `${terms.length} of ${GLOSSARY.length} terms`;
    const host = $('#grid');
    if (!terms.length) { host.replaceChildren(el('div', { class: 'cat-empty' }, 'No terms match your search.')); return; }
    host.replaceChildren(...terms.map(card));
  }

  $('#search').addEventListener('input', (e) => { query = e.target.value.trim(); render(); });

  // Deep-link support: ?q=Term pre-fills the search (used by the ⌘K palette)
  try {
    const q0 = new URLSearchParams(location.search).get('q');
    if (q0) { query = q0; const box = $('#search'); if (box) box.value = q0; }
  } catch (_) {}
  render();
})();
