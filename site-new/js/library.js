/* ============================================================
   AI SCHOOL · library (curated reading list)
   Render-only. Reads the LIBRARY global (js/library-data.js),
   groups by tier, builds topic filter chips. No persistence.
   ============================================================ */
(function () {
  'use strict';
  const { ui } = window.AIS;
  const { el } = ui;
  const $ = (s) => document.querySelector(s);

  const TIERS = [
    { key: 'fundamentals', label: 'Fundamentals', desc: 'Math, Python, and algorithms — the foundation everything else is built on.' },
    { key: 'intermediate', label: 'Intermediate', desc: 'Machine learning, deep learning, NLP, computer vision, data science, Julia.' },
    { key: 'advanced', label: 'Advanced', desc: 'LLMs, agents, MLOps, production systems, Rust, TypeScript, prompt engineering.' }
  ];

  let active = 'all';

  function bookRow(b) {
    return el('a', { class: 'book', 'data-topic': b.topic, href: b.url, target: '_blank', rel: 'noopener' }, [
      el('div', { class: 'book__info' }, [
        el('span', { class: 'book__title' }, b.title),
        el('div', { class: 'book__author' }, b.author),
        el('div', { class: 'book__note' }, b.note)
      ]),
      el('div', { class: 'book__meta' }, [
        el('span', { class: 'book__topic' }, b.topic),
        el('span', { class: 'book__format' }, b.format)
      ])
    ]);
  }

  function tierBlock(t) {
    const books = LIBRARY[t.key] || [];
    return el('div', { class: 'lib-tier', 'data-tier': t.key }, [
      el('div', { class: 'lib-tier__hd' }, [
        el('span', { class: 'lib-tier__label' }, t.label),
        el('span', { class: 'lib-tier__count' }, books.length + ' resources')
      ]),
      el('p', { class: 'lib-tier__desc' }, t.desc),
      el('div', { class: 'book-list' }, books.map(bookRow))
    ]);
  }

  function applyFilter() {
    document.querySelectorAll('.book').forEach((row) => {
      const match = active === 'all' || row.getAttribute('data-topic') === active;
      row.classList.toggle('is-hidden', !match);
    });
    // hide tiers that have no visible rows
    document.querySelectorAll('.lib-tier').forEach((tier) => {
      const any = [...tier.querySelectorAll('.book')].some((r) => !r.classList.contains('is-hidden'));
      tier.classList.toggle('is-hidden', !any);
    });
  }

  function setFilter(topic) {
    active = topic;
    document.querySelectorAll('.chip').forEach((c) => c.classList.toggle('is-on', c.getAttribute('data-topic') === topic));
    applyFilter();
  }

  function render() {
    // tiers
    $('#tiers').replaceChildren(...TIERS.map(tierBlock));

    // topic chips (unique, sorted)
    const topics = new Set();
    TIERS.forEach((t) => (LIBRARY[t.key] || []).forEach((b) => topics.add(b.topic)));
    const chips = $('#chips');
    const allChip = el('button', { class: 'chip is-on', 'data-topic': 'all' }, 'All');
    chips.replaceChildren(allChip, ...[...topics].sort().map((tp) => el('button', { class: 'chip', 'data-topic': tp }, tp)));
    chips.addEventListener('click', (e) => { const b = e.target.closest('.chip'); if (b) setFilter(b.getAttribute('data-topic')); });

    const total = TIERS.reduce((n, t) => n + (LIBRARY[t.key] || []).length, 0);
    $('#count').textContent = total + ' free resources';
  }

  render();
})();
