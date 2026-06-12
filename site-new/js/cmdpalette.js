/* ============================================================
   AI SCHOOL · command palette (⌘K / Ctrl+K, or the search button)
   Searches lesson titles, summaries, phase names, languages, types
   and glossary terms — entirely client-side from data.js. No network.
   Lessons open the in-site reader (lesson.html?path=…); terms deep-link
   into the glossary (glossary.html?q=…).
   Public API: window.CmdPalette.open() / .close()
   ============================================================ */
(function () {
  'use strict';

  var PALETTE_ID = 'cmdPalette', MAX_RESULTS = 12, BODY_ATTR = 'data-palette-open';
  var _index = null, _activeIdx = -1, _isOpen = false, _prevFocus = null;

  /* ---- search index (built once from PHASES + GLOSSARY) ---- */
  function buildIndex() {
    if (_index !== null) return _index;
    _index = [];
    if (typeof PHASES !== 'undefined' && Array.isArray(PHASES)) {
      for (var i = 0; i < PHASES.length; i++) {
        var phase = PHASES[i];
        for (var j = 0; j < phase.lessons.length; j++) {
          var lesson = phase.lessons[j];
          var lessonPath = '';
          if (lesson.url) { var m = lesson.url.match(/(phases\/[^/?#]+\/[^/?#]+)/); if (m) lessonPath = m[1]; }
          _index.push({
            kind: 'lesson', phaseId: phase.id, phaseName: phase.name,
            name: lesson.name || '', summary: lesson.summary || '', keywords: lesson.keywords || '',
            type: lesson.type || '', lang: lesson.lang || '', lessonPath: lessonPath, url: lesson.url || ''
          });
        }
      }
    }
    if (typeof GLOSSARY !== 'undefined' && Array.isArray(GLOSSARY)) {
      for (var k = 0; k < GLOSSARY.length; k++) {
        var g = GLOSSARY[k];
        _index.push({ kind: 'glossary', name: g.term || '', summary: g.means || '', says: g.says || '' });
      }
    }
    return _index;
  }

  /* ---- scoring ---- */
  function scoreItem(item, q) {
    var name = item.name.toLowerCase(), summary = (item.summary || '').toLowerCase();
    var keywords = (item.keywords || '').toLowerCase(), phase = (item.phaseName || '').toLowerCase();
    var lang = (item.lang || '').toLowerCase(), type = (item.type || '').toLowerCase(), says = (item.says || '').toLowerCase();
    var s = 0;
    if (name === q) return 200;
    if (name.indexOf(q) === 0) s += 100; else if (name.indexOf(q) !== -1) s += 70;
    var words = q.split(/\s+/).filter(Boolean);
    if (words.length > 1) {
      var allInName = words.every(function (w) { return name.indexOf(w) !== -1; });
      if (allInName) s += (s === 0 ? 65 : 20);
      else { var blob = name + ' ' + summary + ' ' + keywords + ' ' + phase; if (words.every(function (w) { return blob.indexOf(w) !== -1; })) s += 15; }
    }
    if (summary.indexOf(q) !== -1) s += 25;
    if (keywords.indexOf(q) !== -1) s += 22;
    if (says.indexOf(q) !== -1) s += 22;
    if (phase.indexOf(q) !== -1) s += 18;
    if (lang.indexOf(q) !== -1) s += 14;
    if (type.indexOf(q) !== -1) s += 10;
    if (s === 0 && words.length === 1) {
      var parts = name.split(/[\s\-–—:,]+/).filter(Boolean);
      for (var i = 0; i < parts.length; i++) { if (parts[i].indexOf(q) === 0) { s += 30; break; } }
      if (s === 0 && keywords.indexOf(q) !== -1) s += 18;
      if (s === 0 && summary.indexOf(q) !== -1) s += 12;
    }
    return s;
  }

  function search(query) {
    var q = query.trim().toLowerCase();
    if (!q) return [];
    var items = buildIndex(), results = [];
    for (var i = 0; i < items.length; i++) { var s = scoreItem(items[i], q); if (s > 0) results.push({ item: items[i], s: s }); }
    results.sort(function (a, b) { return b.s - a.s; });
    return results.slice(0, MAX_RESULTS).map(function (r) { return r.item; });
  }

  /* ---- utils ---- */
  function escHtml(str) { var d = document.createElement('div'); d.textContent = (str == null) ? '' : String(str); return d.innerHTML; }
  function highlight(text, query) {
    if (!text) return '';
    if (!query) return escHtml(text);
    var lower = text.toLowerCase(), q = query.trim().toLowerCase(), idx = lower.indexOf(q), len = q.length;
    if (idx === -1) { var words = q.split(/\s+/).filter(Boolean); for (var i = 0; i < words.length; i++) { idx = lower.indexOf(words[i]); if (idx !== -1) { len = words[i].length; break; } } }
    if (idx === -1) return escHtml(text);
    return escHtml(text.slice(0, idx)) + '<mark>' + escHtml(text.slice(idx, idx + len)) + '</mark>' + escHtml(text.slice(idx + len));
  }
  function truncate(str, max) { if (!str || str.length <= max) return str || ''; var cut = str.slice(0, max).replace(/\s+\S*$/, ''); return (cut.length > max * 0.6 ? cut : str.slice(0, max)) + '…'; }

  /* ---- DOM (lazy) ---- */
  function createPaletteDOM() {
    if (document.getElementById(PALETTE_ID)) return;
    var isMac = /Mac|iPhone|iPod|iPad/.test((navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '');
    var shortcut = isMac ? '⌘K' : 'Ctrl+K';
    var el = document.createElement('div');
    el.id = PALETTE_ID; el.setAttribute('role', 'dialog'); el.setAttribute('aria-modal', 'true'); el.setAttribute('aria-label', 'Search lessons and glossary');
    el.innerHTML =
      '<div class="cp-backdrop" id="cpBackdrop"></div>' +
      '<div class="cp-panel">' +
        '<div class="cp-search-row">' +
          '<svg class="cp-search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>' +
          '<input class="cp-input" id="cpInput" type="search" placeholder="Search lessons and glossary…" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" aria-label="Search" aria-controls="cpResults">' +
          '<kbd class="cp-kbd-esc" id="cpKbdEsc">Esc</kbd>' +
        '</div>' +
        '<ul class="cp-results" id="cpResults" role="listbox" aria-label="Search results"></ul>' +
        '<div class="cp-footer">' +
          '<span class="cp-footer-group"><kbd>↑</kbd><kbd>↓</kbd><span class="cp-footer-label">navigate</span></span>' +
          '<span class="cp-footer-group"><kbd>↵</kbd><span class="cp-footer-label">open</span></span>' +
          '<span class="cp-footer-group"><kbd>Esc</kbd><span class="cp-footer-label">close</span></span>' +
          '<span class="cp-footer-shortcut">' + shortcut + '</span>' +
        '</div>' +
      '</div>';
    document.body.appendChild(el);
    document.getElementById('cpBackdrop').addEventListener('click', close);
    document.getElementById('cpKbdEsc').addEventListener('click', close);
    var inp = document.getElementById('cpInput');
    inp.addEventListener('input', _onInput);
    inp.addEventListener('keydown', _onKeyDown);
  }
  function _palEl() { return document.getElementById(PALETTE_ID); }
  function _inputEl() { return document.getElementById('cpInput'); }
  function _listEl() { return document.getElementById('cpResults'); }

  /* ---- open / close ---- */
  function open() {
    if (_isOpen) { var inp0 = _inputEl(); if (inp0) inp0.focus(); return; }
    _prevFocus = document.activeElement || null; _isOpen = true; _activeIdx = -1;
    createPaletteDOM();
    document.body.setAttribute(BODY_ATTR, '');
    requestAnimationFrame(function () {
      var pal = _palEl(); if (pal) pal.classList.add('cp-open');
      requestAnimationFrame(function () { var inp = _inputEl(); if (inp) { inp.focus(); var q = inp.value.trim(); renderResults(q ? search(q) : []); } });
    });
  }
  function close() {
    if (!_isOpen) return;
    _isOpen = false; _activeIdx = -1;
    var pal = _palEl(); if (pal) pal.classList.remove('cp-open');
    document.body.removeAttribute(BODY_ATTR);
    try { if (_prevFocus && _prevFocus.focus) _prevFocus.focus(); } catch (_) {}
    _prevFocus = null;
  }

  /* ---- render ---- */
  function renderResults(results) {
    var list = _listEl(); if (!list) return;
    var query = (_inputEl() ? _inputEl().value : '').trim();
    if (!query) { list.innerHTML = '<li class="cp-empty">Type to search lessons and glossary terms</li>'; _activeIdx = -1; return; }
    if (results.length === 0) { list.innerHTML = '<li class="cp-empty">No results for <em>' + escHtml(query) + '</em></li>'; _activeIdx = -1; return; }
    var html = '';
    for (var i = 0; i < results.length; i++) {
      var r = results[i], dest = '', chip = '', chipClass = 'cp-item-chip';
      if (r.kind === 'lesson') {
        dest = r.lessonPath ? 'lesson.html?path=' + encodeURIComponent(r.lessonPath) : r.url;
        chip = 'Phase ' + String(r.phaseId).padStart(2, '0');
      } else {
        dest = 'glossary.html?q=' + encodeURIComponent(r.name);
        chip = 'Glossary'; chipClass += ' cp-item-chip--alt';
      }
      var snippet = r.summary ? truncate(r.summary, 110) : '';
      var meta = '';
      if (r.kind === 'lesson') { var mp = []; if (r.type && r.type !== '—') mp.push(r.type); if (r.lang && r.lang !== '—') mp.push(r.lang); meta = mp.join(' · '); }
      html +=
        '<li class="cp-item" role="option" aria-selected="false" data-idx="' + i + '" data-href="' + escHtml(dest) + '">' +
          '<div class="cp-item-body">' +
            '<span class="' + chipClass + '">' + escHtml(chip) + '</span>' +
            '<span class="cp-item-name">' + highlight(r.name, query) + '</span>' +
            (snippet ? '<span class="cp-item-summary">' + highlight(snippet, query) + '</span>' : '') +
            (meta ? '<span class="cp-item-meta">' + escHtml(meta) + '</span>' : '') +
          '</div>' +
          '<svg class="cp-item-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>' +
        '</li>';
    }
    list.innerHTML = html; _activeIdx = -1;
    var items = list.querySelectorAll('.cp-item');
    for (var j = 0; j < items.length; j++) { items[j].addEventListener('click', _onItemClick); items[j].addEventListener('mousemove', _onItemMouseMove); }
  }

  function _onInput(e) { renderResults(search(e.target.value)); _activeIdx = -1; }
  function _onKeyDown(e) {
    var list = _listEl(), items = list ? list.querySelectorAll('.cp-item') : [], count = items.length;
    switch (e.key) {
      case 'ArrowDown': e.preventDefault(); if (!count) return; _activeIdx = (_activeIdx + 1) % count; _updateActive(items); break;
      case 'ArrowUp': e.preventDefault(); if (!count) return; _activeIdx = (_activeIdx - 1 + count) % count; _updateActive(items); break;
      case 'Enter': { e.preventDefault(); var target = (_activeIdx >= 0 && items[_activeIdx]) ? items[_activeIdx] : (count === 1 ? items[0] : null); if (target) _navigate(target); break; }
      case 'Tab': e.preventDefault(); break;
      case 'Escape': e.preventDefault(); close(); break;
    }
  }
  function _updateActive(items) {
    for (var i = 0; i < items.length; i++) {
      var active = (i === _activeIdx);
      items[i].classList.toggle('cp-item--active', active);
      items[i].setAttribute('aria-selected', active ? 'true' : 'false');
      if (active) items[i].scrollIntoView({ block: 'nearest' });
    }
  }
  function _onItemClick(e) { _navigate(e.currentTarget); }
  function _onItemMouseMove(e) { var list = _listEl(); if (!list) return; var idx = parseInt(e.currentTarget.getAttribute('data-idx'), 10); if (idx !== _activeIdx) { _activeIdx = idx; _updateActive(list.querySelectorAll('.cp-item')); } }
  function _navigate(item) { var href = item.getAttribute('data-href'); if (!href) return; close(); window.location.href = href; }

  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      if (_isOpen) { var inp = _inputEl(); if (inp) inp.focus(); } else open();
    }
  });

  function _init() {
    var triggers = document.querySelectorAll('[data-cmd-palette]');
    for (var i = 0; i < triggers.length; i++) triggers[i].addEventListener('click', function (e) { e.preventDefault(); open(); });
    buildIndex();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', _init); else _init();

  window.CmdPalette = { open: open, close: close };
}());
