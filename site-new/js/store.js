/* ============================================================
   AI SCHOOL · ProgressStore
   The ONLY thing that touches persistence. Everything else reads
   derived stats from game.js.

   Adapters:
   - localAdapter  → localStorage (default, always available)
   - restAdapter   → WordPress REST /wp-json/aischool/v1/progress
                     Active when window.WP_REST_NONCE is set.
   - vercelAdapter → Upstash Redis via /api/progress
                     Active when sa_user cookie is present (GitHub auth).
   ============================================================ */
(function (global) {
  'use strict';

  const KEY = 'aischool.progress.v1';

  /* --- local adapter (sync) --- */
  const localAdapter = {
    read()  { try { return JSON.parse(localStorage.getItem(KEY)); } catch { return null; } },
    write(p) { localStorage.setItem(KEY, JSON.stringify(p)); },
    clear() { localStorage.removeItem(KEY); }
  };

  /* --- WordPress REST adapter (async) --- */
  const restAdapter = {
    async read() {
      const r = await fetch('/wp-json/aischool/v1/progress', {
        credentials: 'same-origin',
        headers: { 'X-WP-Nonce': window.WP_REST_NONCE || '' }
      });
      if (!r.ok) return null;
      return r.json();
    },
    async write(p) {
      await fetch('/wp-json/aischool/v1/progress', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', 'X-WP-Nonce': window.WP_REST_NONCE || '' },
        body: JSON.stringify(p)
      });
    },
    clear() { return this.write(blank()); }
  };

  /* --- Vercel + Upstash adapter (async) --- */
  const vercelAdapter = {
    async read() {
      try {
        const r = await fetch('/api/progress', { credentials: 'same-origin' });
        return r.ok ? r.json() : null;
      } catch { return null; }
    },
    async write(p) {
      try {
        const r = await fetch('/api/progress', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(p),
          keepalive: true
        });
        if (!r.ok) console.error('[store] progress write failed:', r.status);
      } catch (e) { console.error('[store] progress write error:', e.message); }
    },
    clear() { return this.write(blank()); }
  };

  function hasSaUser() {
    try { return /(?:^|;)\s*sa_user=/.test(document.cookie); } catch { return false; }
  }

  function blank() { return { v: 1, done: {}, days: [], updatedAt: 0 }; }
  function today() { return new Date().toISOString().slice(0, 10); }
  function key(phaseId, idx) { return phaseId + ':' + idx; }

  const Store = {
    adapter: localAdapter,
    _cache: null,
    _ready: null,

    /* Call once per page before rendering. Returns a Promise. */
    init() {
      if (this._ready) return this._ready;

      if (window.WP_REST_NONCE) {
        this.adapter = restAdapter;
        this._ready = restAdapter.read()
          .then((data) => {
            this._cache = data && data.v ? data : (localAdapter.read() || blank());
            localAdapter.write(this._cache);
          })
          .catch(() => { this._cache = localAdapter.read() || blank(); });

      } else if (hasSaUser()) {
        this.adapter = vercelAdapter;
        this._ready = vercelAdapter.read()
          .then((data) => {
            this._cache = data && data.v ? data : (localAdapter.read() || blank());
            localAdapter.write(this._cache);
          })
          .catch(() => { this._cache = localAdapter.read() || blank(); });

      } else {
        this._cache = localAdapter.read() || blank();
        this._ready = Promise.resolve();
      }

      return this._ready;
    },

    load() {
      if (!this._cache) this._cache = localAdapter.read() || blank();
      return this._cache;
    },

    _commit() {
      const p = this._cache;
      p.updatedAt = Date.now();
      localAdapter.write(p);
      if (window.WP_REST_NONCE) restAdapter.write(p);
      else if (this.adapter === vercelAdapter) vercelAdapter.write(p);
    },

    isDone(phaseId, idx) {
      return !!this.load().done[key(phaseId, idx)];
    },

    toggle(phaseId, idx) {
      const p = this.load();
      const k = key(phaseId, idx);
      if (p.done[k]) {
        delete p.done[k];
      } else {
        p.done[k] = Date.now();
        const t = today();
        if (!p.days.includes(t)) p.days.push(t);
      }
      this._commit();
      return !!p.done[k];
    },

    reset() {
      this._cache = blank();
      localAdapter.clear();
      if (window.WP_REST_NONCE) restAdapter.write(blank());
      else if (this.adapter === vercelAdapter) vercelAdapter.write(blank());
    },

    exportJSON() { return JSON.stringify(this.load(), null, 2); },

    importJSON(raw) {
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      this._cache = Object.assign(blank(), parsed);
      this._commit();
    },

    seedIfEmpty(phases) {
      const p = this.load();
      if (Object.keys(p.done).length) return;
      phases.forEach((ph) => {
        const total = ph.lessons.length;
        let upto = 0;
        if (ph.id === 0 || ph.id === 1) upto = total;
        else if (ph.id === 2) upto = Math.round(total * 0.45);
        for (let i = 0; i < upto; i++) p.done[key(ph.id, i)] = Date.now();
      });
      for (let d = 11; d >= 0; d--) {
        const dt = new Date(); dt.setDate(dt.getDate() - d);
        p.days.push(dt.toISOString().slice(0, 10));
      }
      this._commit();
    }
  };

  (global.AIS = global.AIS || {}).store = Store;
})(window);
