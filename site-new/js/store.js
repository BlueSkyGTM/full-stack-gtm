/* ============================================================
   AI SCHOOL · ProgressStore
   The ONLY thing that touches persistence. Everything else reads
   derived stats from game.js. Swap `adapter` to move from
   localStorage → WordPress REST with zero screen changes.
   ============================================================ */
(function (global) {
  'use strict';

  const KEY = 'aischool.progress.v1';

  /* --- adapters: same 3 methods, different backend --- */
  const localAdapter = {
    read()  { try { return JSON.parse(localStorage.getItem(KEY)); } catch { return null; } },
    write(p) { localStorage.setItem(KEY, JSON.stringify(p)); },
    clear() { localStorage.removeItem(KEY); }
  };

  // Later: const restAdapter = { read: () => fetch('/wp-json/aischool/v1/progress')…, … }
  // Store.adapter = restAdapter;  ← only line that changes.

  function blank() {
    return { v: 1, done: {}, days: [], updatedAt: 0 };
  }

  function today() { return new Date().toISOString().slice(0, 10); }
  function key(phaseId, idx) { return phaseId + ':' + idx; }

  const Store = {
    adapter: localAdapter,
    _cache: null,

    load() {
      if (!this._cache) this._cache = this.adapter.read() || blank();
      return this._cache;
    },

    _commit() {
      const p = this._cache;
      p.updatedAt = Date.now();
      this.adapter.write(p);
    },

    isDone(phaseId, idx) {
      return !!this.load().done[key(phaseId, idx)];
    },

    /** Toggle a lesson. Returns the new done state. Records today for streaks. */
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
      this.adapter.clear();
    },

    exportJSON() {
      return JSON.stringify(this.load(), null, 2);
    },

    importJSON(raw) {
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      this._cache = Object.assign(blank(), parsed);
      this._commit();
    },

    /** One-time demo state so the design reads as "in progress" on first open. */
    seedIfEmpty(phases) {
      const p = this.load();
      if (Object.keys(p.done).length) return;
      phases.forEach((ph) => {
        const total = ph.lessons.length;
        let upto = 0;
        if (ph.id === 0 || ph.id === 1) upto = total;          // cleared
        else if (ph.id === 2) upto = Math.round(total * 0.45); // in progress
        for (let i = 0; i < upto; i++) p.done[key(ph.id, i)] = Date.now();
      });
      // 12-day streak
      for (let d = 11; d >= 0; d--) {
        const dt = new Date(); dt.setDate(dt.getDate() - d);
        p.days.push(dt.toISOString().slice(0, 10));
      }
      this._commit();
    }
  };

  (global.AIS = global.AIS || {}).store = Store;
})(window);
