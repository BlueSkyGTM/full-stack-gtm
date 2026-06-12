/* ============================================================
   Synapse Academy · Auth
   Reads sa_user cookie (set by /api/auth/callback).
   Swaps AIS.store to the Vercel adapter when logged in.
   Updates the login button in the header.
   ============================================================ */
(function () {
  'use strict';

  function getSaUser() {
    const m = document.cookie.match(/(?:^|;)\s*sa_user=([^;]*)/);
    if (!m) return null;
    const val = decodeURIComponent(m[1]);
    const colon = val.indexOf(':');
    return colon > 0
      ? { login: val.slice(0, colon), avatar: val.slice(colon + 1) }
      : null;
  }

  /* Vercel adapter — same interface as localAdapter / restAdapter */
  const vercelAdapter = {
    async read() {
      try {
        const r = await fetch('/api/progress', { credentials: 'same-origin' });
        if (!r.ok) return null;
        return r.json();
      } catch { return null; }
    },
    async write(p) {
      try {
        await fetch('/api/progress', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(p)
        });
      } catch { /* silent — local copy already written */ }
    },
    clear() { return this.write({ v: 1, done: {}, days: [], updatedAt: 0 }); }
  };

  /* Swap store adapter before any page JS runs */
  function activateVercelStore(store) {
    store.adapter = vercelAdapter;
    /* Re-run init so pages that call store.init() get the remote data */
    store._ready = null;
  }

  /* Render the login/logout button in the top-right header */
  function renderAuthButton(user) {
    const btn = document.getElementById('sa-auth-btn');
    if (!btn) return;
    if (user) {
      btn.innerHTML = `<a href="/api/logout" class="auth-btn auth-btn--out" title="Sign out ${user.login}">
        ${user.avatar ? `<img src="${user.avatar}&s=22" width="22" height="22" alt="" class="auth-avatar">` : ''}
        <span>${user.login}</span>
      </a>`;
    } else {
      btn.innerHTML = `<a href="/api/auth" class="auth-btn auth-btn--in">Sign in</a>`;
    }
  }

  const user = getSaUser();

  /* Swap adapter as early as possible */
  if (user && window.AIS && window.AIS.store) {
    activateVercelStore(window.AIS.store);
  } else if (user) {
    /* store.js not yet loaded — wait for it */
    document.addEventListener('DOMContentLoaded', function () {
      if (window.AIS && window.AIS.store) activateVercelStore(window.AIS.store);
    });
  }

  /* Render button after DOM is ready */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { renderAuthButton(user); });
  } else {
    renderAuthButton(user);
  }

})();
