/* ============================================================
   AI SCHOOL · ui helpers
   Minimal DOM builder + toast. No framework, no bloat.
   ============================================================ */
(function (global) {
  'use strict';

  /** el('div', {class:'x', onClick:fn, html:'…'}, [child, 'text']) */
  function el(tag, props, children) {
    const node = document.createElement(tag);
    if (props) {
      for (const k in props) {
        const v = props[k];
        if (k === 'class') node.className = v;
        else if (k === 'html') node.innerHTML = v;
        else if (k === 'style') node.style.cssText = v;
        else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2).toLowerCase(), v);
        else if (v != null && v !== false) node.setAttribute(k, v === true ? '' : v);
      }
    }
    (Array.isArray(children) ? children : children != null ? [children] : [])
      .forEach((c) => node.append(c instanceof Node ? c : document.createTextNode(c)));
    return node;
  }

  function host() {
    let h = document.querySelector('.toast-host');
    if (!h) { h = el('div', { class: 'toast-host' }); document.body.append(h); }
    return h;
  }

  function toast(big, sub) {
    const t = el('div', { class: 'toast' }, [
      el('span', { class: 'toast__big' }, big),
      el('span', { class: 'toast__sub' }, sub)
    ]);
    host().append(t);
    setTimeout(() => t.remove(), 2800);
  }

  (global.AIS = global.AIS || {}).ui = { el, toast };
})(window);
