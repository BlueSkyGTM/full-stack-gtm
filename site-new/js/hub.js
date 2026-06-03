/* ============================================================
   AI SCHOOL · hub
   Home page — hero + signposts only. No world map.
   store.seedIfEmpty ensures demo progress is available if
   localStorage is empty (useful for first open).
   ============================================================ */
(function () {
  'use strict';
  window.AIS.store.seedIfEmpty(PHASES);
})();
