/**
 * Local-only progress tracker.
 *
 * Stores everything in the user's own browser (localStorage). No network,
 * no account, no server. Data never leaves the device.
 *
 * Schema (versioned so we can migrate later without nuking users):
 *
 *   aifs:progress:v1 = {
 *     lessons: {
 *       "<lesson-path>": {
 *         answers: { "<qid>": { picked: number, correct: boolean, t: number } },
 *         completedAt: number | null,
 *         visitedAt: number
 *       }
 *     },
 *     updatedAt: number
 *   }
 *
 * "<lesson-path>" matches the path used in lesson.html?path=... and in
 * data.js urls (e.g. "phases/00-setup-and-tooling/01-dev-environment").
 *
 * "<qid>" is "<stage>-q<index>" e.g. "pre-q0", to match the quiz renderer.
 */
(function () {
  var STORAGE_KEY = 'aifs:progress:v1';
  var listeners = [];

  function emptyState() {
    return { lessons: {}, updatedAt: 0 };
  }

  function read() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return emptyState();
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object' || !parsed.lessons) return emptyState();
      return parsed;
    } catch (e) {
      return emptyState();
    }
  }

  function write(state) {
    state.updatedAt = Date.now();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      // quota or disabled storage; fail silently
    }
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](state); } catch (_) {}
    }
  }

  function ensureLesson(state, path) {
    if (!state.lessons[path]) {
      state.lessons[path] = { answers: {}, completedAt: null, visitedAt: 0 };
    }
    return state.lessons[path];
  }

  function recordVisit(path) {
    if (!path) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    lesson.visitedAt = Date.now();
    write(state);
  }

  function recordAnswer(path, qid, picked, correct) {
    if (!path || !qid) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    lesson.answers[qid] = { picked: picked, correct: !!correct, t: Date.now() };
    write(state);
  }

  function markLessonComplete(path) {
    if (!path) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    if (!lesson.completedAt) {
      lesson.completedAt = Date.now();
      write(state);
    }
  }

  function unmarkLessonComplete(path) {
    if (!path) return;
    var state = read();
    if (state.lessons[path] && state.lessons[path].completedAt) {
      state.lessons[path].completedAt = null;
      write(state);
    }
  }

  function getLessonProgress(path) {
    if (!path) return null;
    var state = read();
    return state.lessons[path] || { answers: {}, completedAt: null, visitedAt: 0 };
  }

  function isLessonComplete(path) {
    var lp = getLessonProgress(path);
    return !!(lp && lp.completedAt);
  }

  /**
   * Given a list of lesson urls (full GitHub urls from data.js), count how
   * many the user has completed. Match by the trailing "phases/.../..." path.
   */
  function countCompletedFromUrls(urls) {
    var state = read();
    var n = 0;
    for (var i = 0; i < urls.length; i++) {
      var path = extractPath(urls[i]);
      if (path && state.lessons[path] && state.lessons[path].completedAt) n++;
    }
    return n;
  }

  function extractPath(url) {
    if (!url) return '';
    var m = String(url).match(/(phases\/[^/]+\/[^/]+)\/?/);
    return m ? m[1] : '';
  }

  function totalCompleted() {
    var state = read();
    var n = 0;
    for (var k in state.lessons) {
      if (state.lessons[k].completedAt) n++;
    }
    return n;
  }

  function reset() {
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](emptyState()); } catch (_) {}
    }
  }

  function onChange(fn) {
    if (typeof fn === 'function') listeners.push(fn);
  }

  // Cross-tab sync: if user clears or updates progress in another tab,
  // refresh listeners here too.
  window.addEventListener('storage', function (e) {
    if (e.key !== STORAGE_KEY) return;
    var state = read();
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](state); } catch (_) {}
    }
  });

   function exportProgress() {
     var state = read();
     var blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
     var url = URL.createObjectURL(blob);
     var a = document.createElement('a');
     a.href = url;
     a.download = 'aifs-progress.json';
     document.body.appendChild(a);
     a.click();
     document.body.removeChild(a);
     URL.revokeObjectURL(url);
   }

   function importProgress(file, callback) {
     if (!file) return;
     var reader = new FileReader();
     reader.onload = function (e) {
       try {
         var imported = JSON.parse(e.target.result);
         if (!imported || typeof imported !== 'object' || !imported.lessons) {
           if (callback) callback(new Error('Invalid progress file: missing lessons object'));
           return;
         }
         // Basic validation: lessons must be an object with path keys
         for (var k in imported.lessons) {
           var lesson = imported.lessons[k];
           if (!lesson || typeof lesson !== 'object') {
             if (callback) callback(new Error('Invalid progress file: malformed lesson data'));
             return;
           }
         }
         // Write to localStorage
         write(imported);
         if (callback) callback(null);
       } catch (err) {
         if (callback) callback(err);
       }
     };
     reader.onerror = function () {
       if (callback) callback(new Error('Failed to read file'));
     };
     reader.readAsText(file);
   }

   window.AIFSProgress = {
     recordVisit: recordVisit,
     recordAnswer: recordAnswer,
     markLessonComplete: markLessonComplete,
     unmarkLessonComplete: unmarkLessonComplete,
     getLessonProgress: getLessonProgress,
     isLessonComplete: isLessonComplete,
     countCompletedFromUrls: countCompletedFromUrls,
     extractPath: extractPath,
     totalCompleted: totalCompleted,
     reset: reset,
     onChange: onChange,
     exportProgress: exportProgress,
     importProgress: importProgress,
   };

  // Auto-bind header Export/Import buttons on all pages
  document.addEventListener('DOMContentLoaded', function () {
    var headerExportBtn = document.getElementById('headerExportBtn');
    var headerImportBtn = document.getElementById('headerImportBtn');
    
    if (!headerExportBtn && !headerImportBtn) return;

    // Create hidden file input for import if it doesn't exist
    var importFile = document.getElementById('importFile');
    if (!importFile && headerImportBtn) {
      importFile = document.createElement('input');
      importFile.type = 'file';
      importFile.id = 'importFile';
      importFile.accept = 'application/json';
      importFile.style.display = 'none';
      document.body.appendChild(importFile);
    }

    // Bind export button
    if (headerExportBtn) {
      headerExportBtn.addEventListener('click', function () {
        try {
          window.AIFSProgress.exportProgress();
        } catch (e) {
          console.error('Export failed:', e);
          alert('Failed to export progress. Please try again.');
        }
      });
    }

    // Bind import button
    if (headerImportBtn && importFile) {
      headerImportBtn.addEventListener('click', function () {
        importFile.click();
      });

      importFile.addEventListener('change', function (e) {
        var file = e.target.files[0];
        if (!file) return;

        if (!window.confirm('Importing will replace all your current progress. Continue?')) {
          importFile.value = '';
          return;
        }

        window.AIFSProgress.importProgress(file, function (err) {
          if (err) {
            console.error('Import failed:', err);
            alert('Failed to import progress: ' + (err.message || err));
            importFile.value = '';
          } else {
            alert('Progress imported successfully!');
            importFile.value = '';
          }
        });
      });
    }
  });
 })();
