/* ============================================================
   AI SCHOOL · game rules (pure)
   Input: phases (curriculum data) + progress (from store).
   Output: a stats object the UI renders. No side effects.
   ============================================================ */
(function (global) {
  'use strict';

  const CORE_LANGS = ['Python', 'TypeScript', 'Rust', 'Julia'];

  const XP_BY_TYPE = { Learn: 60, Build: 120, Capstone: 500 };
  const xpFor = (lesson) => XP_BY_TYPE[lesson.type] || 80;

  /* --- level curve: cost rises each level --- */
  const levelCost = (lvl) => 400 + (lvl - 1) * 150;
  function levelFromXp(xp) {
    let lvl = 1, acc = 0;
    while (xp >= acc + levelCost(lvl)) { acc += levelCost(lvl); lvl++; }
    return { level: lvl, into: xp - acc, span: levelCost(lvl) };
  }

  const RANKS = [
    { min: 1,  title: 'Initiate' },
    { min: 4,  title: 'Practitioner' },
    { min: 7,  title: 'Apprentice Engineer' },
    { min: 10, title: 'AI Engineer' },
    { min: 13, title: 'Senior Engineer' },
    { min: 16, title: 'Staff Engineer' },
    { min: 19, title: 'AI Architect' }
  ];
  const rankFor = (lvl) => RANKS.reduce((r, x) => (lvl >= x.min ? x.title : r), RANKS[0].title);

  /* --- streak from activity days --- */
  function streaks(days) {
    if (!days || !days.length) return { current: 0, best: 0 };
    const set = new Set(days);
    const has = (d) => set.has(d.toISOString().slice(0, 10));
    // current: walk back from today (allow today missing → start yesterday)
    let cur = 0; const c = new Date();
    if (!has(c)) c.setDate(c.getDate() - 1);
    while (has(c)) { cur++; c.setDate(c.getDate() - 1); }
    // best: scan sorted unique days
    const sorted = [...set].sort();
    let best = 0, run = 0, prev = null;
    for (const d of sorted) {
      if (prev && (new Date(d) - new Date(prev)) === 86400000) run++;
      else run = 1;
      best = Math.max(best, run); prev = d;
    }
    return { current: cur, best: Math.max(best, cur) };
  }

  /* --- badges: declarative rules over the computed context --- */
  const BADGES = [
    { id: 'setup',    label: 'Booted Up',      test: (c) => c.phaseDone(0) },
    { id: 'math',     label: 'Mathematician',  test: (c) => c.phaseDone(1) },
    { id: 'model',    label: 'First Model',    test: (c) => c.phaseHasDone(2) },
    { id: 'backprop', label: 'Backprop ×Hand', test: (c) => c.phaseHasDone(3) },
    { id: 'polyglot', label: 'Polyglot ×4',    test: (c) => c.languages >= 4 },
    { id: 'streak7',  label: 'Week Warrior',   test: (c) => c.streak >= 7 },
    { id: 'half',     label: 'Halfway',        test: (c) => c.lessonsDone >= c.lessonsTotal / 2 },
    { id: 'agent',    label: 'Agent Pilot',    test: (c) => c.phaseHasDone(14) },
    { id: 'boss',     label: 'Boss Slayer',    test: (c) => c.capstonesDone >= 1 }
  ];

  function derive(phases, progress) {
    const done = progress.done || {};
    const isDone = (pid, i) => !!done[pid + ':' + i];

    let lessonsDone = 0, lessonsTotal = 0, xp = 0, builds = 0, capstonesDone = 0;
    const langSet = new Set();
    const phaseStats = [];

    phases.forEach((ph) => {
      let d = 0;
      ph.lessons.forEach((ls, i) => {
        lessonsTotal++;
        if (isDone(ph.id, i)) {
          d++; lessonsDone++;
          xp += xpFor(ls);
          if (ls.type === 'Build') builds++;
          if (ls.type === 'Capstone') capstonesDone++;
          String(ls.lang || '').split(',').map((s) => s.trim())
            .forEach((l) => { if (CORE_LANGS.includes(l)) langSet.add(l); });
        }
      });
      const total = ph.lessons.length;
      const pct = total ? Math.round((d / total) * 100) : 0;
      phaseStats.push({ id: ph.id, name: ph.name, desc: ph.desc, done: d, total, pct });
    });

    // active = first phase not fully done
    const activeIdx = phaseStats.findIndex((p) => p.pct < 100);
    phaseStats.forEach((p, i) => {
      p.status = p.pct === 100 ? 'done' : i === activeIdx ? 'active' : p.pct > 0 ? 'active' : 'upcoming';
    });

    const phasesCleared = phaseStats.filter((p) => p.pct === 100).length;
    const lvl = levelFromXp(xp);
    const st = streaks(progress.days);

    const ctx = {
      lessonsDone, lessonsTotal, languages: langSet.size, streak: st.current, capstonesDone,
      phaseDone: (id) => { const p = phaseStats.find((x) => x.id === id); return p && p.pct === 100; },
      phaseHasDone: (id) => { const p = phaseStats.find((x) => x.id === id); return p && p.done > 0; }
    };
    const badges = BADGES.map((b) => ({ id: b.id, label: b.label, got: !!b.test(ctx) }));

    return {
      lessonsDone, lessonsTotal,
      phasesCleared, phasesTotal: phases.length,
      languages: langSet.size, languagesTotal: CORE_LANGS.length,
      buildArtifacts: builds, capstonesDone,
      totalXp: xp,
      level: lvl.level, intoLevel: lvl.into, levelSpan: lvl.span,
      pctIntoLevel: Math.round((lvl.into / lvl.span) * 100),
      toNext: lvl.span - lvl.into,
      rank: rankFor(lvl.level),
      streak: st.current, bestStreak: st.best,
      badges, phaseStats
    };
  }

  (global.AIS = global.AIS || {}).game = { derive, xpFor };
})(window);
