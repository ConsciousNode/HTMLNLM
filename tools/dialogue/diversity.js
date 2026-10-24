// ── Diversity check (diversity-1): how alike are a pile of dialogue exports? ──
// Reports, it doesn't judge: the numbers say where runs repeat each other so a human can decide.
const DIVERSITY_VERSION = 'diversity-1';
const STOP = new Set(('i me my you your we our they them their he she him her it its a an the and or but if so to of in on at for with ' +
  'is are was were be been am do did does have has had not no yes just that this what who how why when where there here then ' +
  'can could would should will about from as by up out all some any more very really like know think okay ok well oh').split(' '));
const COMMON_CAPS = new Set(('I The A An And But So Yeah Yes No Well Oh Okay Ok Hey Hi What Who How Why When Where That This It ' +
  'We You They He She Maybe Now Look Right Sure Of Is Are Do Did Can Let Just If Or Not Then There Here My Your Our Their ' +
  'Wait Like Thank Thanks Please Good Great Also Actually Anyway Because Since').split(' '));
const dWords = t => (t.toLowerCase().match(/[a-z'’]+/g) || []);
const dGrams = (ws, n) => { const g = []; for (let i = 0; i + n <= ws.length; i++) g.push(ws.slice(i, i + n).join(' ')); return g; };
const dJacc = (A, B) => { if (!A.size || !B.size) return 0; let i = 0; for (const x of A) if (B.has(x)) i++; return i / (A.size + B.size - i); };

function diversityReport(runs) {   // runs: [{ label, data }] where data is a dialogue export
  const R = runs.map(({ label, data }) => {
    const pips = (data.pips || []).map(p => p.name);
    const lines = (data.dialogue || []).filter(d => d.role === 'pip');
    const ws = lines.flatMap(l => dWords(l.text));
    const content = new Set(ws.filter(w => !STOP.has(w) && w.length > 2));
    const tri = new Set(lines.flatMap(l => dGrams(dWords(l.text), 3)));
    const four = new Set(lines.flatMap(l => dGrams(dWords(l.text), 4)));
    const names = new Set();
    for (const l of lines) for (const m of (l.text.match(/(?<![.!?…]\s|^)\b[A-Z][a-z]{2,}\b/g) || []))
      if (!COMMON_CAPS.has(m) && !pips.includes(m)) names.add(m);
    const s = data.run && data.run.segments && data.run.segments[0];
    return { label, data, lines, ws, content, tri, four, names, pips,
      opener: (data.dialogue || []).find(d => d.role === 'opener' || d.role === 'topic' || d.role === 'pip'),
      firstPip: lines[0] ? lines[0].text : '',
      meta: { seed: data.run && data.run.seed, mode: s && s.settings && s.settings.mode, deck: data.deck ? data.deck.name : null,
              temperature: s && s.settings && s.settings.temperature, model: (data.pips || [])[0] && data.pips[0].fingerprint,
              turns: lines.length, words: ws.length } };
  });
  // whole-pile distinct-n (unique n-grams / total n-grams), a standard diversity number
  const all = R.flatMap(r => r.lines.map(l => dWords(l.text)));
  const distinct = n => { const g = all.flatMap(w => dGrams(w, n)); return g.length ? +(new Set(g).size / g.length).toFixed(3) : 0; };
  // pairwise overlap
  const pairs = [];
  for (let i = 0; i < R.length; i++) for (let j = i + 1; j < R.length; j++)
    pairs.push({ a: R[i].label, b: R[j].label, contentWords: +dJacc(R[i].content, R[j].content).toFixed(3),
                 trigrams: +dJacc(R[i].tri, R[j].tri).toFixed(3) });
  // shared openings first: runs that replay the same lines (same seed and settings) would swamp everything else
  const sharedOpenings = [], skip = R.map(() => 0);
  for (let i = 0; i < R.length; i++) for (let j = i + 1; j < R.length; j++) {
    let k = 0; while (k < R[i].lines.length && k < R[j].lines.length && R[i].lines[k].text === R[j].lines[k].text) k++;
    if (k) { sharedOpenings.push({ a: R[i].label, b: R[j].label, identicalLines: k }); skip[i] = Math.max(skip[i], k); skip[j] = Math.max(skip[j], k); }
  }
  // phrases (4-grams) that turn up in more than one run, counted after any replayed opening
  const phraseRuns = new Map();
  R.forEach((r, i) => { const g4 = new Set(r.lines.slice(skip[i]).flatMap(l => dGrams(dWords(l.text), 4)));
    for (const g of g4) { if (!phraseRuns.has(g)) phraseRuns.set(g, new Set()); phraseRuns.get(g).add(i); } });
  const sharedPhrases = [...phraseRuns].filter(([, s]) => s.size > 1).map(([g, s]) => ({ phrase: g, runs: s.size }))
    .sort((x, y) => y.runs - x.runs || x.phrase.localeCompare(y.phrase)).slice(0, 25);
  // names that show up across runs
  const nameRuns = new Map();
  R.forEach(r => { for (const n of r.names) nameRuns.set(n, (nameRuns.get(n) || 0) + 1); });
  const recurringNames = [...nameRuns].filter(([, c]) => c > 1).map(([name, runs]) => ({ name, runs })).sort((a, b) => b.runs - a.runs);
  // same first generated line
  const firstMap = new Map();
  R.forEach(r => { const k = r.firstPip.trim().toLowerCase(); if (!k) return; firstMap.set(k, [...(firstMap.get(k) || []), r.label]); });
  const sameFirstLine = [...firstMap].filter(([, ls]) => ls.length > 1).map(([line, labels]) => ({ line, runs: labels }));
  const meanPair = key => pairs.length ? +(pairs.reduce((s, p) => s + p[key], 0) / pairs.length).toFixed(3) : null;
  return {
    version: DIVERSITY_VERSION, generatedAt: new Date().toISOString(), runCount: R.length,
    runs: R.map(r => ({ label: r.label, ...r.meta, inventedNames: [...r.names].sort() })),
    pile: { distinct1: distinct(1), distinct2: distinct(2), distinct3: distinct(3),
            meanContentOverlap: meanPair('contentWords'), meanTrigramOverlap: meanPair('trigrams') },
    pairs, sharedOpenings, sameFirstLine, sharedPhrases, recurringNames,
    note: 'Shared phrases are counted after any opening lines two runs replay identically. Higher distinct-n and lower overlap mean more varied runs. Shared openings between runs with the same seed and settings are expected (they replay until something differs).'
  };
}
if (typeof module !== 'undefined') module.exports = { diversityReport, DIVERSITY_VERSION };
