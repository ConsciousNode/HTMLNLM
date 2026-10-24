// ── Dialogue filter (filter-1): flags lines for a human to look at. It never deletes or edits anything. ──
// Each flag is a reason to look, not a verdict: "stab" in "taking a stab at it" gets flagged too, on purpose.
const FILTER_VERSION = 'filter-1';
const HEAVY_THEMES = {
  violence: ['kill', 'killed', 'killing', 'murder', 'murdered', 'stab', 'stabbed', 'shot', 'shoot', 'shooting', 'gun', 'guns',
             'knife', 'beat', 'beaten', 'kicked', 'punched', 'torture', 'tortured', 'blood', 'bleeding', 'attack', 'attacked', 'assault'],
  abuse:    ['abuse', 'abused', 'abusive', 'molest', 'molested', 'rape', 'raped', 'grooming'],
  selfharm: ['suicide', 'suicidal', 'self-harm', 'overdose'],
  crime:    ['jail', 'prison', 'arrested', 'police', 'cops', 'criminal'],
  drugs:    ['cocaine', 'heroin', 'meth', 'drugs', 'overdosed'],
  sexual:   ['sex', 'sexual', 'naked', 'porn', 'nude'],
  death:    ['dead', 'died', 'death', 'funeral', 'corpse'],
};
const GOODBYE_RE = /\b(good ?bye|bye|see (?:you|ya)|take care|good ?night|farewell|talk (?:to you )?later|gotta go|catch you later)\b/i;
const NONLATIN_RE = /[Ѐ-ӿ֐-ۿऀ-ॿ぀-ヿ㐀-鿿가-힯]/;
const EMOJI_RE = /\p{Extended_Pictographic}|️|‍/gu;

function filterDialogue(dialogue, names = []) {
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const who = ['User', 'System', 'Assistant', 'Human', 'AI', ...names].map(esc).join('|');
  const slipRe = new RegExp(`(^|[\\s"“(])(${who})\\s*(\\([^)]*\\))?:`, 'i');
  const tagRe = /["”]\s*,?\s*(says|said|asks|asked|replies|replied|whispers|whispered|shouts|shouted)\b/i;
  const words = t => (t.toLowerCase().match(/[a-z'’-]+/g) || []);
  const grams = (ws, n) => { const g = []; for (let i = 0; i + n <= ws.length; i++) g.push(ws.slice(i, i + n).join(' ')); return g; };
  const jacc = (a, b) => { const A = new Set(a), B = new Set(b); if (!A.size || !B.size) return 0;
    let inter = 0; for (const x of A) if (B.has(x)) inter++; return inter / (A.size + B.size - inter); };
  const norm = t => t.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();

  const lines = dialogue.map(d => ({ ...d, flags: [] }));
  const add = (l, type, detail) => l.flags.push(detail ? { type, detail } : { type });
  const pips = lines.filter(l => l.role === 'pip');

  pips.forEach((l, idx) => {
    const t = l.text || '', ws = words(t);
    if (!t.trim()) { add(l, 'empty'); return; }
    if (l.gen && l.gen.endedBy === 'cap') add(l, 'hit-cap', `${l.gen.tokens} tokens`);
    else if (!l.gen && ws.length >= 15 && !/[.!?…"”)\]]\s*$/.test(t)) add(l, 'maybe-cut', 'no token count in this export; ends mid-sentence');
    const slip = t.match(slipRe); if (slip) add(l, 'speaker-slip', slip[0].trim());
    if (tagRe.test(t)) add(l, 'dialogue-tag');
    if (/^\s*["“]/.test(t) && /["”]/.test(t.trim().slice(1))) add(l, 'quote-opener');
    const h = t.match(/(^|\s)@[A-Za-z0-9_]{2,}/); if (h) add(l, 'handle', h[0].trim());
    if (/https?:\/\/|www\.|\]\(|!\[|\b[a-z0-9-]+\.(com|org|net|io)\b/i.test(t)) add(l, 'url-or-markup');
    if (NONLATIN_RE.test(t)) add(l, 'script-switch');
    const bare = t.replace(EMOJI_RE, '');
    const ok = (bare.match(/[A-Za-z0-9\s.,!?'’"“”…:;()\-–—]/g) || []).length;
    if (bare.length >= 12 && ok / bare.length < 0.85) add(l, 'garble');
    // repetition: inside the line, against recent lines, against this speaker's own recent lines
    const tri = grams(ws, 3), seen = {};
    for (const g of tri) seen[g] = (seen[g] || 0) + 1;
    const stut = Object.entries(seen).find(([, c]) => c >= 3); if (stut) add(l, 'stutter', stut[0]);
    const recent = pips.slice(Math.max(0, idx - 8), idx);
    if (norm(t).length > 3 && recent.some(r => norm(r.text) === norm(t))) add(l, 'repeat');
    else {
      const own = recent.filter(r => r.speaker === l.speaker).slice(-3);
      const best = Math.max(0, ...own.map(r => jacc(tri, grams(words(r.text), 3))));
      if (tri.length >= 3 && best >= 0.5) add(l, 'near-repeat', best.toFixed(2));
    }
    // heavy themes (flag for review; idioms included on purpose)
    const hits = [];
    for (const [cat, list] of Object.entries(HEAVY_THEMES))
      for (const w of list) if (new RegExp(`\\b${esc(w)}\\b`, 'i').test(t)) { hits.push(`${cat}:${w}`); break; }
    if (hits.length) add(l, 'heavy-theme', hits.join(', '));
  });
  // runs across lines: goodbye spirals and fragment runs
  pips.forEach((l, i) => {
    const win = pips.slice(Math.max(0, i - 3), i + 1);
    if (GOODBYE_RE.test(l.text) && win.filter(x => GOODBYE_RE.test(x.text)).length >= 3)
      for (const x of win) if (GOODBYE_RE.test(x.text) && !x.flags.some(f => f.type === 'goodbye-spiral')) add(x, 'goodbye-spiral');
  });
  // fragment: a line of 4 words or fewer, or a short line trailing off ("Look…", "Well…wait a minute…")
  const isFrag = t => { const n = words(t).length; return n <= 4 || (n <= 8 && /(…|\.\.\.)\s*$/.test(t)); };
  for (let i = 0; i < pips.length; ) {
    let j = i; while (j < pips.length && isFrag(pips[j].text)) j++;
    if (j - i >= 3) for (let k = i; k < j; k++) add(pips[k], 'fragment-run', `${j - i} in a row`);
    i = Math.max(j, i + 1);
  }
  const counts = {};
  for (const l of lines) for (const f of l.flags) counts[f.type] = (counts[f.type] || 0) + 1;
  const flagged = lines.filter(l => l.flags.length).length;
  return {
    lines,
    filter: { version: FILTER_VERSION, verdict: flagged ? 'review' : 'clean', flaggedLines: flagged, totalLines: pips.length,
              counts, note: 'Flags mark lines for a human to look at. Nothing was deleted or changed.',
              themes: Object.keys(HEAVY_THEMES) }
  };
}
if (typeof module !== 'undefined') module.exports = { filterDialogue, FILTER_VERSION };
