// Needs Playwright. Run from the repo root: node tests/real/pips-room-repeat-guard.js
// No model needed: a stand-in model and tokenizer make the resident get stuck on purpose
// ("Fazzani is setting a goal (" over and over), in a plain turn and in a thinking turn.
// Checks: the turn stops after the first copy, and the state is rolled back to before the repeats
// (the stand-in state counts forward steps, so it must equal the token count after a rollback).
const { chromium } = require('playwright'); const path = require('path');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage(); const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  await p.goto('file://' + path.resolve('pip-suite/pips-room.html'));
  const r = await p.evaluate(async () => {
    // stand-ins: token i decodes to WORDS[i]; the model's "state" is just a counter object
    const WORDS = ['', ' Fazzani', ' is', ' setting', ' a', ' goal', ' (', ' hi', ' there', '.', '\n', '</think>', ' Hello', ' friend'];
    App.tokenizer = { decode: ids => ids.map(i => WORDS[i] ?? '').join(''), encode: t => t.split('').map(() => 7) };
    const states = [];
    App.model = { forward: (id, st) => { const ns = { n: st.n + 1, id }; states.push(ns); return { logits: new Float32Array(WORDS.length), state: ns }; },
                  getEmptyState: () => ({ n: 0 }), L: 1, C: 1, N: 1, V: WORDS.length };
    App.kind = 'rwkv7'; App.dev.livingMemory = true; App.soul = App.soul || { name: 'Fazzani', traits: {}, values: [], goals: [] };
    App.soul.name = 'Fazzani'; App.userName = 'Kham';
    App.live = { state: { n: 0 }, tokens: 0, print: modelPrint(), sig: soulSig(), lastLogits: new Float32Array(WORDS.length) };
    let script = [];
    window.sampleLogits = () => script.length ? script.shift() : 0;
    const out = {};
    // 1) plain turn: "Hello friend Fazzani is setting a goal ( ×4 …"
    const cycle = [1, 2, 3, 4, 5, 6];
    script = [12, 13, ...cycle, ...cycle, ...cycle, ...cycle, 10];
    let before = App.live.tokens;
    const t1 = await liveTurn('Kham: hi\nFazzani:', 80, 0.8);
    out.plain = { text: t1.text, tokensKept: App.live.tokens - before, stateMatchesCount: App.live.state.n === App.live.tokens, cuts: App.repeatCuts };
    // 2) thinking turn: the thought gets stuck the same way
    App.dev.think = true;
    script = [7, 8, ...cycle, ...cycle, ...cycle, ...cycle, 12, 13, 10, 10];
    const t2 = await liveTurn('\nUser: hi (think)\n\nAssistant:', 40, 0.8, { think: true, endOn: '\n\n' });
    out.think = { thought: t2.thought, answer: t2.text, cuts: App.repeatCuts };
    // 3) a line that repeats only twice is left alone
    App.dev.think = false;
    script = [12, 13, ...cycle, ...cycle, 9, 10];
    const t3 = await liveTurn('Kham: hi\nFazzani:', 80, 0.8);
    out.twice = { text: t3.text, cuts: App.repeatCuts };
    return out;
  });
  console.log(JSON.stringify(r, null, 1));
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
  await b.close();
})();
