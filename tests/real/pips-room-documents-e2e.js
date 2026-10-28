// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/pips-room-documents-e2e.js rwkv7-g1d-0.1b.pth some_soul.json
// Documents: a text file you give them is split into passages; a question with its words brings the right passage
// in (keywords, no model time); asking about "the file" brings one in by meaning; an unrelated message brings none;
// "Read it now" feeds it into their living state; the passages travel in the soul file.
const { chromium } = require('playwright'); const path = require('path');
const [MODEL, SOUL] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096'] }); const p = await b.newPage();
  const errs = []; let bad = 0; p.on('pageerror', e => errs.push(e.message));
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) bad++; };
  await p.goto('file://' + path.resolve('pip-suite/pips-room.html'));
  await p.setInputFiles('#import-file-input', SOUL); await p.waitForFunction(() => App.soul);
  await p.evaluate(() => { App.userName = 'Kham'; App.dev.maxTokens = 30; });
  await p.setInputFiles('#file-input', MODEL); await p.waitForFunction(() => App.isAwake, null, { timeout: 600000 }); await p.evaluate(() => clearInterval(App.loopTimer));
  await p.setInputFiles('#doc-file-input', path.resolve('tests/fixtures/frog-pond-notes.txt'));
  await p.waitForFunction(() => document.getElementById('panel-overlay').classList.contains('active'));
  const d = await p.evaluate(() => ({ parts: App.memories.filter(m => m.type === 'document').length, panel: document.getElementById('panel-body').innerText }));
  ok(d.parts >= 3 && /Read it to/.test(d.panel), `the file becomes ${d.parts} passages and can be read in`);
  const t0 = await p.evaluate(() => App.live.tokens);
  await p.click('#panel-body .settings-btn.primary'); await p.waitForFunction(() => !App.loopBusy && App.memories.some(m => m.text === 'Read "frog-pond-notes.txt".'), null, { timeout: 600000 });
  const t1 = await p.evaluate(() => App.live.tokens);
  ok(t1 - t0 > 200, `"Read it now" fed it into their living state (+${t1 - t0} tokens)`);
  await p.evaluate(async () => { while (await indexOneMemory()); });
  const ask = async q => p.evaluate(async q => { const r = await recallFor(q); App.recallRecent = []; return r; }, q);
  const a = await ask('Why are the dogs not allowed near the pond after dark?');
  ok(/Mara once chased a bullfrog/.test(a.lines), `keywords bring the right passage: ${a.notes.join(' · ')}`);
  const c = await ask('What does the file say about baby frogs?');
  console.log('  ' + (c.lines.match(/From the document[^\n]*/) || [''])[0].slice(0, 150));
  ok(/From the document "frog-pond-notes.txt"/.test(c.lines), `asking about "the file" brings a passage in by meaning: ${c.notes.join(' · ')}`);
  const e = await ask('Can you tell me a story about a beetle?');
  ok(!/From the document/.test(e.lines), 'an unrelated message brings no passage');
  const saved = await p.evaluate(() => JSON.parse(JSON.stringify({ memories: App.memories })).memories.filter(m => m.type === 'document').length);
  ok(saved === d.parts, 'the passages are kept with their memories (and so in the soul file)');
  ok(!errs.length, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); console.log(bad ? `${bad} FAILED` : 'all passed'); process.exit(bad ? 1 : 0);
})();
