// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/pips-room-recall-e2e.js rwkv7-g1d-0.1b.pth some_soul.json
// Recall (the dual memory index). Checks: idle indexing gives every exchange and summary a vector, and a
// message you send stops it at once; asking them to remember brings back the right exchange word for word
// (it shows in the prompt and under the reply); an ordinary question with no cue costs no model time and
// brings back no exchange; a new summary links to the exchanges it covers; the vectors travel in the soul
// file and a fresh page doesn't index again.
const { chromium } = require('playwright'); const path = require('path'); const fs = require('fs'); const os = require('os');
const [MODEL, SOUL] = process.argv.slice(2);
const page = 'file://' + path.resolve('pip-suite/pips-room.html');
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096'] }); const errs = []; let bad = 0;
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) bad++; };
  const ctx = await b.newContext({ acceptDownloads: true });
  let p = await ctx.newPage(); p.on('pageerror', e => errs.push(e.message));
  const open = async soul => {
    await p.goto(page);
    await p.setInputFiles('#import-file-input', soul);
    await p.waitForFunction(() => App.soul, null, { timeout: 30000 });
    await p.evaluate(() => { App.userName = 'Kham'; App.dev.livingMemory = false; App.dev.maxTokens = 24; });
    await p.setInputFiles('#file-input', MODEL);
    await p.waitForFunction(() => App.isAwake, null, { timeout: 600000 });
    await p.evaluate(() => { clearInterval(App.loopTimer); });
  };
  let SOUL2 = null;
  await open(SOUL);

  // indexing stops the moment you speak
  const stopped = await p.evaluate(async () => {
    const t = indexOneMemory(); await new Promise(r => setTimeout(r, 300)); App.isGenerating = true;
    const r = await t; App.isGenerating = false; return r;
  });
  ok(stopped === false, 'idle indexing gives up as soon as a message is being answered');

  let t0 = Date.now();
  const total = await p.evaluate(() => recallable().length);
  await p.evaluate(async () => { while (await indexOneMemory()); });
  const secs = (Date.now() - t0) / 1000;
  const info = await p.evaluate(() => document.getElementById('recall-info').textContent);
  console.log(`indexed ${total} memories in ${secs.toFixed(0)} s (${(secs / total).toFixed(1)} s each) · "${info}"`);
  ok(info.startsWith(`${total} of ${total}`), 'every exchange and summary has a vector');

  const say = async msg => {
    const n = await p.$$eval('.message.pip', e => e.length);
    await p.fill('#message-input', msg); await p.press('#message-input', 'Enter');
    await p.waitForFunction(k => document.querySelectorAll('.message.pip').length > k && !App.isGenerating, n, { timeout: 300000 });
    return p.evaluate(() => ({ prompt: App.lastPrompt, note: [...document.querySelectorAll('.message.pip')].at(-1).querySelector('.message-recall')?.textContent || '',
                               reply: [...document.querySelectorAll('.message.pip')].at(-1).children[1].innerText, ms: App.lastRecallMs }));
  };
  const cases = [
    ['Do you remember how many pets I have?', /4 cats and 2 dogs/],
    ['Do you remember which town I live in?', /Ninety Six/],
    ['Remember which piece I was in our monopoly game?', /battleship|thimble/],
    ['What was your favorite word again?', /favorite word/],
  ];
  let hits = 0;
  for (const [q, want] of cases) {
    t0 = Date.now(); const r = await say(q);
    const line = (r.prompt.match(/System: \S+ remembers this exchange[^\n]*/) || [''])[0];
    const hit = want.test(line); hits += hit;
    console.log(`  ${hit ? '✓' : '✗'} "${q}" (turn ${((Date.now() - t0) / 1000).toFixed(0)} s, recall ${(r.ms / 1000).toFixed(1)} s)\n      ${line.slice(0, 160) || '(nothing recalled)'}\n      note: ${r.note.slice(0, 120)}\n      reply: ${r.reply.slice(0, 100)}`);
  }
  ok(hits >= 3, `asked to remember, the right exchange comes back word for word (${hits}/${cases.length})`);
  const plain = await say('What is the capital of Peru?');
  ok(!/remembers this exchange/.test(plain.prompt) && plain.ms < 1000, `an ordinary question with no cue brings back no exchange and costs no model time (${plain.ms} ms)`);
  const n1 = await p.$$eval('.message-recall', e => e.length);
  ok(n1 >= 3, 'what was remembered is shown under the reply');

  // a new summary links back to the exchanges it covers
  const linked = await p.evaluate(async () => { await loopConsolidate(); return App.consolidateFrom || []; });
  ok(linked.length > 0 && linked.every(k => k.startsWith('conversation:')), `a new summary links to the ${linked.length} exchanges it covers`);

  // vectors travel in the soul file
  await p.evaluate(async () => { while (await indexOneMemory()); });
  await p.evaluate(() => document.getElementById('settings-panel').classList.add('open'));
  const [dl] = await Promise.all([p.waitForEvent('download', { timeout: 120000 }), p.click('#export-btn')]);
  SOUL2 = path.join(os.tmpdir(), 'recall-soul.json'); await dl.saveAs(SOUL2);
  const saved = JSON.parse(fs.readFileSync(SOUL2, 'utf8'));
  ok(saved.recall && Object.keys(saved.recall.vecs).length >= total, `the soul file carries ${Object.keys(saved.recall?.vecs || {}).length} vectors (${(JSON.stringify(saved.recall).length / 1024).toFixed(0)} KB)`);
  await p.close(); p = await ctx.newPage(); p.on('pageerror', e => errs.push(e.message));
  await open(SOUL2);
  const info2 = await p.evaluate(() => { updateRecallInfo(); return document.getElementById('recall-info').textContent; });
  ok(/^(\d+) of \1/.test(info2), `a fresh page doesn't index again ("${info2}")`);
  ok(!errs.length, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); console.log(bad ? `${bad} FAILED` : 'all passed'); process.exit(bad ? 1 : 0);
})();
