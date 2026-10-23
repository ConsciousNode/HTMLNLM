// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/pips-room-think-e2e.js rwkv7-g1d-0.1b.pth
// "Think before answering": the resident reasons inside <think>…</think> (into their state and the
// journal), then answers on one line. Prints the reasoning, the answer and what was fed.
const { chromium } = require('playwright'); const path = require('path');
const MODEL = process.argv[2];
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096'] }); const errs = [];
  const p = await b.newPage(); p.on('pageerror', e => errs.push(e.message));
  await p.goto('file://' + path.resolve('pip-suite/pips-room.html'));
  await p.fill('#pip-name', 'Wren');
  await p.setInputFiles('#file-input', MODEL);
  await p.waitForFunction(() => App.isAwake, null, { timeout: 600000 });
  await p.evaluate(() => { clearInterval(App.loopTimer); App.userName = 'Kham'; App.dev.think = true; App.privacy.innerVoice = true; });
  for (const q of ['There are 3 frogs on a log and 2 more hop on. How many frogs are on the log now?', 'Why do frogs sing at night?']) {
    const n = await p.$$eval('.message.pip', e => e.length), t = Date.now();
    await p.fill('#message-input', q); await p.press('#message-input', 'Enter');
    await p.waitForFunction(k => document.querySelectorAll('.message.pip').length > k && !App.isGenerating, n, { timeout: 600000 });
    const ans = (await p.$$eval('.message.pip', e => e.at(-1).querySelector('div:nth-child(2)').innerText)).trim();
    const thought = await p.evaluate(() => App.memories.filter(m => m.type === 'reasoning').at(-1)?.text);
    console.log(`\nKham: ${q}\n  🤔 ${JSON.stringify(thought)}\n  Wren: ${ans}   (${((Date.now() - t) / 1000).toFixed(0)}s)`);
    console.log('  fed: ' + JSON.stringify(await p.evaluate(() => App.lastPrompt.split('\n').slice(1).join('\n'))));
  }
  console.log('\njournal 🤔 count:', await p.evaluate(() => App.memories.filter(m => m.type === 'reasoning').length));
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
