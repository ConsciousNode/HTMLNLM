// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/pips-room-loops-e2e.js rwkv7-g1d-0.1b.pth
// Wakes Pip's Room with a real RWKV-7 checkpoint, chats once, fast-forwards all seven
// autonomous loops (inner voice kept private), exports her soul, reloads the page, imports
// the soul BEFORE loading the model, and checks she wakes up as herself with her room restored.
const { chromium } = require('playwright'); const path = require('path'); const fs = require('fs'); const os = require('os');
const MODEL = process.argv[2];
const page = 'file://' + path.resolve('pip-suite/pips-room.html');
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096'] });
  const errs = [];
  let p = await b.newPage(); p.on('pageerror', e => errs.push(e.message));
  await p.goto(page);
  await p.fill('#pip-name', 'Fazzani').catch(() => {});
  await p.setInputFiles('#file-input', MODEL);
  await p.waitForFunction(() => typeof App !== 'undefined' && App.isAwake, null, { timeout: 300000 });
  await p.fill('#message-input', 'Do you like frogs?'); await p.press('#message-input', 'Enter');
  await p.waitForFunction(() => document.querySelectorAll('.message.pip').length > 0 && !App.isGenerating, null, { timeout: 300000 });
  console.log('chat:', (await p.$$eval('.message.pip', els => els.at(-1).innerText)).replace(/\s+/g, ' '));

  // fast-forward: every loop due now, inner voice NOT shared (should stay private)
  await p.evaluate(() => { App.privacy.innerVoice = false; Object.keys(LOOPS).forEach(k => { App.loopLast[k] = 0; }); });
  for (let i = 0; i < 7; i++) {
    await p.evaluate(() => runDueLoop());
    await p.waitForFunction(() => !App.loopBusy, null, { timeout: 300000 });
  }
  const life = await p.$$eval('.inner-life-entry', els => els.map(e => e.className.replace('inner-life-entry ', '') + ' | ' + e.querySelector('.inner-life-text').innerText));
  console.log('\ninner life (newest first):\n  ' + life.join('\n  '));
  const state = await p.evaluate(() => ({ goals: App.soul.goals, mems: App.memories.map(m => m.type + (m.shared === false ? '(private)' : '')), consolidated: !!App.lastConsolidated }));
  console.log('\ngoals:', JSON.stringify(state.goals), '\nmemories:', state.mems.join(', '), '\nlastConsolidated set:', state.consolidated);

  // export exactly what the Export button exports, then start over in a fresh page
  const soul = await p.evaluate(() => JSON.stringify({ soul: App.soul, memories: App.memories, conversationCount: App.conversationCount,
    tokensGenerated: App.tokensGenerated, genesisDate: App.genesisDate, exportedAt: Date.now() }));
  const soulFile = path.join(os.tmpdir(), 'fazzani_soul.json'); fs.writeFileSync(soulFile, soul);
  await p.close();
  p = await b.newPage(); p.on('pageerror', e => errs.push(e.message));
  await p.goto(page);
  await p.setInputFiles('#import-file-input', soulFile);           // soul first...
  await p.waitForFunction(() => App.soul && App.soul.name === 'Fazzani', null, { timeout: 30000 });
  await p.setInputFiles('#file-input', MODEL);                     // ...then the model
  await p.waitForFunction(() => App.isAwake, null, { timeout: 300000 });
  const back = await p.evaluate(() => ({ name: App.soul.name, mems: App.memories.length,
    msgs: document.querySelectorAll('.message').length, life: document.querySelectorAll('.inner-life-entry').length,
    timers: App.loopTimer ? 1 : 0 }));
  console.log('\nafter reload + import-then-load:', JSON.stringify(back));
  console.log(back.name === 'Fazzani' && back.msgs >= 2 && back.life >= 1 ? 'RESTORED OK' : 'RESTORE FAILED');
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
