// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/pips-room-e2e.js rwkv7-g1d-0.1b.pth
// Headless Chromium: wake Pip's Room with the real RWKV-7 0.1B checkpoint and talk to her.
const { chromium } = require('playwright'); const path = require('path');
const MODEL = process.argv[2];
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096'] });
  const p = await b.newPage(); const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  await p.goto('file://' + path.resolve('pip-suite/pips-room.html'));
  await p.fill('#pip-name', 'Goose').catch(() => {});
  const t0 = Date.now();
  await p.setInputFiles('#file-input', MODEL);
  await p.waitForFunction(() => typeof App !== 'undefined' && App.isAwake, null, { timeout: 300000 });
  console.log(`awake in ${((Date.now() - t0) / 1000).toFixed(1)}s`,
    'arch:', await p.textContent('#setting-arch'), '| params:', await p.textContent('#setting-params'),
    '| vocab:', await p.textContent('#setting-vocab'), '| name:', await p.evaluate(() => App.soul.name));
  // keep the inner-voice / dream loops out of the way for a clean measurement
  await p.evaluate(() => { App.privacy.innerVoice = false; App.privacy.dreams = false; });
  for (const msg of ['Hello! What do frogs say at night?', 'Do you like the sound?', 'What is your name?']) {
    const before = await p.$$eval('.message.pip', els => els.length);
    await p.fill('#message-input', msg); await p.press('#message-input', 'Enter');
    const t1 = Date.now();
    await p.waitForFunction(n => document.querySelectorAll('.message.pip').length > n && !App.isGenerating, before, { timeout: 300000 });
    const reply = await p.$$eval('.message.pip', els => els[els.length - 1].innerText);
    console.log(`\nUser: ${msg}\n${reply.replace(/\s+/g, ' ').trim()}   (${((Date.now() - t1) / 1000).toFixed(0)}s)`);
  }
  console.log('\nprompt for the last turn:\n' + await p.evaluate(() => {
    const n = App.soul.name, rec = App.memories.filter(m => m.type === 'conversation').slice(-3);
    return rec.map(m => `User: ${m.user}\n${n}: ${m.response}`).join('\n');
  }));
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : '\nno page errors');
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
