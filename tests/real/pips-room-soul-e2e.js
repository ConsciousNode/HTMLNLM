// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/pips-room-soul-e2e.js rwkv7-g1d-0.1b.pth some_soul.json [screenshot-prefix]
// Imports a soul, wakes the real model, sets your name + an About note + developer-console knobs,
// chats, runs two loops, checks the system prompt the model saw, that traits/mood moved, and that
// the real Export button's file brings all of it back in a fresh page.
const { chromium } = require('playwright'); const path = require('path'); const fs = require('fs'); const os = require('os');
const [MODEL, SOUL, SHOT] = process.argv.slice(2);
const page = 'file://' + path.resolve('pip-suite/pips-room.html');
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096'] }); const errs = [];
  const ctx = await b.newContext({ acceptDownloads: true, viewport: { width: 412, height: 860 }, deviceScaleFactor: 2 });
  let p = await ctx.newPage(); p.on('pageerror', e => errs.push(e.message));
  await p.goto(page);
  await p.setInputFiles('#import-file-input', SOUL);
  await p.waitForFunction(() => App.soul, null, { timeout: 30000 });
  await p.setInputFiles('#file-input', MODEL);
  await p.waitForFunction(() => App.isAwake, null, { timeout: 300000 });
  await p.evaluate(() => { App.privacy.innerVoice = true; App.privacy.dreams = true; App.loopTimer && clearInterval(App.loopTimer); });

  // settings: rename, your name, about, dev knobs (through the real inputs)
  await p.evaluate(() => document.getElementById('settings-panel').classList.add('open'));
  await p.fill('#pip-name', 'Fazzani'); await p.press('#pip-name', 'Tab');
  await p.fill('#user-name', 'Kham'); await p.press('#user-name', 'Tab');
  await p.fill('#about-note', 'They live in a little room on a phone, and they love frogs.'); await p.press('#about-note', 'Tab');
  await p.click('#dev-toggle');
  await p.fill('#dev-fields input[data-key="temperature"]', '0.7'); await p.press('#dev-fields input[data-key="temperature"]', 'Tab');
  await p.fill('#dev-fields input[data-key="topK"]', '40'); await p.press('#dev-fields input[data-key="topK"]', 'Tab');
  const before = await p.evaluate(() => ({ traits: { ...App.soul.traits }, mood: App.soul.mood }));

  await p.evaluate(() => document.getElementById('settings-panel').classList.remove('open'));
  for (const msg of ['Hi Fazzani! It is Kham. How are you feeling today?', 'I am so glad to see you again! Do you remember the frogs?']) {
    const n = await p.$$eval('.message.pip', e => e.length);
    await p.fill('#message-input', msg); await p.press('#message-input', 'Enter');
    await p.waitForFunction(k => document.querySelectorAll('.message.pip').length > k && !App.isGenerating, n, { timeout: 300000 });
    console.log(`Kham: ${msg}\n  ${(await p.$$eval('.message.pip', e => e.at(-1).querySelector('div:nth-child(2)').innerText))}`);
  }
  console.log('\n--- prompt the model saw ---\n' + await p.evaluate(() => App.lastPrompt) + '\n---');
  for (const k of ['dreams', 'goalEvolution']) {
    await p.evaluate(key => { App.loopLast = Object.fromEntries(Object.keys(LOOPS).map(x => [x, Date.now()])); App.loopLast[key] = 0; }, k);
    await p.evaluate(() => runDueLoop()); await p.waitForFunction(() => !App.loopBusy, null, { timeout: 300000 });
  }
  const after = await p.evaluate(() => ({ traits: { ...App.soul.traits }, mood: App.soul.mood, user: document.querySelector('.message.user .message-sender').textContent }));
  console.log('\nmood:', before.mood, '→', after.mood, '| user bubble label:', after.user);
  for (const k of Object.keys(before.traits)) console.log(`  ${k.padEnd(12)} ${before.traits[k].toFixed(4)} → ${after.traits[k].toFixed(4)}`);

  if (SHOT) {
    await p.evaluate(() => { document.getElementById('settings-panel').classList.add('open'); refreshPromptView(); });
    await p.evaluate(() => document.getElementById('dev-section').scrollIntoView());
    await p.waitForTimeout(300); await p.screenshot({ path: SHOT + '-dev.png' });
    await p.evaluate(() => document.querySelector('.soul-field').scrollIntoView());
    await p.waitForTimeout(300); await p.screenshot({ path: SHOT + '-soul.png' });
  }

  // the real Export button → fresh page → import → everything back
  const [dl] = await Promise.all([p.waitForEvent('download'), p.click('#export-btn')]);
  const file = path.join(os.tmpdir(), 'soul_roundtrip.json'); await dl.saveAs(file);
  await p.close();
  p = await ctx.newPage(); p.on('pageerror', e => errs.push(e.message));
  await p.goto(page);
  await p.setInputFiles('#import-file-input', file);
  await p.waitForFunction(() => App.soul, null, { timeout: 30000 });
  const back = await p.evaluate(() => ({ name: App.soul.name, user: App.userName, about: App.about, temp: App.dev.temperature, topK: App.dev.topK,
    userField: document.getElementById('user-name').value, mood: App.soul.mood, curiosity: App.soul.traits.curiosity.toFixed(4) }));
  console.log('\nafter export → import:', JSON.stringify(back));
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
