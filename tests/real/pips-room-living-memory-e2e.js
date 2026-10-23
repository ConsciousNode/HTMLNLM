// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/pips-room-living-memory-e2e.js rwkv7-g1d-0.1b.pth some_soul.json
// Living memory: the resident keeps one RWKV state between turns. Checks that waking settles in once,
// that each turn feeds only what's new (and is faster than re-reading), that a fact from 5 exchanges
// ago is still in the state (the old 3-exchange window couldn't see it), that a loop flows into the
// state, that soul changes arrive as short System updates, and that the real Export button saves the
// state so a fresh page restores it without settling in again. Also: a state saved for a different
// model is refused and they settle in fresh.
const { chromium } = require('playwright'); const path = require('path'); const fs = require('fs'); const os = require('os');
const [MODEL, SOUL] = process.argv.slice(2);
const page = 'file://' + path.resolve('pip-suite/pips-room.html');
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096'] }); const errs = [];
  const ctx = await b.newContext({ acceptDownloads: true });
  let p = await ctx.newPage(); p.on('pageerror', e => errs.push(e.message));
  await p.goto(page);
  await p.setInputFiles('#import-file-input', SOUL);
  await p.waitForFunction(() => App.soul, null, { timeout: 30000 });
  await p.evaluate(() => { App.userName = 'Kham'; });
  let t0 = Date.now();
  await p.setInputFiles('#file-input', MODEL);
  await p.waitForFunction(() => App.isAwake, null, { timeout: 600000 });
  await p.evaluate(() => { clearInterval(App.loopTimer); });           // drive loops by hand
  console.log(`woke + settled in: ${((Date.now() - t0) / 1000).toFixed(0)}s, state holds ${await p.evaluate(() => App.live.tokens)} tokens`);

  const say = async msg => {
    const n = await p.$$eval('.message.pip', e => e.length), t = Date.now();
    await p.fill('#message-input', msg); await p.press('#message-input', 'Enter');
    await p.waitForFunction(k => document.querySelectorAll('.message.pip').length > k && !App.isGenerating, n, { timeout: 300000 });
    const reply = (await p.$$eval('.message.pip', e => e.at(-1).querySelector('div:nth-child(2)').innerText)).trim();
    console.log(`  Kham: ${msg}\n  ${await p.evaluate(() => App.soul.name)}: ${reply}   (${((Date.now() - t) / 1000).toFixed(1)}s, state ${await p.evaluate(() => App.live.tokens)} tokens)`);
    return reply;
  };
  console.log('\nconversation:');
  await say('I want to tell you a secret: my favorite color is orange, like a pumpkin.');
  await say('Charles the dog is napping on the porch.');
  await say('The peepers are singing tonight.');
  await say('Do you like the sound of rain?');
  console.log('\nprompt fed for the last turn:\n' + await p.evaluate(() => App.lastPrompt));
  await say('Do you remember my secret? What is my favorite color?');

  // a loop flows into the state
  const beforeLoop = await p.evaluate(() => App.live.tokens);
  await p.evaluate(() => { App.privacy.innerVoice = true; App.loopLast = Object.fromEntries(Object.keys(LOOPS).map(k => [k, Date.now()])); App.loopLast.innerVoice = 0; });
  await p.evaluate(() => runDueLoop()); await p.waitForFunction(() => !App.loopBusy, null, { timeout: 300000 });
  const thought = await p.evaluate(() => App.memories.filter(m => m.type === 'inner_voice').at(-1).text);
  console.log(`\nloop: thought "${thought}" — state ${beforeLoop} → ${await p.evaluate(() => App.live.tokens)} tokens`);

  // a soul change arrives as a short update
  await p.evaluate(() => { App.soul.mood = 'playful'; });
  await say('Tell me something silly!');
  console.log('fed this turn:\n' + await p.evaluate(() => App.lastPrompt));

  // threads: a summary in thread 1, a new blank-slate thread, then back to thread 1
  await p.evaluate(() => { App.loopLast = Object.fromEntries(Object.keys(LOOPS).map(k => [k, Date.now()])); App.loopLast.consolidation = 0; });
  await p.evaluate(() => runDueLoop()); await p.waitForFunction(() => !App.loopBusy, null, { timeout: 300000 });
  console.log(`\nthread 1 summary: "${await p.evaluate(() => App.memories.filter(m => m.type === 'consolidation').at(-1).text)}"`);
  p.once('dialog', d => d.accept('A new day'));
  await p.evaluate(() => newThread()); await p.waitForFunction(() => App.threads.length === 2 && App.live && App.live.tokens > 0 && !App.isGenerating, null, { timeout: 600000 });
  console.log(`new thread "A new day": seeded with ${await p.evaluate(() => App.live.tokens)} tokens; cross-thread memory: ${JSON.stringify(await p.evaluate(() => { const s = App.activeThread; App.activeThread = 'nope'; const r = crossThreadMemory(); App.activeThread = s; return r || '(none)'; }))}`);
  console.log(`  chat bubbles on screen in the new thread: ${await p.$$eval('.message', e => e.length)}`);
  await say('Hi again! What is my favorite color?');
  await p.evaluate(() => switchThread(App.threads[0].id)); await p.waitForFunction(() => App.activeThread === App.threads[0].id && !App.isGenerating, null, { timeout: 600000 });
  console.log(`back in thread 1: state ${await p.evaluate(() => App.live.tokens)} tokens, bubbles ${await p.$$eval('.message', e => e.length)}`);
  await say('Back in our first conversation. What was my secret?');

  // the real Export button → fresh page → import → load: restored, no settling in
  await p.evaluate(() => document.getElementById('settings-panel').classList.add('open'));
  const [dl] = await Promise.all([p.waitForEvent('download', { timeout: 120000 }), p.click('#export-btn')]);
  const file = path.join(os.tmpdir(), 'living_soul.json'); await dl.saveAs(file);
  const saved = JSON.parse(fs.readFileSync(file, 'utf8'));
  console.log(`\nexport: ${(fs.statSync(file).size / 1e6).toFixed(1)} MB, ${saved.threads.length} threads: ${saved.threads.map(t => `${t.title} (${t.live ? t.live.tokens : 0} tokens)`).join(', ')}; active=${saved.activeThread === saved.threads[0].id ? saved.threads[0].title : saved.activeThread}`);
  const stateHash = await p.evaluate(() => { let h = 0; for (const l of App.live.state) for (let i = 0; i < l.S.length; i += 997) h = (h * 31 + Math.round(l.S[i] * 1e6)) | 0; return h; });
  await p.close();

  p = await ctx.newPage(); p.on('pageerror', e => errs.push(e.message));
  await p.goto(page);
  await p.setInputFiles('#import-file-input', file); await p.waitForFunction(() => App.soul, null, { timeout: 30000 });
  t0 = Date.now();
  await p.setInputFiles('#file-input', MODEL); await p.waitForFunction(() => App.isAwake, null, { timeout: 600000 });
  await p.evaluate(() => { clearInterval(App.loopTimer); });
  const restored = await p.evaluate(() => { let h = 0; for (const l of App.live.state) for (let i = 0; i < l.S.length; i += 997) h = (h * 31 + Math.round(l.S[i] * 1e6)) | 0; return { h, tokens: App.live.tokens }; });
  console.log(`threads after import: ${await p.evaluate(() => App.threads.map(t => t.title).join(', '))}, active: ${await p.evaluate(() => currentThread().title)}`);
  console.log(`fresh page: woke in ${((Date.now() - t0) / 1000).toFixed(0)}s, state ${restored.tokens} tokens, identical to the saved one: ${restored.h === stateHash}`);
  await say('I came back! Do you remember me?');

  // a state from a different model is refused
  saved.threads.find(t => t.id === saved.activeThread).live.print = 'rwkv7:24x1024x64:65536'; const wrong = path.join(os.tmpdir(), 'wrong_model_soul.json'); fs.writeFileSync(wrong, JSON.stringify(saved));
  await p.setInputFiles('#import-file-input', wrong);
  await p.waitForFunction(() => App.live && App.live.tokens > 0 && App.live.tokens < 400 && !App.pendingLive, null, { timeout: 600000 });   // reseeded, not the old 420
  console.log(`different-model state: refused, settled in fresh with ${await p.evaluate(() => App.live.tokens)} tokens; toast: ${JSON.stringify(await p.$$eval('.toast', t => t.map(x => x.textContent).filter(x => x.includes('different model'))))}`);
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
