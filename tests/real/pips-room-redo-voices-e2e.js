// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/pips-room-redo-voices-e2e.js rwkv7-g1d-0.1b.pth some_soul.json
// Redo, voices, speed and streaming. Checks: the reply streams in token by token; each reply shows tokens and tok/s; ↻ takes the reply back (living state
// back to the exact token count and the exact state from before the turn, the exchange gone from memory)
// and answers again; ✎ takes the turn back and puts your message in the box; only the latest turn has
// ↻/✎; ▶ is on every bubble and reads with autoread off; the two voices have their own presets and sliders.
const { chromium } = require('playwright'); const path = require('path');
const [MODEL, SOUL] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096', '--autoplay-policy=no-user-gesture-required'] });
  const p = await b.newPage(); const errs = []; let bad = 0; p.on('pageerror', e => errs.push(e.message));
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) bad++; };
  await p.goto('file://' + path.resolve('pip-suite/pips-room.html'));
  await p.setInputFiles('#import-file-input', SOUL); await p.waitForFunction(() => App.soul, null, { timeout: 30000 });
  await p.evaluate(() => { App.userName = 'Kham'; App.dev.maxTokens = 20; App.dev.recall = false; App.dev.speak = false; });
  await p.setInputFiles('#file-input', MODEL); await p.waitForFunction(() => App.isAwake, null, { timeout: 600000 });
  await p.evaluate(() => clearInterval(App.loopTimer));
  const say = async msg => {
    const n = await p.$$eval('.message.pip', e => e.length);
    await p.fill('#message-input', msg); await p.press('#message-input', 'Enter');
    await p.waitForFunction(k => document.querySelectorAll('.message.pip').length > k && !App.isGenerating, n, { timeout: 300000 });
  };
  const view = () => p.evaluate(() => ({ tokens: App.live.tokens, mems: App.memories.length, convs: App.memories.filter(m => m.type === 'conversation').length,
    bubbles: document.querySelectorAll('#conversation-area .message').length, last: [...document.querySelectorAll('.message.pip')].at(-1).children[1].innerText,
    speed: [...document.querySelectorAll('.message.pip')].at(-1).querySelector('.message-speed')?.textContent || '',
    redo: [...document.querySelectorAll('.msg-redo')].map(x => x.textContent + (x.closest('.message').classList.contains('pip') ? 'pip' : 'user')).join(','),
    plays: document.querySelectorAll('.msg-play').length, msgs: document.querySelectorAll('#conversation-area .message').length }));

  await say('Hi Fazzani! What is your favorite color?');
  // streaming: their bubble fills in token by token, then becomes the finished message
  await p.evaluate(() => { window.__frames = []; const orig = streamBubble;
    streamBubble = () => { const s = orig(); return { ...s, update: t => { const el = document.querySelector('.message.pip.streaming');
      if (el) __frames.push(el.children[1].textContent); s.update(t); }, remove: s.remove }; }; });
  await say('Tell me a little story about a frog.');
  const frames = await p.evaluate(() => __frames);
  ok(frames.length >= 3 && frames.every(f => f.endsWith('▍')) && new Set(frames).size >= 3 && !(await p.$('.message.pip.streaming')),
     `the reply streams in (${frames.length} updates, e.g. ${JSON.stringify(frames[Math.floor(frames.length / 2)])}) and the stream bubble is replaced`);
  const a = await view();
  console.log(`reply: "${a.last}" · ${a.speed}`);
  ok(/^\d+ tok · [\d.]+ tok\/s$/.test(a.speed), `the reply shows tokens and speed ("${a.speed}")`);
  ok(a.redo === '✎user,↻pip', `only the latest turn has ✎ and ↻ (${a.redo})`);
  ok(a.plays === a.msgs, `▶ is on every bubble (${a.plays}/${a.msgs})`);

  // the exact state before turn 2, to compare after ↻ takes it back
  const before = await p.evaluate(() => { const S = App.turnSnap; return { tokens: S.live.tokens, sum: Array.from(S.live.state.at(-1).S.slice(0, 64)).reduce((x, y) => x + y, 0) }; });
  await p.evaluate(() => { const r = redoTurn; window._redoDone = false; App.turnSnap.live.state && (window._probe = () => {
    const s = App.live.state.at(-1).S; return Array.from(s.slice(0, 64)).reduce((x, y) => x + y, 0); }); });
  // edit: take the turn back, the text comes back to the box, nothing generated
  await p.click('.message.user .msg-redo');
  const e = await p.evaluate(() => ({ tokens: App.live.tokens, sum: _probe(), input: document.getElementById('message-input').value,
    convs: App.memories.filter(m => m.type === 'conversation').length, bubbles: document.querySelectorAll('#conversation-area .message').length }));
  ok(e.tokens === before.tokens && Math.abs(e.sum - before.sum) < 1e-9, `✎ puts the living state back exactly (${a.tokens} → ${e.tokens} tokens)`);
  ok(e.input === 'Tell me a little story about a frog.' && e.convs === a.convs - 1 && e.bubbles === a.bubbles - 2, '✎ takes the exchange back and puts your message in the box');
  await p.fill('#message-input', 'Tell me a little story about a toad.'); await p.press('#message-input', 'Enter');
  await p.waitForFunction(k => document.querySelectorAll('.message.pip').length > k && !App.isGenerating, e.bubbles / 2, { timeout: 300000 });
  const f = await view();
  ok(f.convs === a.convs && await p.evaluate(() => App.memories.filter(m => m.type === 'conversation').at(-1).user) === 'Tell me a little story about a toad.', 'the edited message is the one remembered');

  // regenerate: same message, answered again from the same state
  await p.evaluate(() => { App.dev.temperature = 1.2; });
  const tok0 = await p.evaluate(() => App.turnSnap.live.tokens);
  await p.click('.message.pip .msg-redo');
  await p.waitForFunction(() => !App.isGenerating && App.turnSnap, null, { timeout: 300000 });
  const g = await view();
  console.log(`regenerated: "${f.last}" → "${g.last}"`);
  ok(g.convs === f.convs && g.bubbles === f.bubbles, `↻ replaces the reply instead of adding one (${g.convs} exchanges, ${g.bubbles} bubbles)`);
  ok(await p.evaluate(t => App.turnSnap.live.tokens === t, tok0), '↻ answers again from the same state as before');
  ok(await p.evaluate(() => App.memories.filter(m => m.type === 'conversation').at(-1).user) === 'Tell me a little story about a toad.', '↻ asks the same message again');

  // voices: two presets, sliders change only that voice, ▶ reads with autoread off
  await p.evaluate(() => { document.getElementById('settings-panel').classList.add('open'); document.getElementById('dev-body').hidden = false; buildDevConsole(); });
  const v = await p.evaluate(() => { const blocks = document.querySelectorAll('.voice-block');
    const r = blocks[1].querySelector('input[type=range]'); r.value = 150; r.dispatchEvent(new Event('input'));
    return { blocks: blocks.length, pip: voiceOf('pip'), user: voiceOf('user') }; });
  ok(v.blocks === 2 && v.pip.preset === 'female' && v.user.preset === 'narrator' && v.user.pitch === 150 && v.pip.pitch == null,
     `two voices, sliders change only their own (${JSON.stringify(v)})`);
  const spoke = await p.evaluate(async () => { let got = null; const orig = WebsterVoice.speak;
    WebsterVoice.speak = (t, voice) => { got = voice; return Promise.resolve(); };
    document.querySelector('.message.user .msg-play').click(); await new Promise(r => setTimeout(r, 50)); WebsterVoice.speak = orig; return got; });
  ok(spoke && spoke.pitch === 150, '▶ on your bubble reads in your voice with autoread off');
  ok(!errs.length, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); console.log(bad ? `${bad} FAILED` : 'all passed'); process.exit(bad ? 1 : 0);
})();
