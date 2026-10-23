// Needs Playwright (npm i playwright). Run from the repo root:
//   node tests/real/suite-prompts-e2e.js rwkv7-g1d-0.1b.pth
// Multi-Pip Chat: two copies of the real model, a ✎ note on one, your name and top-K from the
// developer console, a 4-turn chat; prints the prompt the last speaker saw.
// Junto: a node with a Role answers in chat format (one line); a node without one is continued raw.
const { chromium } = require('playwright'); const path = require('path');
const MODEL = process.argv[2];
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=4096'] }); const errs = [];

  // ── Multi-Pip Chat ──
  let p = await b.newPage(); p.on('pageerror', e => errs.push('multi: ' + e.message));
  const answers = ['Goose', 'Gander', 'You are a goose who loves frogs and speaks gently.'];
  p.on('dialog', d => d.accept(answers.shift()));
  await p.goto('file://' + path.resolve('pip-suite/multi-pip-chat.html'));
  for (let i = 0; i < 2; i++) {
    await p.setInputFiles('#file-input', MODEL);
    await p.waitForFunction(n => ChatRoom.pips.length > n, i, { timeout: 300000 });
  }
  await p.click('.pip-item:nth-child(1) .pip-name');                 // rename → Goose
  await p.click('.pip-item:nth-child(2) .pip-name');                 // rename → Gander
  await p.click('.pip-item:nth-child(1) .pip-note');                 // ✎ note on Goose
  await p.click('#dev-btn');
  await p.fill('#dev-fields input[data-key="userName"]', 'Kham'); await p.press('#dev-fields input[data-key="userName"]', 'Tab');
  await p.fill('#dev-fields input[data-key="topK"]', '40'); await p.press('#dev-fields input[data-key="topK"]', 'Tab');
  await p.fill('#chat-turns', '4');
  await p.click('#start-chat-btn');
  await p.waitForFunction(() => !ChatRoom.isRunning && ChatRoom.conversationHistory.filter(m => m.sender !== 'system').length >= 4, null, { timeout: 900000 });
  const lines = await p.evaluate(() => ChatRoom.conversationHistory.filter(m => m.sender !== 'system').map(m => `${m.sender}: ${m.text}`));
  console.log('MULTI-PIP chat:\n  ' + lines.join('\n  '));
  console.log('last prompt ' + await p.textContent('#dev-prompt-who') + ':\n' + await p.textContent('#dev-prompt'));
  console.log('notes:', JSON.stringify(await p.evaluate(() => ChatRoom.pips.map(x => [x.name, x.about || '']))), '| DEV.topK =', await p.evaluate(() => DEV.topK));
  await p.close();

  // ── Junto ──
  p = await b.newPage(); p.on('pageerror', e => errs.push('junto: ' + e.message));
  await p.goto('file://' + path.resolve('pip-suite/junto-orchestrator.html'));
  await p.setInputFiles('#file-input', MODEL);
  await p.waitForFunction(() => Swarm.nodes.length > 0, null, { timeout: 300000 });
  const ask = async q => {
    await p.fill('#inference-input', q); await p.click('#inference-run-btn');
    await p.waitForFunction(() => !document.querySelector('#inference-output').innerText.includes('Processing'), null, { timeout: 300000 });
    return (await p.$eval('#inference-output .output-response', e => e.innerText)).trim();
  };
  p.on('dialog', d => d.accept('Frogbert'));
  await p.click('#node-detail-name');                                 // rename the node
  await p.click('#dev-btn');
  console.log('\nJUNTO raw (no role): ' + JSON.stringify(await ask('Frogs are')));
  console.log('  prompt: ' + JSON.stringify(await p.textContent('#dev-prompt')));
  await p.fill('#node-role', 'You are the frog expert of the swarm. You answer in one short, friendly sentence.'); await p.press('#node-role', 'Tab');
  console.log('JUNTO with role: ' + JSON.stringify(await ask('What do frogs eat?')));
  console.log('  prompt: ' + JSON.stringify(await p.textContent('#dev-prompt')));
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
