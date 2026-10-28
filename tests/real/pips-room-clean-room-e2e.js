// Needs Playwright (npm i playwright). Run from the repo root (a 0.4B is the size that can reason):
//   node tests/real/pips-room-clean-room-e2e.js rwkv7-g1-0.4b.gguf some_soul.json
// Clean-room thinking + the thought panel. For each question: the reasoning streams into a folded panel above the
// reply (closed unless tapped), the finished reply keeps it (closed), the reasoning is kept with its conclusion, the
// living state gets only the conclusion line (not the reasoning), and a restored conversation shows the panel again.
const { chromium } = require('playwright'); const path = require('path');
const [MODEL, SOUL] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ args: ['--js-flags=--max-old-space-size=6000'] }); const p = await b.newPage();
  const errs = []; let bad = 0; p.on('pageerror', e => errs.push(e.message));
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) bad++; };
  await p.goto('file://' + path.resolve('pip-suite/pips-room.html'));
  await p.setInputFiles('#import-file-input', SOUL); await p.waitForFunction(() => App.soul);
  await p.evaluate(() => { App.userName = 'Kham'; App.dev.maxTokens = 40; App.dev.contextTurns = 1; });
  await p.setInputFiles('#file-input', MODEL); await p.waitForFunction(() => App.isAwake, null, { timeout: 1800000 }); await p.evaluate(() => clearInterval(App.loopTimer));
  await p.evaluate(async () => { while (await indexOneMemory()); });
  await p.evaluate(() => Object.assign(App.dev, { think: true, thinkLength: 'bit', thinkTokens: 600, cleanRoom: true }));
  await p.evaluate(() => { window.__think = []; const orig = streamBubble; streamBubble = () => { const s = orig();
    return { ...s, think: t => { __think.push(t.length); s.think(t); } }; }; });
  const Q = ['Do you remember how many pets I have?', 'Do you remember which town I live in?', 'If I have 4 cats and adopt 3 more, how many cats do I have?'];
  for (const q of Q) {
    const n = await p.$$eval('.message.pip', e => e.length), t0 = Date.now(), tok0 = await p.evaluate(() => App.live.tokens);
    await p.evaluate(() => { __think = []; });
    await p.fill('#message-input', q); await p.press('#message-input', 'Enter');
    await p.waitForFunction(k => document.querySelectorAll('.message.pip').length > k && !App.isGenerating, n, { timeout: 3600000 });
    const r = await p.evaluate(() => { const m = [...document.querySelectorAll('.message.pip')].at(-1), d = m.querySelector('details.message-thought');
      const rs = App.memories.filter(x => x.type === 'reasoning').at(-1);
      return { reply: m.querySelector(':scope > div:not(.message-sender):not(.message-actions):not(.message-recall)')?.innerText || '',
               panel: !!d, open: d ? d.open : null, summary: d?.querySelector('summary').textContent, conclusion: rs?.conclusion, thought: rs?.text || '',
               frames: __think.length, prompt: App.lastPrompt }; });
    const grew = await p.evaluate(() => App.live.tokens) - tok0;
    console.log(`\n### ${q} (${((Date.now() - t0) / 1000).toFixed(0)} s)\n  thought: ${r.thought.replace(/\n/g, ' ').slice(0, 300)}…\n  conclusion: ${r.conclusion}\n  reply: ${r.reply}\n  panel: ${r.summary}`);
    ok(r.panel && r.open === false && r.frames > 5, `the reasoning streamed into a folded panel (${r.frames} updates) that stays closed on the reply`);
    ok(r.thought.length > 0 && typeof r.conclusion === 'string' && grew < 200 && !r.prompt.split('[then, in their living state]')[1].includes(r.thought.slice(0, 40)),
       `only the conclusion entered the living state (+${grew} tokens; the reasoning didn't)`);
  }
  await p.evaluate(() => renderRestored());
  ok(await p.$$eval('details.message-thought', e => e.length) >= Q.length, 'a restored conversation shows the thought panels again');
  ok(!errs.length, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); console.log(bad ? `${bad} FAILED` : 'all passed'); process.exit(bad ? 1 : 0);
})();
