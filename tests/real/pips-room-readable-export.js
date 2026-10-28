// Needs Playwright. Run from the repo root: node tests/real/pips-room-readable-export.js some_soul.json
// No model needed. "Export data (JSON)" and "Export as text (Markdown)": everything except the living
// state(s) and recall vectors. Checks every exchange and inner-life entry is there, no state data, and the size.
const { chromium } = require('playwright'); const path = require('path'); const fs = require('fs'); const os = require('os');
const SOUL = process.argv[2];
(async () => {
  const b = await chromium.launch(); const ctx = await b.newContext({ acceptDownloads: true }); const p = await ctx.newPage();
  const errs = []; let bad = 0; p.on('pageerror', e => errs.push(e.message));
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) bad++; };
  await p.goto('file://' + path.resolve('pip-suite/pips-room.html'));
  await p.setInputFiles('#import-file-input', SOUL); await p.waitForFunction(() => App.soul, null, { timeout: 30000 });
  await p.evaluate(() => document.getElementById('settings-panel').classList.add('open'));
  const grab = async id => { const [dl] = await Promise.all([p.waitForEvent('download'), p.click(id)]);
    const f = path.join(os.tmpdir(), dl.suggestedFilename()); await dl.saveAs(f); return f; };
  const src = JSON.parse(fs.readFileSync(SOUL, 'utf8'));
  const fj = await grab('#export-data-btn'), fm = await grab('#export-text-btn');
  const d = JSON.parse(fs.readFileSync(fj, 'utf8')), md = fs.readFileSync(fm, 'utf8');
  const entries = d.threads.flatMap(t => t.entries);
  console.log(`soul file ${(fs.statSync(SOUL).size / 1024).toFixed(0)} KB → ${path.basename(fj)} ${(fs.statSync(fj).size / 1024).toFixed(0)} KB, ${path.basename(fm)} ${(fs.statSync(fm).size / 1024).toFixed(0)} KB`);
  ok(entries.length === src.memories.length, `every memory is in the JSON (${entries.length}/${src.memories.length})`);
  ok(entries.filter(e => e.kind === 'exchange').length === src.memories.filter(m => m.type === 'conversation').length, 'every exchange is there');
  ok(!/"(state|b64|vecs)"/.test(fs.readFileSync(fj, 'utf8')), 'no living state or vectors in it');
  ok(entries.every(e => !isNaN(Date.parse(e.time))), 'times are ISO dates');
  ok(src.memories.filter(m => m.type === 'conversation').every(m => md.includes(m.user.split('\n')[0])), 'every line you said is in the Markdown');
  ok(md.startsWith('# ' + src.soul.name) && d.threads.every(t => md.includes('## ' + t.title)), 'Markdown has the resident and every thread as headings');
  console.log('\n' + md.split('\n').slice(0, 24).join('\n') + '\n…');
  ok(!errs.length, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); console.log(bad ? `${bad} FAILED` : 'all passed'); process.exit(bad ? 1 : 0);
})();
