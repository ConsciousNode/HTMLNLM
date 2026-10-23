// Needs Playwright. Run from the repo root: node tests/real/pips-room-layout.js [page.html]
// The message bar must be on screen at phone sizes, in a phone browser's "desktop site" mode
// (a ~980px-wide layout on a tall screen, with browser bars covering part of it), and on desktops,
// even with a long conversation and the inner-life panel open.
const { chromium, devices } = require('playwright'); const path = require('path');
const PAGE = 'file://' + path.resolve(process.argv[2] || 'pip-suite/pips-room.html');
const cases = [
  ['phone', { ...devices['Pixel 7'] }],
  ['phone, desktop site', { viewport: { width: 980, height: 1900 }, screen: { width: 980, height: 2180 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true }],
  ['phone, desktop site, landscape', { viewport: { width: 980, height: 400 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true }],
  ['laptop', { viewport: { width: 1280, height: 720 } }],
  ['short window', { viewport: { width: 900, height: 420 } }],
];
(async () => {
  const b = await chromium.launch();
  for (const [name, opts] of cases) {
    const ctx = await b.newContext(opts), p = await ctx.newPage();
    await p.goto(PAGE);
    const r = await p.evaluate(() => {
      for (const id of ['conversation-area', 'input-area', 'inner-life-panel']) document.getElementById(id).classList.remove('hidden', 'collapsed');
      document.getElementById('empty-state').style.display = 'none';
      const c = document.getElementById('conversation-area');
      for (let i = 0; i < 40; i++) { const d = document.createElement('div'); d.className = 'message pip'; d.textContent = 'A long line about frogs. '.repeat(8); c.appendChild(d); }
      const box = document.getElementById('message-input').getBoundingClientRect();
      const vh = window.visualViewport ? window.visualViewport.height : innerHeight;
      return { bottom: Math.round(box.bottom), top: Math.round(box.top), visible: Math.round(vh), onScreen: box.top >= 0 && box.bottom <= vh + 1 && box.height > 0 };
    });
    console.log(`${name.padEnd(32)} message bar ${r.top}–${r.bottom}px of ${r.visible}px visible: ${r.onScreen ? 'on screen' : 'OFF SCREEN'}`);
    await ctx.close();
  }
  await b.close();
})();
