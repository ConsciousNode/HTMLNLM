// Needs Playwright. Run from the repo root: node tests/real/multi-pip-factory.js
// No model needed: stand-in RWKV-7 models (real World tokenizer, scripted logits) in Multi-Pip Chat.
// Checks: the topic actually reaches the models; an interjection appears once in the prompt; a seed
// replays a chat exactly; stateful and stateless runs match while the chat fits the 5-line window;
// a scenario deck renames the Pips, opens with its line and hands the turn on; the export carries the
// provenance stamp, how each line ended, and filter-1's flags.
const { chromium } = require('playwright'); const path = require('path');
const FAKE = `(() => {
  const tok = worldTokenizer();
  const W = [' frog',' moss',' rain',' yes',' maybe',' the',' pond',' hello',' sing',' night',' green','!','.',' I',' you'].map(w => tok.encode(w)[0]);
  const nl = tok.encode('\\n')[0];
  window.mk = (name) => ({ id: 'pip-' + name, name, version: 'x070', kind: 'rwkv7', named: true, fingerprint: 'sha256-headtail:test',
    params: '0.0M', fileName: name + '.pth', about: '', tokenizer: tok,
    model: { getEmptyState: () => ({ n: 0, h: 0 }), forward: (id, st) => { const h = (st.h * 31 + id) % 9973, l = new Float32Array(65536).fill(-30);
      W.forEach((w, i) => l[w] = ((i * 7 + h) % 5) * 0.3); l[nl] = (st.n % 9) > 6 ? 2 : -3; return { logits: l, state: { n: st.n + 1, h } }; } } });
})();`;
(async () => {
  const b = await chromium.launch(); const p = await b.newPage(); const errs = []; let bad = 0;
  p.on('pageerror', e => errs.push(e.message)); p.on('dialog', d => d.accept());
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) bad++; };
  await p.goto('file://' + path.resolve('pip-suite/multi-pip-chat.html')); await p.evaluate(FAKE);
  await p.evaluate(`ChatRoom.addPip(mk('Wren')); ChatRoom.addPip(mk('Moss'));`);
  const run = async (opts) => { await p.evaluate(o => { ChatRoom.exported = true; ChatRoom.clear(); Object.assign(DEV, o.dev);
      document.getElementById('chat-topic').value = o.topic; document.getElementById('chat-turns').value = String(o.turns);
      document.getElementById('chat-delay').value = '100'; ChatRoom.start(); }, opts);
    await p.waitForFunction('!ChatRoom.isRunning', null, { timeout: 120000 }); return p.evaluate('ChatRoom.buildExport()'); };
  // topic reaches the model (stateless prompt shows it as the first chat line)
  const e1 = await run({ dev: { seed: '7', stateful: false, userName: 'Kham' }, topic: 'philosophy', turns: 2 });
  const prompt = await p.evaluate('DEV.lastPrompt');
  ok(prompt.includes('Kham: Let\'s discuss the nature of consciousness'), 'topic is in the prompt as a chat line');
  // interjection appears once, no "Wren:User:"
  await p.evaluate(`ChatRoom.userMessage('do you like frogs?')`); await p.waitForTimeout(1500);
  const ip = await p.evaluate('DEV.lastPrompt');
  ok(ip.split('do you like frogs?').length === 2 && !/:User:/.test(ip), 'an interjection appears once, no malformed "Pip:User:"');
  // seed replay, and stateful vs stateless
  const s1 = await run({ dev: { seed: '420', stateful: true }, topic: 'free', turns: 10 });
  const s2 = await run({ dev: { seed: '420', stateful: true }, topic: 'free', turns: 10 });
  const s3 = await run({ dev: { seed: '420', stateful: false }, topic: 'free', turns: 10 });
  const T = e => e.dialogue.map(d => d.text);
  ok(JSON.stringify(T(s1)) === JSON.stringify(T(s2)), 'same seed + settings replays the chat exactly');
  let k = 0; while (k < 10 && T(s1)[k] === T(s3)[k]) k++;
  ok(k >= 5, `stateful and stateless match while the chat fits the 5-line window (${k} identical lines)`);
  ok(s1.run.segments[0].settings.mode === 'stateful' && s3.run.segments[0].settings.mode === 'stateless', 'mode recorded per run');
  ok(s1.dialogue.every(d => d.gen && ['newline', 'eos', 'cap'].includes(d.gen.endedBy)), 'every line records how it ended');
  ok(/^filter-1/.test(s1.filter.version) &&Array.isArray(s1.dialogue[0].flags), 'export carries filter-1 flags');
  ok(s1.pips.every(x => x.fingerprint && x.systemBlock) && s1.run.seed === '420', 'provenance: fingerprint, system line, seed');
  // interjections and pause/resume mid-turn (the Oct 27 two-loops bug): one loop, rotation intact, turn count kept
  await p.evaluate(`ChatRoom.exported = true; ChatRoom.clear(); ChatRoom.addPip(mk('Fern'));`);
  await p.evaluate(o => { Object.assign(DEV, { seed: '9', stateful: true }); document.getElementById('chat-topic').value = 'free';
    document.getElementById('chat-turns').value = '12'; document.getElementById('chat-delay').value = '50'; ChatRoom.start(); }, {});
  const midTurn = () => p.waitForFunction('!!document.getElementById("typing-indicator")', null, { timeout: 30000 });
  await midTurn(); await p.waitForTimeout(120);
  await p.evaluate(`ChatRoom.userMessage('frogs or toads?')`);
  await p.waitForTimeout(150); await p.evaluate(`ChatRoom.pause()`); await p.waitForTimeout(30); await p.evaluate(`ChatRoom.pause()`);   // pause + resume mid-turn
  await p.waitForTimeout(80); await p.evaluate(`ChatRoom.userMessage('say hi to the moss')`);
  await p.waitForTimeout(40); await p.evaluate(`ChatRoom.pause()`); await p.waitForTimeout(400); await p.evaluate(`ChatRoom.pause()`);
  await p.waitForFunction('!ChatRoom.isRunning', null, { timeout: 120000 });
  const iv = await p.evaluate('ChatRoom.buildExport()');
  const pipLines = iv.dialogue.filter(d => d.role === 'pip'), humans = iv.dialogue.filter(d => d.role === 'human').map(d => d.text);
  const rot = ['Wren', 'Moss', 'Fern'];
  ok(pipLines.length === 12, `exactly 12 Pip lines for 12 turns (${pipLines.length})`);
  ok(pipLines.every((d, i) => d.speaker === rot[i % 3]), 'rotation never breaks: ' + pipLines.map(d => d.speaker[0]).join(''));
  ok(JSON.stringify(humans) === JSON.stringify(['frogs or toads?', 'say hi to the moss']), 'each interjection appears once, in order');
  ok(iv.run.segments.length === 1 && iv.run.endReason === 'complete', `one segment, ends complete (${iv.run.segments.length}, ${iv.run.endReason})`);
  // when the chat isn't running, your line gets one answer from the next Pip in the rotation
  await p.evaluate(`ChatRoom.userMessage('goodnight')`); await p.waitForFunction('ChatRoom.conversationHistory.at(-1).kind === "pip"', null, { timeout: 30000 });
  const after = await p.evaluate('ChatRoom.conversationHistory.slice(-2).map(m => m.kind + ":" + m.sender)');
  ok(after[0] === 'human:You' && after[1] === 'pip:Wren', 'idle: the next Pip in the rotation answers you once (' + after.join(', ') + ')');
  await p.evaluate(`ChatRoom.exported = true; ChatRoom.clear(); ChatRoom.removePip('pip-Fern');`);
  // deck
  await p.evaluate(`ChatRoom.exported = true; ChatRoom.clear(); ChatRoom.addPip(mk('Clover'));`);
  await p.selectOption('#chat-topic', 'deck:frog-pond');
  const names = await p.evaluate('ChatRoom.pips.map(x => x.name)');
  ok(names[0] === 'Wren' && names[1] === 'Moss', 'deck renames the Pips in order');
  const d = await run({ dev: { seed: '3' }, topic: 'deck:frog-pond', turns: 3 });
  ok(d.dialogue[0].role === 'opener' && d.dialogue[0].speaker === 'Wren' && d.dialogue[1].speaker === 'Moss', 'deck opener from Wren, then Moss answers');
  ok(d.deck && d.deck.id === 'frog-pond' && d.run.segments[0].deck.id === 'frog-pond', 'deck recorded in the export');
  ok(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); console.log(bad ? `\n${bad} FAILED` : '\nALL OK'); process.exit(bad ? 1 : 0);
})();
