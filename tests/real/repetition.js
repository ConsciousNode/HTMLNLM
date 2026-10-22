// Repetition-control check on a real RWKV-7 checkpoint.
// Runs the same two-speaker chat loop Multi-Pip Chat uses (same context format, same
// one-line-per-turn rule, same sampler), with the anti-repeat setting off vs on, and
// counts how often a speaker repeats one of its own earlier lines.
// The sampler and tracker are pulled out of the page source itself, so this tests the
// code that ships, not a copy of it.
//
//   node --max-old-space-size=6000 tests/real/repetition.js rwkv7-g1d-0.1b.pth <penalty> <seed> [turns]
//
const fs = require('fs'), vm = require('vm'), path = require('path');
const [ckpt, penArg, seedArg, turnsArg] = process.argv.slice(2);
const PEN = +penArg, SEED = +seedArg, TURNS = +(turnsArg || 8);

// seeded RNG (mulberry32) so off/on runs are comparable
let s = SEED >>> 0;
const rand = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const SeededMath = Object.create(Math); SeededMath.random = rand;

const ctx = { TextDecoder, TextEncoder, DataView, Uint8Array, Uint16Array, Uint32Array, Float32Array, Float64Array,
  BigInt, Map, Set, Math: SeededMath, JSON, Object, Number, String, Error, Array, Symbol, console };
vm.createContext(ctx);
const root = path.join(__dirname, '..', '..');
vm.runInContext(fs.readFileSync(path.join(root, 'engine/rwkv7-engine.js'), 'utf8') +
  '\nthis.load=loadRWKV7FromBuffer;this.Tok=RWKVWorldTokenizer;', ctx);
const page = fs.readFileSync(path.join(root, 'pip-suite/multi-pip-chat.html'), 'utf8');
const a = page.indexOf('const REP_DEFAULTS'), b = page.indexOf('function base64ToBuffer', a);
if (a < 0 || b < 0) throw new Error('could not find sampler code in multi-pip-chat.html');
vm.runInContext(page.slice(a, b) + '\nthis.makeRepTracker=makeRepTracker;this.sampleLogits=sampleLogits;', ctx);

const buf = fs.readFileSync(ckpt);
const { model } = ctx.load(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), ckpt);
const tok = new ctx.Tok(JSON.parse(fs.readFileSync(path.join(root, 'engine/rwkv_world_vocab.json'), 'utf8')).tokens);

const speakers = ['Goose', 'Gander'];
const history = [{ sender: 'User', text: 'Good evening, you two. What do frogs say at night?' }];
function respond(name) {
  let context = '';
  for (const m of history.slice(-5)) context += `${m.sender}: ${m.text}\n`;
  context += `${name}:`;
  const ids = tok.encode(context);
  let st = model.getEmptyState(), r;
  for (const id of ids) { r = model.forward(id, st); st = r.state; }
  let rep = null;
  if (PEN > 0) {                                   // same seeding as the page
    rep = ctx.makeRepTracker({ presence: PEN, frequency: PEN });
    for (const m of history.slice(-8)) if (m.sender === name) for (const id of tok.encode(m.text)) rep.add(id);
  }
  const gen = [];
  for (let i = 0; i < 40; i++) {
    const logits = rep ? rep.apply(r.logits) : r.logits;
    let next = ctx.sampleLogits(logits, 0.85, 0.9);
    if (!tok.decode(gen).trim())                     // same no-empty-line guard as the page
      for (let k = 0; k < 8 && (next === 0 || tok.decode([next]).includes('\n')); k++)
        next = ctx.sampleLogits(logits, 0.85, 0.9);
    if (next === 0) break;
    gen.push(next);
    if (rep) { rep.step(); rep.add(next); }
    if (tok.decode(gen).includes('\n')) break;
    r = model.forward(next, st); st = r.state;
  }
  return tok.decode(gen).split('\n')[0].trim();
}

const norm = t => t.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim();
let selfRepeats = 0;
console.log(`penalty=${PEN} seed=${SEED}`);
for (let t = 0; t < TURNS; t++) {
  const name = speakers[t % 2];
  const text = respond(name);
  const repeat = text && history.some(m => m.sender === name && norm(m.text) === norm(text));
  if (repeat) selfRepeats++;
  history.push({ sender: name, text });
  console.log(`${repeat ? '  ↺ ' : '    '}${name}: ${text}`);
}
const words = history.slice(1).map(m => norm(m.text)).join(' ').split(' ').filter(Boolean);
console.log(`RESULT penalty=${PEN} seed=${SEED} self-repeated-lines=${selfRepeats}/${TURNS} distinct-words=${new Set(words).size}/${words.length}`);
