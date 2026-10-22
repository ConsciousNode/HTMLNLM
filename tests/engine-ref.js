// Does the embedded RWKV-7 engine compute exactly what official RWKV-7 computes?
// 1) python3 tests/ref/rwkv7_ref.py writes a random checkpoint in the official format plus reference logits
//    (numpy transcription of BlinkDL's rwkv/model.py; copied from Brymar College).
// 2) Each engine copy (engine/rwkv7-engine.js and the copies embedded in pip-suite/*.html) loads that file
//    through its own browser loader and must match the reference to < 1e-9 (float64).
// Zero dependencies beyond python3 + numpy: `node tests/engine-ref.js`
const fs = require('fs'), path = require('path'), vm = require('vm'), os = require('os'), { execFileSync } = require('child_process');
const root = path.join(__dirname, '..');
const BEGIN = '// ==== RWKV7 ENGINE', END = '// ==== END RWKV7 ENGINE ====';
const extract = src => { const i = src.indexOf(BEGIN), j = src.indexOf(END, i); if (i < 0 || j < 0) throw new Error('engine markers not found'); return src.slice(i, j + END.length); };
const canonical = extract(fs.readFileSync(path.join(root, 'engine', 'rwkv7-engine.js'), 'utf8'));
const copies = [['engine/rwkv7-engine.js', canonical]];
for (const f of ['pip-suite/junto-orchestrator.html', 'pip-suite/multi-pip-chat.html']) {
  const p = path.join(root, f); if (!fs.existsSync(p)) continue;
  const src = fs.readFileSync(p, 'utf8'); if (!src.includes(BEGIN)) { console.log(`${f}: no engine embedded (skipped)`); continue; }
  copies.push([f, extract(src)]);
}
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rwkv7ref-'));
execFileSync('python3', [path.join(__dirname, 'ref', 'rwkv7_ref.py'), dir], { stdio: 'pipe' });
const exp = JSON.parse(fs.readFileSync(path.join(dir, 'expected.json'), 'utf8'));
const buf = fs.readFileSync(path.join(dir, 'mini.safetensors')), ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
let fails = 0;
for (const [name, code] of copies) {
  if (code !== canonical) { console.log(`${name}: embedded engine differs from engine/rwkv7-engine.js  FAIL (run engine/embed.py)`); fails++; continue; }
  const ctx = { TextDecoder, TextEncoder, DataView, Uint8Array, Uint16Array, Uint32Array, Float32Array, Float64Array, BigInt, Map, Math, JSON, Object, Number, String, Error, Array, Symbol, console };
  vm.createContext(ctx); vm.runInContext(code + '\nthis.E = { loadRWKV7FromBuffer };', ctx);
  const { model, cfg, report } = ctx.E.loadRWKV7FromBuffer(ab, 'mini.safetensors', { Float: Float64Array });
  let st = model.getEmptyState(), maxd = 0;
  exp.tokens.forEach((t, i) => { const r = model.forward(t, st); st = r.state; for (let j = 0; j < r.logits.length; j++) maxd = Math.max(maxd, Math.abs(r.logits[j] - exp.logits[i][j])); });
  const ok = maxd < 1e-9 && report.unused.length === 0 && report.missing.length === 0 && Object.keys(model.g).length === 0;
  if (!ok) fails++;
  console.log(`${name}: dim ${cfg.dim}, ${cfg.numLayers} layers, ${report.loaded}/${report.total} tensors · max |Δlogit| vs official ${maxd.toExponential(2)} · grad buffers ${Object.keys(model.g).length}  ${ok ? 'MATCH' : 'FAIL'}`);
}
fs.rmSync(dir, { recursive: true, force: true });
console.log(fails ? `\n${fails} FAILED` : '\nALL MATCH');
process.exit(fails ? 1 : 0);
