// Checks the engine's browser .pth reader (zip + pickle, no PyTorch) and .safetensors reader against
// a PyTorch-layout archive written by tests/ref/make_pth.py: BF16 + F32 storages, shared storages with
// offsets, OrderedDict with _metadata. Every tensor must come back bit-for-bit. `node tests/pth-reader.js`
const fs = require('fs'), path = require('path'), vm = require('vm'), os = require('os'), { execFileSync } = require('child_process');
const root = path.join(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'engine', 'rwkv7-engine.js'), 'utf8');
const ctx = { TextDecoder, TextEncoder, DataView, Uint8Array, Uint16Array, Uint32Array, Float32Array, Float64Array, BigInt, Map, Math, JSON, Object, Number, String, Error, Array, Symbol, console };
vm.createContext(ctx); vm.runInContext(src + '\nthis.E = { readPthBuffer, readSafetensorsBuffer, loadRWKV7FromBuffer };', ctx);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pth-'));
execFileSync('python3', [path.join(__dirname, 'ref', 'rwkv7_ref.py'), dir], { stdio: 'pipe' });
const st = path.join(dir, 'mini.safetensors'), pth = path.join(dir, 'mini.pth'), expf = path.join(dir, 'pth_expected.json');
console.log(execFileSync('python3', [path.join(__dirname, 'ref', 'make_pth.py'), st, pth, expf]).toString().trim());
const ab = f => { const b = fs.readFileSync(f); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); };
const exp = JSON.parse(fs.readFileSync(expf, 'utf8')), got = ctx.E.readPthBuffer(ab(pth));
let bad = 0;
for (const [n, e] of Object.entries(exp)) {
  const g = got[n]; if (!g) { console.log(`missing ${n}`); bad++; continue; }
  const u = new Uint32Array(g.data.buffer, g.data.byteOffset, g.data.length); let h = 0n; for (const x of u) h += BigInt(x);
  const ok = JSON.stringify(g.shape) === JSON.stringify(e.shape) && h === BigInt(e.hash) && e.first.every((v, i) => Object.is(Math.fround(v), g.data[i]));
  if (!ok) { bad++; console.log(`MISMATCH ${n}`); }
}
const extra = Object.keys(got).filter(n => !exp[n]);
console.log(`.pth: ${Object.keys(exp).length - bad}/${Object.keys(exp).length} tensors bit-exact${extra.length ? `, ${extra.length} unexpected` : ''}`);
const m = ctx.E.loadRWKV7FromBuffer(ab(pth), 'mini.pth');
console.log(`.pth → RWKV7: ${m.report.loaded}/${m.report.total} tensors, ${m.report.unused.length} unused, ${m.report.missing.length} missing, ${(m.nParams / 1e3).toFixed(1)}K params`);
fs.rmSync(dir, { recursive: true, force: true });
const ok = bad === 0 && !extra.length && m.report.unused.length === 0 && m.report.missing.length === 0;
console.log(ok ? 'PASS' : 'FAIL'); process.exit(ok ? 0 : 1);
