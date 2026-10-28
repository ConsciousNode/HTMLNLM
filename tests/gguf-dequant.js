// Checks the engine's GGUF dequantizers (and the fused row·x kernels) against the gguf Python package, which is
// llama.cpp's own reference, on random blocks of every supported type. `node tests/gguf-dequant.js` (needs: pip install gguf)
// Also checks the loader's refusals: a file with zeroed weights, and a checkpoint with unknown tensors.
const fs = require('fs'), path = require('path'), vm = require('vm'), os = require('os'), { execFileSync } = require('child_process');
const root = path.join(__dirname, '..');
const ctx = { TextDecoder, TextEncoder, DataView, Uint8Array, Int8Array, Uint16Array, Uint32Array, Float32Array, Float64Array, BigInt, Map, Set, Math, JSON, Object, Number, String, Error, Array, Symbol, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(root, 'engine', 'rwkv7-engine.js'), 'utf8') + '\nthis.E = { QTensor, _dequantBlocks, loadRWKV7FromBuffer };', ctx);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gguf-')), f = path.join(dir, 'blocks.json');
console.log(execFileSync('python3', [path.join(__dirname, 'ref', 'make_gguf_blocks.py'), f]).toString().trim());
const ref = JSON.parse(fs.readFileSync(f, 'utf8')); let bad = 0;
for (const [type, r] of Object.entries(ref)) {
  const q = new ctx.E.QTensor(type, r.rows, r.cols, Uint8Array.from(r.bytes));
  let dmax = 0; for (let i = 0; i < r.rows; i++) { const row = q.row(i); for (let c = 0; c < r.cols; c++) dmax = Math.max(dmax, Math.abs(row[c] - r.deq[i * r.cols + c])); }
  const y = q.matvec(Float32Array.from(r.x), new Float32Array(r.rows));
  let ymax = 0, yscale = 0; for (let i = 0; i < r.rows; i++) { ymax = Math.max(ymax, Math.abs(y[i] - r.dot[i])); yscale = Math.max(yscale, Math.abs(r.dot[i])); }
  const ok = dmax < 1e-6 && ymax <= 1e-5 * Math.max(1, yscale);
  if (!ok) bad++;
  console.log(`${type.padEnd(5)} dequant max|Δ| ${dmax.toExponential(1)}   row·x max|Δ| ${ymax.toExponential(1)} (of ${yscale.toFixed(2)})   ${ok ? 'ok' : 'FAIL'}`);
}
// refusals, on a tiny real-layout checkpoint written by tests/ref (same as tests/pth-reader.js)
execFileSync('python3', [path.join(__dirname, 'ref', 'rwkv7_ref.py'), dir], { stdio: 'pipe' });
const st = fs.readFileSync(path.join(dir, 'mini.safetensors')), ab = st.buffer.slice(st.byteOffset, st.byteOffset + st.length);
const edit = fn => {                            // rewrite the safetensors header/data through a callback
  const dv = new DataView(ab), hl = Number(dv.getBigUint64(0, true)), h = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 8, hl)));
  const data = new Uint8Array(ab.slice(8 + hl)); fn(h, data);
  const hb = new TextEncoder().encode(JSON.stringify(h)), out = new Uint8Array(8 + hb.length + data.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(hb.length), true); out.set(hb, 8); out.set(data, 8 + hb.length); return out.buffer; };
const refuses = (name, buf, re) => { try { ctx.E.loadRWKV7FromBuffer(buf, 'x.safetensors'); console.log(`${name}: loaded (FAIL, should refuse)`); bad++; }
  catch (e) { const ok = re.test(e.message); if (!ok) bad++; console.log(`${name}: refused ${ok ? 'ok' : 'FAIL'} - ${e.message.slice(0, 110)}`); } };
refuses('zeroed weights', edit((h, d) => { const [s, e] = h['blocks.1.att.key.weight'].data_offsets; d.fill(0, s, e); }), /damaged/);
refuses('rwkv7a extra tensor', edit((h, d) => { h['blocks.1.ffn.s_emb.weight'] = { ...h['blocks.1.ln1.weight'] }; }), /RWKV-7a/);
fs.rmSync(dir, { recursive: true, force: true });
console.log(bad ? `FAIL (${bad})` : 'PASS'); process.exit(bad ? 1 : 0);
