// Training-speed benchmark on the code in HTMLNLM.html. Zero dependencies.
//   node tests/bench.js [dim=256] [layers=4] [tokens=64]
// Reports forward / backward / optimizer time for one sequence, warm (post-JIT).
const fs = require('fs'), path = require('path'), vm = require('vm');
const src = fs.readFileSync(path.join(__dirname, '..', 'HTMLNLM.html'), 'utf8');
const cut = (a, b) => { const i = src.indexOf(a), j = src.indexOf(b, i); if (i < 0 || j < 0) throw new Error('marker ' + a); return src.slice(i, j); };
const code = [cut('const CONFIG = {', '\nfunction sysLog'), cut('class TMAC {', 'class Trainer'), cut('class RWKVv7Block {', 'const JuntoNode'),
  'this.M = { LanguageModel, AdamOptimizer, MuonOptimizer };'].join('\n');
const ctx = { Math, Float32Array, Int8Array, Uint8Array, Array, Object, console, Promise, setTimeout };
vm.createContext(ctx); vm.runInContext(code, ctx);
const [dim, L, T] = process.argv.slice(2).map(Number); const D = dim || 256, LY = L || 4, TK = T || 64;
const M = ctx.M, m = new M.LanguageModel({ vocab: 2048, dim: D, layers: LY });
const seq = Array.from({ length: TK + 1 }, (_, i) => (i * 37) % 2048), now = () => performance.now();
const fwd = () => { let st = m.getEmptyState(); const o = []; for (let i = 0; i < TK; i++) { const r = m.forward(seq[i], st); st = r.state; o.push(r); } return o; };
const bwd = outs => { let g = null; for (let i = TK - 1; i >= 0; i--) { const p = Float32Array.from(outs[i].logits); p[seq[i + 1]] -= 1; g = m.backward(p, outs[i].caches, g).gradStatesOut; } };
(async () => {
  for (let w = 0; w < 2; w++) bwd(fwd());                       // warm up the JIT
  let a = now(); const outs = fwd(); const tf = now() - a;
  a = now(); bwd(outs); const tb = now() - a;
  a = now(); await new M.AdamOptimizer(1e-3).step(m); const ta = now() - a;
  a = now(); await new M.MuonOptimizer(1e-3).step(m); const tm = now() - a;
  const seqMs = 2 * tf + tb;                                    // OOMB runs the forward twice
  console.log(`dim ${D}, ${LY} layers, ${TK} tokens`);
  console.log(`  forward ${tf.toFixed(0)} ms (x2 with OOMB recompute) · backward ${tb.toFixed(0)} ms · ${(TK / (seqMs / 1000)).toFixed(1)} tokens/s`);
  console.log(`  optimizer step: AdamW ${ta.toFixed(0)} ms · Muon ${tm.toFixed(0)} ms`);
})();
