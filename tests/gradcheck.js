// Finite-difference gradient check for RWKVv7Block / LanguageModel backward,
// run against the classes lifted straight out of HTMLNLM.html.
// A 3-token sequence exercises BPTT through the WKV num/den state AND the
// token-shift (prevX / prevX2) state. Zero dependencies: `node tests/gradcheck.js`
const fs = require('fs'), path = require('path'), vm = require('vm');
const src = fs.readFileSync(path.join(__dirname, '..', 'HTMLNLM.html'), 'utf8');
const cut = (a, b) => { const i = src.indexOf(a), j = src.indexOf(b, i); if (i < 0 || j < 0) throw new Error('marker ' + a); return src.slice(i, j); };
const code = [cut('const CONFIG = {', '\nfunction sysLog'), cut('class TMAC {', 'class AdamOptimizer'), cut('class RWKVv7Block {', 'const JuntoNode'),
  'this.M = { BitLinear, LanguageModel };'].join('\n');
const ctx = { Math, Float32Array, Int8Array, Uint8Array, Array, Object, console }; vm.createContext(ctx); vm.runInContext(code, ctx);
const M = ctx.M;
let seed = 5;   // a well-conditioned draw: see the |den| guard below ctx.Math.random = Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
// STE semantics: backward is the gradient of the float model, so check against a float forward.
M.BitLinear.prototype.forward = function (x) { const o = new Float32Array(this.outDim); for (let r = 0; r < this.outDim; r++) { let a = 0; for (let i = 0; i < this.inDim; i++) a += this.weight[r * this.inDim + i] * x[i]; o[r] = a; } return o; };
const cfg = { vocab: 20, dim: 12, layers: 2 };
const m = new M.LanguageModel(cfg);
for (const b of m.blocks) { for (let i = 0; i < b.W_x_w.weight.length; i++) b.W_x_w.weight[i] *= 0.3; b.mu_r.fill(0.4); b.mu_k.fill(0.6); }
const seq = [3, 8, 14], target = 5;
const ce = lg => { let mx = -1e9; for (const v of lg) mx = Math.max(mx, v); let s = 0; for (const v of lg) s += Math.exp(v - mx); return -(lg[target] - mx - Math.log(s)); };
const run = () => { let st = m.getEmptyState(); const o = []; for (const t of seq) { const r = m.forward(t, st); st = r.state; o.push(r); } return o; };
const loss = () => ce(run().at(-1).logits);
m.zeroGrad(); m.embGrad.fill(0);
const outs = run();
const p = Float32Array.from(outs.at(-1).logits); let mx = -1e9; for (const v of p) mx = Math.max(mx, v); let s = 0;
for (let i = 0; i < p.length; i++) { p[i] = Math.exp(p[i] - mx); s += p[i]; } for (let i = 0; i < p.length; i++) p[i] /= s; p[target] -= 1;
let g = m.backward(p, outs.at(-1).caches, null).gradStatesOut;
for (let t = outs.length - 2; t >= 0; t--) g = m.backward(new Float32Array(cfg.vocab), outs[t].caches, g).gradStatesOut;
// Finite differences are only trustworthy when wkv = num/|den| is well conditioned.
// If any |den| is tiny, a FAIL below may be the measuring stick, not the code.
const minDen = Math.min(...outs.flatMap(o => o.caches.blocks.map(c => Math.min(...Array.from(c.den_new, Math.abs)))));
console.log(`min |den| = ${minDen.toExponential(1)}${minDen < 5e-3 ? '  (WARNING: ill-conditioned — pick another seed)' : ''}`);
const eps = 1e-3; let fails = 0, worst = 1;
function check(label, W, G) {
  let dot = 0, na = 0, nn = 0;
  for (let i = 0; i < W.length; i++) { const o = W[i]; W[i] = o + eps; const a = loss(); W[i] = o - eps; const b = loss(); W[i] = o; const num = (a - b) / (2 * eps); dot += G[i] * num; na += G[i] * G[i]; nn += num * num; }
  if (nn < 1e-14) { console.log(`${label.padEnd(16)} (unused by the forward pass)`); return; }
  const cos = dot / Math.sqrt(na * nn), ratio = Math.sqrt(na / nn), ok = cos > 0.99 && Math.abs(ratio - 1) < 0.05;
  if (!ok) fails++; worst = Math.min(worst, cos);
  console.log(`${label.padEnd(16)} cos=${cos.toFixed(4)}  |analytic|/|true|=${ratio.toFixed(4)}  ${ok ? 'ok' : 'FAIL'}`);
}
for (const L of [1, 0]) {
  for (const n of ['W_x_r', 'W_x_k', 'W_x_v', 'W_x_kappa', 'W_x_a', 'W_x_w', 'W_o', 'W_g', 'W_x_rx2', 'W_x_kx2', 'Wr2', 'Wk2', 'Wv2']) check(`L${L}.${n}`, m.blocks[L][n].weight, m.blocks[L][n].grad);
  for (const mu of ['mu_r', 'mu_k', 'mu_v', 'mu_w']) check(`L${L}.${mu}`, m.blocks[L][mu], m.blocks[L][mu + '_grad']);
  check(`L${L}.ln1.w`, m.blocks[L].ln1.weight, m.blocks[L].ln1.grad_w || m.blocks[L].ln1.gradW || m.blocks[L].ln1.grad);
}
check('head', m.head.weight, m.head.grad); check('embedding', m.embedding, m.embGrad);
console.log(fails ? `\n${fails} FAILED (worst cos ${worst.toFixed(4)})` : `\nALL OK (worst cos ${worst.toFixed(4)})`);
process.exit(fails ? 1 : 0);
