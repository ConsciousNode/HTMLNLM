// What a quantized GGUF costs, measured against the checkpoint it was made from: same text through both,
// perplexity, top-1 agreement and mean KL(pth || gguf) per token, plus RAM after load and the GGUF's vocab
// against ours. usage: node --max-old-space-size=6000 tests/real/gguf-vs-pth.js model.pth model.gguf
const fs = require('fs'), vm = require('vm'), path = require('path');
const ctx = { TextDecoder, TextEncoder, DataView, Uint8Array, Int8Array, Uint16Array, Uint32Array, Float32Array, Float64Array, BigInt, Map, Set, Math, JSON, Object, Number, String, Error, Array, Symbol, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../../engine/rwkv7-engine.js'), 'utf8') + '\nthis.E = { load: loadRWKV7FromBuffer, Tok: RWKVWorldTokenizer, readGGUFBuffer };', ctx);
const [PTH, GGUF] = process.argv.slice(2);
const ab = f => { const b = fs.readFileSync(f); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); };
const vocab = JSON.parse(fs.readFileSync(path.join(__dirname, '../../engine/rwkv_world_vocab.json'), 'utf8')).tokens;
const tok = new ctx.E.Tok(vocab);
const TEXT = `User: Tell me about frogs.

Assistant: Frogs are amphibians that live both in water and on land. Most frogs lay their eggs in ponds, and the eggs hatch into tadpoles, which breathe with gills and swim with long tails. Over a few weeks the tadpoles grow legs, lose their tails, and become adult frogs that breathe air. Many frogs sing at night in spring, and each species has its own call.`;
const ids = tok.encode(TEXT);
function run(file) {
  const r0 = process.memoryUsage().rss, buf = ab(file), { model, report } = ctx.E.load(buf, file), rss = process.memoryUsage().rss - r0;
  let st = model.getEmptyState(); const lps = [];
  for (let i = 0; i < ids.length; i++) { const r = model.forward(ids[i], st); st = r.state; lps.push(logSoftmax(r.logits)); }
  return { lps, rss, report };
}
function logSoftmax(l) { let m = -Infinity; for (const v of l) if (v > m) m = v; let s = 0; for (const v of l) s += Math.exp(v - m); const z = m + Math.log(s), o = new Float64Array(l.length); for (let i = 0; i < l.length; i++) o[i] = l[i] - z; return o; }
const argmax = a => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
const P = run(PTH); console.log(`pth:  ${(P.rss / 1e6).toFixed(0)} MB RAM after load`);
const G = run(GGUF); console.log(`gguf: ${(G.rss / 1e6).toFixed(0)} MB RAM after load, quant ${G.report.quant}`);
let nllP = 0, nllG = 0, agree = 0, kl = 0, n = 0;
for (let i = 0; i < ids.length - 1; i++) {
  const nx = ids[i + 1], p = P.lps[i], g = G.lps[i]; nllP -= p[nx]; nllG -= g[nx]; n++;
  if (argmax(p) === argmax(g)) agree++;
  let k = 0; for (let j = 0; j < p.length; j++) { const e = Math.exp(p[j]); if (e > 1e-12) k += e * (p[j] - g[j]); } kl += k;
}
console.log(`${n} next-token predictions on a ${ids.length}-token text`);
console.log(`perplexity  pth ${Math.exp(nllP / n).toFixed(3)}   gguf ${Math.exp(nllG / n).toFixed(3)}`);
console.log(`top-1 agreement ${(100 * agree / n).toFixed(1)}%   mean KL(pth||gguf) ${(kl / n).toFixed(4)} nats/token`);
// the GGUF's own vocab, in llama.cpp's escaped form, against rwkv_world_vocab.json (hex bytes)
const meta = ctx.E.readGGUFBuffer(ab(GGUF)).meta, gt = meta['tokenizer.ggml.tokens'];
const unesc = s => { const out = []; for (let i = 0; i < s.length;) { if (s[i] === '\\' && s[i + 1] === 'x') { out.push(parseInt(s.substr(i + 2, 2), 16)); i += 4; }
  else if (s[i] === '\\' && 'ntr\\'.includes(s[i + 1])) { out.push({ n: 10, t: 9, r: 13, '\\': 92 }[s[i + 1]]); i += 2; } else { const cp = s.codePointAt(i); out.push(...new TextEncoder().encode(String.fromCodePoint(cp))); i += cp > 0xffff ? 2 : 1; } } return out; };
let same = 0, diff = []; for (let i = 1; i < vocab.length; i++) { const a = vocab[i] || '', b = Buffer.from(unesc(gt[i] || '')).toString('hex'); if (a === b) same++; else if (diff.length < 5) diff.push([i, a, gt[i]]); }
console.log(`vocab: ${same}/${vocab.length - 1} tokens identical${diff.length ? ', first differences: ' + JSON.stringify(diff) : ''}`);
