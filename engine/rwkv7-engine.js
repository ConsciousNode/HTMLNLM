// ==== RWKV7 ENGINE (real RWKV-7 "Goose" x070) ====
// MIT License, ConsciousNode SoftWorks. Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim.
// Canonical source: HTMLNLM/engine/rwkv7-engine.js, embedded verbatim into the pip-suite pages by engine/embed.py.
//
// RWKV7: copied from Brymar College v4.0.0 model/rwkv7.js (Kehai Interim, 2026-10-11): line-for-line the
// official math (BlinkDL's `rwkv` package), per-head matrix memory, official tensor names, exact backward + LoRA.
// Two HTMLNLM changes, both in the constructor: opts.tensors adopts loaded weights without copying, and
// opts.inference skips gradient buffers. tests/engine-ref.js checks the embedded copies against the official math.
// Also here: the RWKV World tokenizer (vocab rwkv_vocab_v20230424, Apache-2.0, BlinkDL/RWKV-LM),
// and browser loaders for .safetensors and PyTorch .pth (zip + pickle, no PyTorch needed).

const SIG = z => 1 / (1 + Math.exp(-z));
const W_SCALE = 0.606531;               // exp(-0.5), official constant

class RWKV7 {
  /**
   * @param {object} cfg  { vocabSize, dim, numLayers, headSize, dimFFN, dDecay, dAAA, dMV, dGate }
   * @param {object} [opts] { Float: Float32Array | Float64Array, tensors?: {name:{data}}, inference?: bool }
   */
  constructor(cfg, opts = {}) {
    const F = this.Float = opts.Float || Float32Array;
    const C = this.C = cfg.dim, N = this.N = cfg.headSize || 64, H = this.H = C / N, L = this.L = cfg.numLayers, V = this.V = cfg.vocabSize;
    if (!Number.isInteger(H)) throw new Error(`dim ${C} is not a multiple of headSize ${N}`);
    const FF = this.FF = cfg.dimFFN || 4 * C;
    const DW = cfg.dDecay || 64, DA = cfg.dAAA || 64, DV = cfg.dMV || 32, DG = cfg.dGate || 128;
    this.cfg = { ...cfg, headSize: N, dimFFN: FF, dDecay: DW, dAAA: DA, dMV: DV, dGate: DG };
    this.p = {}; this.shape = {}; this.g = {};
    const def = (name, shape, init) => {
      const n = shape.reduce((a, b) => a * b, 1), given = opts.tensors && opts.tensors[name];
      let arr;
      if (given && given.data.length === n) arr = given.data instanceof F ? given.data : F.from(given.data);   // HTMLNLM: adopt loaded weights, no copy
      else { arr = new F(n); if (init) for (let i = 0; i < n; i++) arr[i] = init(i); }
      this.p[name] = arr; this.shape[name] = shape; if (!opts.inference) this.g[name] = new F(n);   // HTMLNLM: inference needs no gradient buffers
    };
    const rnd = s => () => (Math.random() * 2 - 1) * s, one = () => 1, half = () => 0.5;
    def('emb.weight', [V, C], rnd(1e-2));
    def('blocks.0.ln0.weight', [C], one); def('blocks.0.ln0.bias', [C]);
    for (let l = 0; l < L; l++) {
      const b = `blocks.${l}.`, a = b + 'att.', f = b + 'ffn.';
      def(b + 'ln1.weight', [C], one); def(b + 'ln1.bias', [C]); def(b + 'ln2.weight', [C], one); def(b + 'ln2.bias', [C]);
      for (const x of ['x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g']) def(a + x, [1, 1, C], half);
      def(a + 'w0', [1, 1, C], () => -1); def(a + 'w1', [C, DW]); def(a + 'w2', [DW, C], rnd(0.1));
      def(a + 'a0', [1, 1, C]); def(a + 'a1', [C, DA]); def(a + 'a2', [DA, C], rnd(0.1));
      if (l > 0) { def(a + 'v0', [1, 1, C], one); def(a + 'v1', [C, DV]); def(a + 'v2', [DV, C], rnd(0.1)); }
      def(a + 'g1', [C, DG]); def(a + 'g2', [DG, C], rnd(0.1));
      def(a + 'k_k', [1, 1, C], () => 0.85); def(a + 'k_a', [1, 1, C], one); def(a + 'r_k', [H, N]);
      for (const m of ['receptance', 'key', 'value', 'output']) def(a + m + '.weight', [C, C], rnd(Math.sqrt(3 / C)));
      def(a + 'ln_x.weight', [C], one); def(a + 'ln_x.bias', [C]);
      def(f + 'x_k', [1, 1, C], half); def(f + 'key.weight', [FF, C], rnd(Math.sqrt(3 / C))); def(f + 'value.weight', [C, FF], rnd(Math.sqrt(3 / FF)));
    }
    def('ln_out.weight', [C], one); def('ln_out.bias', [C]); def('head.weight', [V, C], rnd(Math.sqrt(3 / C)));
    this.lora = {};          // name -> { A [r, in], B [out, r], gA, gB, scale, r, inD, outD }
    this.frozen = false;     // true: base weights get no gradient (LoRA-only training)
  }

  // ── LoRA ─────────────────────────────────────────────────────────────────
  injectLoRA(rank = 4, alpha = 8, targets = ['receptance', 'key', 'value', 'output']) {
    const F = this.Float;
    for (let l = 0; l < this.L; l++) for (const t of targets) {
      const name = `blocks.${l}.att.${t}.weight`, [outD, inD] = this.shape[name], A = new F(rank * inD), sA = Math.sqrt(1 / inD);
      for (let i = 0; i < A.length; i++) A[i] = (Math.random() * 2 - 1) * sA;
      this.lora[name] = { A, B: new F(outD * rank), gA: new F(rank * inD), gB: new F(outD * rank), scale: alpha / rank, r: rank, inD, outD };
    }
    this.frozen = true;
    this.g = {};                       // frozen base weights need no gradient buffers (saves a full model's worth of RAM)
  }
  /** Fold LoRA deltas into the base weights (for export) and remove the adapters. */
  mergeLoRA() {
    for (const [name, a] of Object.entries(this.lora)) {
      const W = this.p[name];
      for (let o = 0; o < a.outD; o++) for (let i = 0; i < a.inD; i++) { let s = 0; for (let k = 0; k < a.r; k++) s += a.B[o * a.r + k] * a.A[k * a.inD + i]; W[o * a.inD + i] += a.scale * s; }
    }
    this.lora = {}; this.frozen = false;
    for (const [n, w] of Object.entries(this.p)) this.g[n] = new this.Float(w.length);
  }

  // ── parameter / gradient access (for optimizers, checkpoints) ────────────
  /** Trainable tensors as { name, weight, grad, rows, cols } (base weights unless frozen, plus LoRA). */
  trainable() {
    const out = [];
    if (!this.frozen) for (const [name, w] of Object.entries(this.p)) { const s = this.shape[name]; out.push({ name, weight: w, grad: this.g[name], rows: s.length > 1 ? s[s.length - 2] : 1, cols: s[s.length - 1] }); }
    for (const [name, a] of Object.entries(this.lora)) {
      out.push({ name: name + '.lora_A', weight: a.A, grad: a.gA, rows: a.r, cols: a.inD });
      out.push({ name: name + '.lora_B', weight: a.B, grad: a.gB, rows: a.outD, cols: a.r });
    }
    return out;
  }
  zeroGrad() { for (const g of Object.values(this.g)) g.fill(0); for (const a of Object.values(this.lora)) { a.gA.fill(0); a.gB.fill(0); } }

  getEmptyState() {
    const F = this.Float, s = [];
    for (let l = 0; l < this.L; l++) s.push({ xa: new F(this.C), S: new F(this.H * this.N * this.N), xf: new F(this.C) });
    return s;
  }
  cloneState(st) { return st.map(s => ({ xa: s.xa.slice(), S: s.S.slice(), xf: s.xf.slice() })); }

  // ── small dense helpers ──────────────────────────────────────────────────
  _lin(name, x) {                               // y = W·x (+ LoRA), W [out, in]
    const W = this.p[name], [outD, inD] = this.shape[name], y = new this.Float(outD);
    for (let o = 0; o < outD; o++) { let s = 0; const r = o * inD; for (let i = 0; i < inD; i++) s += W[r + i] * x[i]; y[o] = s; }
    const a = this.lora[name];
    if (a) { const ax = new this.Float(a.r); for (let k = 0; k < a.r; k++) { let s = 0; for (let i = 0; i < inD; i++) s += a.A[k * inD + i] * x[i]; ax[k] = s; }
      for (let o = 0; o < outD; o++) { let s = 0; for (let k = 0; k < a.r; k++) s += a.B[o * a.r + k] * ax[k]; y[o] += a.scale * s; } }
    return y;
  }
  _linBack(name, x, gy) {                       // returns dL/dx; accumulates dL/dW (unless frozen) and LoRA grads
    const W = this.p[name], G = this.g[name], [outD, inD] = this.shape[name], gx = new this.Float(inD);
    for (let o = 0; o < outD; o++) { const go = gy[o]; if (go === 0) continue; const r = o * inD;
      for (let i = 0; i < inD; i++) { gx[i] += W[r + i] * go; if (!this.frozen) G[r + i] += go * x[i]; } }
    const a = this.lora[name];
    if (a) {
      const ax = new this.Float(a.r); for (let k = 0; k < a.r; k++) { let s = 0; for (let i = 0; i < inD; i++) s += a.A[k * inD + i] * x[i]; ax[k] = s; }
      const gax = new this.Float(a.r);
      for (let o = 0; o < outD; o++) { const go = a.scale * gy[o]; if (go === 0) continue; for (let k = 0; k < a.r; k++) { a.gB[o * a.r + k] += go * ax[k]; gax[k] += a.B[o * a.r + k] * go; } }
      for (let k = 0; k < a.r; k++) { const gk = gax[k]; if (gk === 0) continue; for (let i = 0; i < inD; i++) { a.gA[k * inD + i] += gk * x[i]; gx[i] += a.A[k * inD + i] * gk; } }
    }
    return gx;
  }
  _xM(name, x) {                                // h = x·M, M [C, D]
    const M = this.p[name], [Cc, D] = this.shape[name], h = new this.Float(D);
    for (let c = 0; c < Cc; c++) { const xc = x[c]; if (xc === 0) continue; const r = c * D; for (let d = 0; d < D; d++) h[d] += xc * M[r + d]; }
    return h;
  }
  _xMBack(name, x, gh, gx) {                    // accumulates into gx, and dL/dM unless frozen
    const M = this.p[name], G = this.g[name], [Cc, D] = this.shape[name];
    for (let c = 0; c < Cc; c++) { const r = c * D; let s = 0; for (let d = 0; d < D; d++) { s += M[r + d] * gh[d]; if (!this.frozen) G[r + d] += x[c] * gh[d]; } gx[c] += s; }
  }
  _ln(x, wn, bn, eps = 1e-5) {
    const n = x.length, w = this.p[wn], b = this.p[bn]; let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) v += (x[i] - m) ** 2; v /= n;
    const rstd = 1 / Math.sqrt(v + eps), xh = new this.Float(n), y = new this.Float(n);
    for (let i = 0; i < n; i++) { xh[i] = (x[i] - m) * rstd; y[i] = xh[i] * w[i] + b[i]; }
    return { y, c: { xh, rstd, wn, bn } };
  }
  _lnBack(gy, c) {
    const n = gy.length, w = this.p[c.wn], gw = this.g[c.wn], gb = this.g[c.bn], gxh = new this.Float(n); let m1 = 0, m2 = 0;
    for (let i = 0; i < n; i++) { if (!this.frozen) { gw[i] += gy[i] * c.xh[i]; gb[i] += gy[i]; } gxh[i] = gy[i] * w[i]; m1 += gxh[i]; m2 += gxh[i] * c.xh[i]; }
    m1 /= n; m2 /= n; const gx = new this.Float(n);
    for (let i = 0; i < n; i++) gx[i] = c.rstd * (gxh[i] - m1 - c.xh[i] * m2);
    return gx;
  }

  // ── time mix (one token, one layer) ──────────────────────────────────────
  _tmix(l, x, xp, vFirst, S0) {
    const F = this.Float, C = this.C, H = this.H, N = this.N, a = `blocks.${l}.att.`, P = this.p, c = { l, x, xp };
    const mix = n => { const m = P[a + n], o = new F(C); for (let i = 0; i < C; i++) o[i] = x[i] + (xp[i] - x[i]) * m[i]; return o; };
    const xr = mix('x_r'), xw = mix('x_w'), xk = mix('x_k'), xv = mix('x_v'), xa = mix('x_a'), xg = mix('x_g');
    const r = this._lin(a + 'receptance.weight', xr), k0 = this._lin(a + 'key.weight', xk), vb = this._lin(a + 'value.weight', xv);
    const hw = this._xM(a + 'w1', xw), tw = hw.map(Math.tanh), wl = this._xM(a + 'w2', tw);
    const ha = this._xM(a + 'a1', xa), aa = this._xM(a + 'a2', ha), av = new F(C); for (let i = 0; i < C; i++) av[i] = SIG(P[a + 'a0'][i] + aa[i]);
    const hg = this._xM(a + 'g1', xg), sg = hg.map(SIG), g = this._xM(a + 'g2', sg);
    // kk = normalize(k0 * k_k) per head (F.normalize, eps 1e-12)
    const kkr = new F(C), kk = new F(C), knrm = new F(H);
    for (let i = 0; i < C; i++) kkr[i] = k0[i] * P[a + 'k_k'][i];
    for (let h = 0; h < H; h++) { let s = 0; for (let j = 0; j < N; j++) s += kkr[h * N + j] ** 2; knrm[h] = Math.max(Math.sqrt(s), 1e-12); for (let j = 0; j < N; j++) kk[h * N + j] = kkr[h * N + j] / knrm[h]; }
    const k = new F(C); for (let i = 0; i < C; i++) k[i] = k0[i] * (1 + (av[i] - 1) * P[a + 'k_a'][i]);
    let v, gate = null, hv = null;
    if (l === 0) { v = vb; vFirst = vb; }
    else { hv = this._xM(a + 'v1', xv); const vv = this._xM(a + 'v2', hv); gate = new F(C); v = new F(C);
      for (let i = 0; i < C; i++) { gate[i] = SIG(P[a + 'v0'][i] + vv[i]); v[i] = vb[i] + (vFirst[i] - vb[i]) * gate[i]; } }
    const sw = new F(C), w = new F(C); for (let i = 0; i < C; i++) { sw[i] = SIG(P[a + 'w0'][i] + wl[i]); w[i] = Math.exp(-W_SCALE * sw[i]); }
    // memory: S = S·diag(w) + S·ab + v kᵀ,  ab = (−kk)(kk∘a)ᵀ  ⇒  (S·ab)[i][j] = −(S·kk)[i] · kk[j]·a[j]
    const S = new F(H * N * N), q = new F(C), o = new F(C);
    for (let h = 0; h < H; h++) {
      const base = h * N * N, hb = h * N;
      for (let i = 0; i < N; i++) { let s = 0; for (let m = 0; m < N; m++) s += S0[base + i * N + m] * kk[hb + m]; q[hb + i] = s; }
      for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) {
        const val = S0[base + i * N + j] * w[hb + j] - q[hb + i] * kk[hb + j] * av[hb + j] + v[hb + i] * k[hb + j];
        S[base + i * N + j] = val; s += val * r[hb + j]; } o[hb + i] = s; }
    }
    // group norm per head (eps 64e-5), then the "bonus" term, then gate and output projection
    const lw = P[a + 'ln_x.weight'], lb = P[a + 'ln_x.bias'], on = new F(C), grstd = new F(H), y = new F(C), bon = new F(H);
    for (let h = 0; h < H; h++) { const hb = h * N; let m = 0; for (let j = 0; j < N; j++) m += o[hb + j]; m /= N;
      let vr = 0; for (let j = 0; j < N; j++) vr += (o[hb + j] - m) ** 2; vr /= N; grstd[h] = 1 / Math.sqrt(vr + 64e-5);
      for (let j = 0; j < N; j++) { on[hb + j] = (o[hb + j] - m) * grstd[h]; y[hb + j] = on[hb + j] * lw[hb + j] + lb[hb + j]; }
      let s = 0; for (let j = 0; j < N; j++) s += r[hb + j] * k[hb + j] * P[a + 'r_k'][hb + j]; bon[h] = s;
      for (let j = 0; j < N; j++) y[hb + j] += s * v[hb + j]; }
    const z = new F(C); for (let i = 0; i < C; i++) z[i] = y[i] * g[i];
    const out = this._lin(a + 'output.weight', z);
    Object.assign(c, { xr, xw, xk, xv, xa, xg, r, k0, vb, hw, tw, ha, av, hg, sg, g, kkr, kk, knrm, k, v, gate, hv, vFirst, sw, w, S0, S, q, on, grstd, bon, y, z });
    return { out, S, vFirst, c };
  }

  _tmixBack(c, gout, gS, gvFirst) {
    // gout: dL/d(out); gS: dL/d(S_new) from t+1 (or null); gvFirst: accumulated dL/d(v_first) from layers above
    // returns { gx, gxp, gS0 }; for layer 0 it consumes gvFirst; for l > 0 it adds into gvFirst
    const F = this.Float, C = this.C, H = this.H, N = this.N, a = `blocks.${c.l}.att.`, P = this.p, G = this.g, fr = this.frozen;
    const gz = this._linBack(a + 'output.weight', c.z, gout);
    const gy = new F(C), gg = new F(C); for (let i = 0; i < C; i++) { gy[i] = gz[i] * c.g[i]; gg[i] = gz[i] * c.y[i]; }
    const gr = new F(C), gk = new F(C), gv = new F(C), go = new F(C);
    const lw = P[a + 'ln_x.weight'];
    for (let h = 0; h < H; h++) { const hb = h * N;
      let gbon = 0; for (let j = 0; j < N; j++) { gbon += gy[hb + j] * c.v[hb + j]; gv[hb + j] += gy[hb + j] * c.bon[h]; }
      for (let j = 0; j < N; j++) { const rk = P[a + 'r_k'][hb + j]; gr[hb + j] += gbon * c.k[hb + j] * rk; gk[hb + j] += gbon * c.r[hb + j] * rk; if (!fr) G[a + 'r_k'][hb + j] += gbon * c.r[hb + j] * c.k[hb + j]; }
      let m1 = 0, m2 = 0; const gon = new F(N);
      for (let j = 0; j < N; j++) { if (!fr) { G[a + 'ln_x.weight'][hb + j] += gy[hb + j] * c.on[hb + j]; G[a + 'ln_x.bias'][hb + j] += gy[hb + j]; }
        gon[j] = gy[hb + j] * lw[hb + j]; m1 += gon[j]; m2 += gon[j] * c.on[hb + j]; }
      m1 /= N; m2 /= N; for (let j = 0; j < N; j++) go[hb + j] = c.grstd[h] * (gon[j] - m1 - c.on[hb + j] * m2);
    }
    // through the memory: o[i] = Σ_j S[i][j] r[j]
    const gS0 = new F(H * N * N), gw = new F(C), gkk = new F(C), ga = new F(C);
    for (let h = 0; h < H; h++) { const base = h * N * N, hb = h * N, gq = new F(N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const idx = base + i * N + j, gSn = go[hb + i] * c.r[hb + j] + (gS ? gS[idx] : 0);
        gr[hb + j] += c.S[idx] * go[hb + i];
        if (gSn === 0) continue;
        gS0[idx] += gSn * c.w[hb + j];
        gw[hb + j] += gSn * c.S0[idx];
        const b = c.kk[hb + j] * c.av[hb + j];
        gq[i] -= gSn * b;
        const gb = -gSn * c.q[hb + i]; gkk[hb + j] += gb * c.av[hb + j]; ga[hb + j] += gb * c.kk[hb + j];
        gv[hb + i] += gSn * c.k[hb + j]; gk[hb + j] += gSn * c.v[hb + i];
      }
      for (let i = 0; i < N; i++) { const g_ = gq[i]; if (g_ === 0) continue; for (let m = 0; m < N; m++) { gS0[base + i * N + m] += g_ * c.kk[hb + m]; gkk[hb + m] += g_ * c.S0[base + i * N + m]; } }
    }
    // decay: w = exp(-c·sigmoid(w0 + tanh(xw·w1)·w2))
    const gu = new F(C); for (let i = 0; i < C; i++) gu[i] = gw[i] * c.w[i] * (-W_SCALE) * c.sw[i] * (1 - c.sw[i]);
    if (!fr) for (let i = 0; i < C; i++) G[a + 'w0'][i] += gu[i];
    const gxw = new F(C), gtw = new F(c.tw.length); this._xMBack(a + 'w2', c.tw, gu, gtw);
    const ghw = gtw.map((gt, d) => gt * (1 - c.tw[d] * c.tw[d])); this._xMBack(a + 'w1', c.xw, ghw, gxw);
    // value residual
    const gvb = new F(C), gxv = new F(C);
    if (c.l === 0) { for (let i = 0; i < C; i++) gvb[i] = gv[i] + gvFirst[i]; }
    else { const gpre = new F(C);
      for (let i = 0; i < C; i++) { gvb[i] = gv[i] * (1 - c.gate[i]); gvFirst[i] += gv[i] * c.gate[i]; gpre[i] = gv[i] * (c.vFirst[i] - c.vb[i]) * c.gate[i] * (1 - c.gate[i]); }
      if (!fr) for (let i = 0; i < C; i++) G[a + 'v0'][i] += gpre[i];
      const ghv = new F(c.hv.length); this._xMBack(a + 'v2', c.hv, gpre, ghv); this._xMBack(a + 'v1', c.xv, ghv, gxv); }
    // k = k0·(1 + (a−1)·k_a);  kk = normalize(k0·k_k)
    const gk0 = new F(C);
    for (let i = 0; i < C; i++) { const ka = P[a + 'k_a'][i]; gk0[i] += gk[i] * (1 + (c.av[i] - 1) * ka); ga[i] += gk[i] * c.k0[i] * ka; if (!fr) G[a + 'k_a'][i] += gk[i] * c.k0[i] * (c.av[i] - 1); }
    for (let h = 0; h < H; h++) { const hb = h * N; let dot = 0; for (let j = 0; j < N; j++) dot += c.kk[hb + j] * gkk[hb + j];
      for (let j = 0; j < N; j++) { const gkr = (gkk[hb + j] - c.kk[hb + j] * dot) / c.knrm[h]; gk0[hb + j] += gkr * P[a + 'k_k'][hb + j]; if (!fr) G[a + 'k_k'][hb + j] += gkr * c.k0[hb + j]; } }
    // a = sigmoid(a0 + xa·a1·a2)
    const gpa = new F(C); for (let i = 0; i < C; i++) gpa[i] = ga[i] * c.av[i] * (1 - c.av[i]);
    if (!fr) for (let i = 0; i < C; i++) G[a + 'a0'][i] += gpa[i];
    const gha = new F(c.ha.length), gxa = new F(C); this._xMBack(a + 'a2', c.ha, gpa, gha); this._xMBack(a + 'a1', c.xa, gha, gxa);
    // g = sigmoid(xg·g1)·g2
    const gsg = new F(c.sg.length), gxg = new F(C); this._xMBack(a + 'g2', c.sg, gg, gsg);
    const ghg = gsg.map((s, d) => s * c.sg[d] * (1 - c.sg[d])); this._xMBack(a + 'g1', c.xg, ghg, gxg);
    // the four big projections
    const gxr = this._linBack(a + 'receptance.weight', c.xr, gr), gxk = this._linBack(a + 'key.weight', c.xk, gk0), gxv2 = this._linBack(a + 'value.weight', c.xv, gvb);
    for (let i = 0; i < C; i++) gxv[i] += gxv2[i];
    // token shift: x_* = x + (xp − x)·m
    const gx = new F(C), gxp = new F(C);
    for (const [n, gsrc] of [['x_r', gxr], ['x_w', gxw], ['x_k', gxk], ['x_v', gxv], ['x_a', gxa], ['x_g', gxg]]) {
      const m = P[a + n], gm = G[a + n];
      for (let i = 0; i < C; i++) { gx[i] += gsrc[i] * (1 - m[i]); gxp[i] += gsrc[i] * m[i]; if (!fr) gm[i] += gsrc[i] * (c.xp[i] - c.x[i]); }
    }
    return { gx, gxp, gS0 };
  }

  // ── channel mix ──────────────────────────────────────────────────────────
  _cmix(l, x, xp) {
    const F = this.Float, C = this.C, f = `blocks.${l}.ffn.`, m = this.p[f + 'x_k'], kin = new F(C);
    for (let i = 0; i < C; i++) kin[i] = x[i] + (xp[i] - x[i]) * m[i];
    const hk = this._lin(f + 'key.weight', kin), rel = hk.map(z => z > 0 ? z * z : 0);
    return { out: this._lin(f + 'value.weight', rel), c: { l, x, xp, kin, hk, rel } };
  }
  _cmixBack(c, gout) {
    const F = this.Float, C = this.C, f = `blocks.${c.l}.ffn.`;
    const grel = this._linBack(f + 'value.weight', c.rel, gout), ghk = grel.map((g, i) => c.hk[i] > 0 ? g * 2 * c.hk[i] : 0);
    const gkin = this._linBack(f + 'key.weight', c.kin, ghk), m = this.p[f + 'x_k'], gm = this.g[f + 'x_k'], gx = new F(C), gxp = new F(C);
    for (let i = 0; i < C; i++) { gx[i] = gkin[i] * (1 - m[i]); gxp[i] = gkin[i] * m[i]; if (!this.frozen) gm[i] += gkin[i] * (c.xp[i] - c.x[i]); }
    return { gx, gxp };
  }

  // ── full model, one token ────────────────────────────────────────────────
  forward(tokenId, state) {
    const C = this.C, E = this.p['emb.weight'], e = E.subarray(tokenId * C, tokenId * C + C);
    const ln0 = this._ln(e, 'blocks.0.ln0.weight', 'blocks.0.ln0.bias');
    let x = ln0.y, vFirst = null; const layers = [], newState = [];
    for (let l = 0; l < this.L; l++) {
      const b = `blocks.${l}.`, st = state[l];
      const l1 = this._ln(x, b + 'ln1.weight', b + 'ln1.bias');
      const tm = this._tmix(l, l1.y, st.xa, vFirst, st.S); vFirst = tm.vFirst;
      const x1 = x.map((v, i) => v + tm.out[i]);
      const l2 = this._ln(x1, b + 'ln2.weight', b + 'ln2.bias');
      const cm = this._cmix(l, l2.y, st.xf);
      x = x1.map((v, i) => v + cm.out[i]);
      layers.push({ l1: l1.c, tm: tm.c, l2: l2.c, cm: cm.c });
      newState.push({ xa: l1.y, S: tm.S, xf: l2.y });
    }
    const lo = this._ln(x, 'ln_out.weight', 'ln_out.bias');
    const logits = this._lin('head.weight', lo.y);
    return { logits, state: newState, cache: { tokenId, ln0: ln0.c, layers, lo: lo.c, xo: lo.y } };
  }

  /**
   * @param {ArrayLike<number>} gLogits dL/dlogits for this token (may be all zeros)
   * @param {object} cache from forward()
   * @param {Array|null} gState dL/d(state produced by this token): [{ xa, S, xf }] per layer, or null
   * @returns {Array} dL/d(state consumed by this token), same layout, for the previous token
   */
  backward(gLogits, cache, gState = null) {
    const F = this.Float, C = this.C;
    let gx = this._lnBack(this._linBack('head.weight', cache.xo, gLogits), cache.lo);
    const gvFirst = new F(C), out = new Array(this.L);
    for (let l = this.L - 1; l >= 0; l--) {
      const lc = cache.layers[l], gs = gState ? gState[l] : null;
      const cm = this._cmixBack(lc.cm, gx);
      const ga2 = cm.gx; if (gs) for (let i = 0; i < C; i++) ga2[i] += gs.xf[i];
      const gx1 = this._lnBack(ga2, lc.l2); for (let i = 0; i < C; i++) gx1[i] += gx[i];
      const tm = this._tmixBack(lc.tm, gx1, gs ? gs.S : null, gvFirst);
      const ga1 = tm.gx; if (gs) for (let i = 0; i < C; i++) ga1[i] += gs.xa[i];
      const gx0 = this._lnBack(ga1, lc.l1); for (let i = 0; i < C; i++) gx0[i] += gx1[i];
      gx = gx0;
      out[l] = { xa: tm.gxp, S: tm.gS0, xf: cm.gxp };
    }
    const ge = this._lnBack(gx, cache.ln0);
    if (!this.frozen) { const G = this.g['emb.weight'], o = cache.tokenId * C; for (let i = 0; i < C; i++) G[o + i] += ge[i]; }
    return out;
  }
}

// ── RWKV World tokenizer (greedy longest match over UTF-8 bytes, like BlinkDL's TRIE_TOKENIZER) ──
class RWKVWorldTokenizer {
  /** @param {string[]} tokens hex-encoded byte strings, index = token id (rwkv_world_vocab.json "tokens") */
  constructor(tokens) {
    this.size = tokens.length;
    this.idx2bytes = tokens.map(h => h ? Uint8Array.from(h.match(/../g).map(b => parseInt(b, 16))) : null);
    this.root = { to: new Map(), id: -1 };
    this.idx2bytes.forEach((bytes, id) => {
      if (!bytes) return;
      let node = this.root;
      for (const b of bytes) { let nx = node.to.get(b); if (!nx) { nx = { to: new Map(), id: -1 }; node.to.set(b, nx); } node = nx; }
      node.id = id;
    });
  }
  encodeBytes(src) {
    const out = []; let i = 0;
    while (i < src.length) {
      let node = this.root, j = i, bestId = -1, bestEnd = i;
      while (j < src.length) { const nx = node.to.get(src[j]); if (!nx) break; node = nx; j++; if (node.id >= 0) { bestId = node.id; bestEnd = j; } }
      if (bestId < 0) throw new Error(`byte ${src[i]} at ${i} has no token`);
      out.push(bestId); i = bestEnd;
    }
    return out;
  }
  encode(text) { return this.encodeBytes(new TextEncoder().encode(text)); }
  decodeBytes(ids) { const parts = ids.map(id => this.idx2bytes[id] || new Uint8Array(0)), n = parts.reduce((a, p) => a + p.length, 0), out = new Uint8Array(n); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; } return out; }
  decode(ids) { return new TextDecoder('utf-8', { fatal: false }).decode(this.decodeBytes(ids)); }
}

// ── weight files → { name: { shape, data: Float32Array } } ──
function _halfToF32(u16, bf16) {
  const out = new Float32Array(u16.length);
  if (bf16) { const u32 = new Uint32Array(out.buffer); for (let i = 0; i < u16.length; i++) u32[i] = u16[i] << 16; return out; }
  for (let i = 0; i < u16.length; i++) { const h = u16[i], sg = h & 0x8000 ? -1 : 1, ex = (h >> 10) & 31, f = h & 1023;
    out[i] = ex === 0 ? sg * 2 ** -14 * (f / 1024) : ex === 31 ? (f ? NaN : sg * Infinity) : sg * 2 ** (ex - 15) * (1 + f / 1024); }
  return out;
}
function _toF32(bytes, dtype, name) {           // bytes: Uint8Array view (any alignment)
  const copy = bytes.slice();                    // own, aligned buffer
  if (dtype === 'F32') return new Float32Array(copy.buffer, 0, copy.length >> 2);
  if (dtype === 'BF16' || dtype === 'F16') return _halfToF32(new Uint16Array(copy.buffer, 0, copy.length >> 1), dtype === 'BF16');
  throw new Error(`unsupported dtype ${dtype} for ${name}`);
}
/** .safetensors (F32 / F16 / BF16) */
function readSafetensorsBuffer(ab) {
  const dv = new DataView(ab), hlen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 8, hlen))), base = 8 + hlen, out = {};
  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [s, e] = meta.data_offsets;
    out[name] = { shape: meta.shape, data: _toF32(new Uint8Array(ab, base + s, e - s), meta.dtype, name) };
  }
  return out;
}
/** Minimal ZIP reader for PyTorch archives (torch.save stores entries uncompressed). */
function _zipEntries(ab) {
  const u8 = new Uint8Array(ab), dv = new DataView(ab);
  let eocd = -1;
  for (let i = u8.length - 22; i >= Math.max(0, u8.length - 65557); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error('not a zip archive (no end-of-central-directory record)');
  let count = dv.getUint16(eocd + 10, true), cdOff = dv.getUint32(eocd + 16, true);
  if (cdOff === 0xffffffff || count === 0xffff) {            // ZIP64 (files > 4 GB)
    const loc = eocd - 20; if (dv.getUint32(loc, true) !== 0x07064b50) throw new Error('zip64 locator missing');
    const z = Number(dv.getBigUint64(loc + 8, true)); count = Number(dv.getBigUint64(z + 32, true)); cdOff = Number(dv.getBigUint64(z + 48, true));
  }
  const entries = {}, td = new TextDecoder();
  let p = cdOff;
  for (let n = 0; n < count; n++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('bad zip central directory');
    const method = dv.getUint16(p + 10, true), nlen = dv.getUint16(p + 28, true), xlen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true);
    let csize = dv.getUint32(p + 20, true), usize = dv.getUint32(p + 24, true), lho = dv.getUint32(p + 42, true);
    const name = td.decode(u8.subarray(p + 46, p + 46 + nlen));
    let x = p + 46 + nlen; const xe = x + xlen;
    while (x + 4 <= xe) {                                    // ZIP64 extra field
      const id = dv.getUint16(x, true), sz = dv.getUint16(x + 2, true); let q = x + 4;
      if (id === 1) { if (usize === 0xffffffff) { usize = Number(dv.getBigUint64(q, true)); q += 8; } if (csize === 0xffffffff) { csize = Number(dv.getBigUint64(q, true)); q += 8; } if (lho === 0xffffffff) { lho = Number(dv.getBigUint64(q, true)); } }
      x += 4 + sz;
    }
    if (method !== 0) throw new Error(`zip entry ${name} is compressed (method ${method}); torch.save archives are not`);
    const lnlen = dv.getUint16(lho + 26, true), lxlen = dv.getUint16(lho + 28, true), start = lho + 30 + lnlen + lxlen;
    entries[name] = { start, size: usize };
    p += 46 + nlen + xlen + clen;
  }
  return entries;
}
/** Minimal unpickler: just enough for torch state dicts (OrderedDict of _rebuild_tensor_v2 tensors). */
function _unpickleTorch(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), td = new TextDecoder();
  let p = 0; const stack = [], marks = [], memo = new Map();
  const MARK = Symbol('mark');
  const popMark = () => { const m = marks.pop(); const items = stack.splice(m); return items; };
  const glob = (mod, name) => ({ __global: `${mod}.${name}` });
  const call = (fn, args) => {
    const g = fn && fn.__global;
    if (g === 'torch._utils._rebuild_tensor_v2' || g === 'torch._utils._rebuild_tensor') return { __tensor: true, storage: args[0], offset: args[1], shape: args[2], stride: args[3] };
    if (g === 'collections.OrderedDict') return new Map();
    if (g === 'torch._utils._rebuild_parameter') return args[0];
    return { __obj: g, args };
  };
  for (;;) {
    const op = u8[p++];
    switch (op) {
      case 0x80: p++; break;                                    // PROTO
      case 0x95: p += 8; break;                                 // FRAME
      case 0x7d: stack.push(new Map()); break;                  // EMPTY_DICT
      case 0x5d: stack.push([]); break;                         // EMPTY_LIST
      case 0x29: stack.push([]); break;                         // EMPTY_TUPLE
      case 0x28: marks.push(stack.length); break;               // MARK
      case 0x71: memo.set(u8[p++], stack[stack.length - 1]); break;                       // BINPUT
      case 0x72: memo.set(dv.getUint32(p, true), stack[stack.length - 1]); p += 4; break;  // LONG_BINPUT
      case 0x94: memo.set(memo.size, stack[stack.length - 1]); break;                     // MEMOIZE
      case 0x68: stack.push(memo.get(u8[p++])); break;                                    // BINGET
      case 0x6a: stack.push(memo.get(dv.getUint32(p, true))); p += 4; break;              // LONG_BINGET
      case 0x58: { const n = dv.getUint32(p, true); p += 4; stack.push(td.decode(u8.subarray(p, p + n))); p += n; break; }   // BINUNICODE
      case 0x8c: { const n = u8[p++]; stack.push(td.decode(u8.subarray(p, p + n))); p += n; break; }                       // SHORT_BINUNICODE
      case 0x8d: { const n = Number(dv.getBigUint64(p, true)); p += 8; stack.push(td.decode(u8.subarray(p, p + n))); p += n; break; } // BINUNICODE8
      case 0x55: { const n = u8[p++]; stack.push(td.decode(u8.subarray(p, p + n))); p += n; break; }                       // SHORT_BINSTRING
      case 0x54: { const n = dv.getUint32(p, true); p += 4; stack.push(td.decode(u8.subarray(p, p + n))); p += n; break; }   // BINSTRING
      case 0x43: { const n = u8[p++]; stack.push(u8.slice(p, p + n)); p += n; break; }                                     // SHORT_BINBYTES
      case 0x42: { const n = dv.getUint32(p, true); p += 4; stack.push(u8.slice(p, p + n)); p += n; break; }                 // BINBYTES
      case 0x63: { let e = u8.indexOf(10, p); const mod = td.decode(u8.subarray(p, e)); p = e + 1; e = u8.indexOf(10, p); const name = td.decode(u8.subarray(p, e)); p = e + 1; stack.push(glob(mod, name)); break; } // GLOBAL
      case 0x93: { const name = stack.pop(), mod = stack.pop(); stack.push(glob(mod, name)); break; }                       // STACK_GLOBAL
      case 0x4a: stack.push(dv.getInt32(p, true)); p += 4; break;     // BININT
      case 0x4b: stack.push(u8[p++]); break;                          // BININT1
      case 0x4d: stack.push(dv.getUint16(p, true)); p += 2; break;    // BININT2
      case 0x8a: { const n = u8[p++]; let v = 0n; for (let i = 0; i < n; i++) v |= BigInt(u8[p + i]) << BigInt(8 * i); if (n && (u8[p + n - 1] & 0x80)) v -= 1n << BigInt(8 * n); p += n; stack.push(Number(v)); break; } // LONG1
      case 0x47: stack.push(dv.getFloat64(p, false)); p += 8; break;  // BINFLOAT (big-endian)
      case 0x4e: stack.push(null); break;                             // NONE
      case 0x88: stack.push(true); break;                             // NEWTRUE
      case 0x89: stack.push(false); break;                            // NEWFALSE
      case 0x74: stack.push(popMark()); break;                        // TUPLE
      case 0x85: stack.push([stack.pop()]); break;                    // TUPLE1
      case 0x86: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break; }               // TUPLE2
      case 0x87: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break; } // TUPLE3
      case 0x6c: stack.push(popMark()); break;                        // LIST
      case 0x61: { const v = stack.pop(); stack[stack.length - 1].push(v); break; }                    // APPEND
      case 0x65: { const items = popMark(); stack[stack.length - 1].push(...items); break; }           // APPENDS
      case 0x73: { const v = stack.pop(), k = stack.pop(), d = stack[stack.length - 1]; if (d instanceof Map) d.set(k, v); break; } // SETITEM
      case 0x75: { const items = popMark(), d = stack[stack.length - 1]; if (d instanceof Map) for (let i = 0; i < items.length; i += 2) d.set(items[i], items[i + 1]); break; } // SETITEMS
      case 0x51: { const pid = stack.pop(); stack.push({ __persistent: pid }); break; }               // BINPERSID
      case 0x52: { const args = stack.pop(), fn = stack.pop(); stack.push(call(fn, args)); break; }    // REDUCE
      case 0x81: { const args = stack.pop(), cls = stack.pop(); stack.push(call(cls, args)); break; }  // NEWOBJ
      case 0x62: stack.pop(); break;                                  // BUILD (state ignored)
      case 0x2e: return stack.pop();                                  // STOP
      default: throw new Error(`unsupported pickle opcode 0x${(op ?? 0).toString(16)} at ${p - 1}`);
    }
  }
}
/** PyTorch .pth / .bin (zip format, torch >= 1.6). Contiguous tensors in F32 / F16 / BF16. */
function readPthBuffer(ab) {
  const entries = _zipEntries(ab), names = Object.keys(entries), pkl = names.find(n => n.endsWith('data.pkl'));
  if (!pkl) throw new Error('not a PyTorch zip checkpoint (no data.pkl). Very old .pth files are not supported.');
  const prefix = pkl.slice(0, -'data.pkl'.length), u8 = new Uint8Array(ab);
  let root = _unpickleTorch(u8.subarray(entries[pkl].start, entries[pkl].start + entries[pkl].size));
  if (root instanceof Map && root.has('state_dict') && root.get('state_dict') instanceof Map) root = root.get('state_dict');
  if (!(root instanceof Map)) throw new Error('checkpoint is not a state dict');
  const DT = { FloatStorage: ['F32', 4], HalfStorage: ['F16', 2], BFloat16Storage: ['BF16', 2] }, out = {};
  for (const [name, t] of root) {
    if (!t || !t.__tensor) continue;
    const pid = t.storage.__persistent;            // ['storage', <global torch.XStorage>, key, location, numel]
    const cls = String(pid[1] && pid[1].__global || '').split('.').pop(), dt = DT[cls];
    if (!dt) throw new Error(`${name}: unsupported storage ${cls}`);
    const shape = t.shape, n = shape.reduce((a, b) => a * b, 1);
    let expect = 1; for (let i = shape.length - 1; i >= 0; i--) { if (shape[i] !== 1 && t.stride[i] !== expect) throw new Error(`${name}: non-contiguous tensor`); expect *= shape[i]; }
    const e = entries[`${prefix}data/${pid[2]}`]; if (!e) throw new Error(`${name}: storage ${pid[2]} missing from archive`);
    const off = e.start + t.offset * dt[1];
    out[name] = { shape, data: _toF32(new Uint8Array(ab, off, n * dt[1]), dt[0], name) };
  }
  return out;
}
/** Config from official tensor shapes (same rules as Brymar's configFromTensors). */
function rwkv7ConfigFromTensors(t) {
  const need = n => { if (!t[n]) throw new Error(`not an RWKV-7 (x070) checkpoint: missing ${n}`); return t[n].shape; };
  const [V, C] = need('emb.weight'), [H, N] = need('blocks.0.att.r_k');
  let L = 0; for (const k of Object.keys(t)) { const m = k.match(/^blocks\.(\d+)\./); if (m) L = Math.max(L, +m[1] + 1); }
  const sz = s => s[s.length - 1];
  return { vocabSize: V, dim: C, numLayers: L, headSize: N, dimFFN: need('blocks.0.ffn.key.weight')[0],
    dDecay: sz(need('blocks.0.att.w1')), dAAA: sz(need('blocks.0.att.a1')), dMV: L > 1 ? sz(need('blocks.1.att.v1')) : 32, dGate: sz(need('blocks.0.att.g1')) };
}
/** ArrayBuffer of a .safetensors or .pth → ready RWKV7 (inference) + load report. Nothing is skipped silently. */
function loadRWKV7FromBuffer(ab, fileName = '', opts = {}) {
  const isST = /\.safetensors$/i.test(fileName) || (!/\.(pth|bin|pt)$/i.test(fileName) && new DataView(ab).getUint32(0, true) !== 0x04034b50);
  const t = isST ? readSafetensorsBuffer(ab) : readPthBuffer(ab), cfg = rwkv7ConfigFromTensors(t);
  const model = new RWKV7(cfg, { tensors: t, inference: opts.inference !== false, Float: opts.Float });
  const unused = Object.keys(t).filter(n => !model.p[n]), missing = Object.keys(model.p).filter(n => !t[n]);
  const wrong = Object.keys(t).filter(n => model.p[n] && model.p[n].length !== t[n].data.length);
  if (wrong.length) throw new Error('tensor size mismatch: ' + wrong.join(', '));
  const nParams = Object.values(model.p).reduce((a, w) => a + w.length, 0);
  return { model, cfg, nParams, report: { loaded: Object.keys(t).length - unused.length, total: Object.keys(t).length, unused, missing } };
}
// ==== END RWKV7 ENGINE ====
