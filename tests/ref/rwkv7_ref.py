# Reference RWKV-7 (x070) forward in numpy, transcribed line by line from BlinkDL's official
# implementation (pip package `rwkv`, rwkv/model.py: RWKV_x070_TMix_one / RWKV_x070_CMix_one / forward_one).
# Usage: python3 tests/ref/rwkv7_ref.py OUT_DIR
#   writes OUT_DIR/mini.safetensors (random weights, official names/shapes/layout)
#   and OUT_DIR/expected.json (token sequence + logits at every step), for tests/forward-ref.mjs.
import json, struct, sys, os
import numpy as np

rng = np.random.default_rng(7)
C, H, L, V, DW, DA, DV, DG = 64, 4, 3, 50, 16, 16, 8, 32
N, F = C // H, 4 * C
P = {}
def put(name, arr): P[name] = np.asarray(arr, dtype=np.float32)
put('emb.weight', rng.normal(0, 1, (V, C)))
for i in range(L):
    b, a, f = f'blocks.{i}.', f'blocks.{i}.att.', f'blocks.{i}.ffn.'
    if i == 0: put(b + 'ln0.weight', 1 + 0.1 * rng.normal(size=C)); put(b + 'ln0.bias', 0.1 * rng.normal(size=C))
    for ln in ('ln1', 'ln2'): put(b + ln + '.weight', 1 + 0.1 * rng.normal(size=C)); put(b + ln + '.bias', 0.1 * rng.normal(size=C))
    for x in ('x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g'): put(a + x, rng.uniform(0, 1, (1, 1, C)))
    put(a + 'w0', rng.uniform(-3, 0, (1, 1, C))); put(a + 'w1', 0.3 * rng.normal(size=(C, DW))); put(a + 'w2', 0.3 * rng.normal(size=(DW, C)))
    put(a + 'a0', rng.normal(0, 1, (1, 1, C))); put(a + 'a1', 0.3 * rng.normal(size=(C, DA))); put(a + 'a2', 0.3 * rng.normal(size=(DA, C)))
    if i > 0: put(a + 'v0', rng.normal(0, 1, (1, 1, C))); put(a + 'v1', 0.3 * rng.normal(size=(C, DV))); put(a + 'v2', 0.3 * rng.normal(size=(DV, C)))
    put(a + 'g1', 0.3 * rng.normal(size=(C, DG))); put(a + 'g2', 0.3 * rng.normal(size=(DG, C)))
    put(a + 'k_k', rng.uniform(0.5, 1.5, (1, 1, C))); put(a + 'k_a', rng.uniform(0, 1, (1, 1, C))); put(a + 'r_k', rng.normal(0, 0.5, (H, N)))
    for m in ('receptance', 'key', 'value', 'output'): put(a + m + '.weight', rng.normal(0, 1 / np.sqrt(C), (C, C)))   # [out, in]
    put(a + 'ln_x.weight', 1 + 0.1 * rng.normal(size=C)); put(a + 'ln_x.bias', 0.1 * rng.normal(size=C))
    put(f + 'x_k', rng.uniform(0, 1, (1, 1, C))); put(f + 'key.weight', rng.normal(0, 1 / np.sqrt(C), (F, C))); put(f + 'value.weight', rng.normal(0, 1 / np.sqrt(F), (C, F)))
put('ln_out.weight', 1 + 0.1 * rng.normal(size=C)); put('ln_out.bias', 0.1 * rng.normal(size=C)); put('head.weight', rng.normal(0, 1 / np.sqrt(C), (V, C)))

def layer_norm(x, w, b, eps=1e-5):
    m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / np.sqrt(v + eps) * w + b
def group_norm(x, groups, w, b, eps):
    x = x.reshape(groups, -1); m = x.mean(1, keepdims=True); v = ((x - m) ** 2).mean(1, keepdims=True)
    return ((x - m) / np.sqrt(v + eps)).reshape(-1) * w + b
sig = lambda z: 1 / (1 + np.exp(-z))
def normalize(x, eps=1e-12):  # torch.nn.functional.normalize(p=2, dim=-1)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)

def TMix_one(layer_id, H, N, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g
    r = xr @ R_
    w = np.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = sig(a0 + (xa @ a1) @ a2)
    g = sig(xg @ g1) @ g2
    kk = normalize((k * k_k).reshape(H, N)).reshape(H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * sig(v0 + (xv @ v1) @ v2)
    w = np.exp(-0.606531 * sig(w0 + w))
    vk = v.reshape(H, N, 1) @ k.reshape(H, 1, N)
    ab = (-kk).reshape(H, N, 1) @ (kk * a).reshape(H, 1, N)
    state = state * w.reshape(H, 1, N) + state @ ab + vk
    xx = (state @ r.reshape(H, N, 1)).reshape(H * N)
    xx = group_norm(xx, H, ln_w, ln_b, 64e-5)
    xx = xx + ((r * k * r_k).reshape(H, N).sum(-1, keepdims=True) * v.reshape(H, N)).reshape(H * N)
    return (xx * g) @ O_, x, state, v_first

def CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = np.maximum(k @ K_, 0) ** 2
    return k @ V_, x

# load exactly like the official loader: transpose the big matrices, squeeze, flatten r_k, fold ln0 into emb
z = {}
for k, t in P.items():
    if any(s in k for s in ('key.weight', 'value.weight', 'receptance.weight', 'output.weight', 'head.weight')): t = t.T
    t = np.squeeze(t)
    if k.endswith('att.r_k'): t = t.reshape(-1)
    z[k] = t.astype(np.float64)
z['emb.weight'] = np.stack([layer_norm(e, z['blocks.0.ln0.weight'], z['blocks.0.ln0.bias']) for e in z['emb.weight']])

def forward_one(idx, state):
    x = z['emb.weight'][idx]
    v_first = np.zeros(C)
    for i in range(L):
        bbb, att, ffn = f'blocks.{i}.', f'blocks.{i}.att.', f'blocks.{i}.ffn.'
        xx = layer_norm(x, z[bbb+'ln1.weight'], z[bbb+'ln1.bias'])
        g = lambda n: z.get(att + n)
        xx, state[i*3+0], state[i*3+1], v_first = TMix_one(i, H, N, xx, state[i*3+0], v_first, state[i*3+1],
            g('x_r'), g('x_w'), g('x_k'), g('x_v'), g('x_a'), g('x_g'), g('w0'), g('w1'), g('w2'), g('a0'), g('a1'), g('a2'), g('v0'), g('v1'), g('v2'),
            g('g1'), g('g2'), g('k_k'), g('k_a'), g('r_k'), g('receptance.weight'), g('key.weight'), g('value.weight'), g('output.weight'),
            g('ln_x.weight'), g('ln_x.bias'))
        x = x + xx
        xx = layer_norm(x, z[bbb+'ln2.weight'], z[bbb+'ln2.bias'])
        xx, state[i*3+2] = CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
        x = x + xx
    x = layer_norm(x, z['ln_out.weight'], z['ln_out.bias'])
    return x @ z['head.weight'], state

state = []
for i in range(L): state += [np.zeros(C), np.zeros((H, N, N)), np.zeros(C)]
tokens = [int(t) for t in rng.integers(0, V, 12)]
logits = []
for t in tokens:
    out, state = forward_one(t, state); logits.append(out.tolist())

out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
hdr, off, blobs = {}, 0, []
for k, t in P.items():
    b = t.astype('<f4').tobytes(); hdr[k] = {'dtype': 'F32', 'shape': list(t.shape), 'data_offsets': [off, off + len(b)]}; off += len(b); blobs.append(b)
hs = json.dumps(hdr).encode(); hs += b' ' * ((8 - (8 + len(hs)) % 8) % 8)
with open(os.path.join(out_dir, 'mini.safetensors'), 'wb') as fh: fh.write(struct.pack('<Q', len(hs))); fh.write(hs); [fh.write(b) for b in blobs]
json.dump({'tokens': tokens, 'logits': logits, 'config': dict(C=C, H=H, L=L, V=V)}, open(os.path.join(out_dir, 'expected.json'), 'w'))
print(f'wrote reference: {len(P)} tensors, {len(tokens)} tokens, logits[0][:3] = {np.round(logits[0][:3], 4).tolist()}')
