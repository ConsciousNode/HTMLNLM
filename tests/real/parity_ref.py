import sys, json, struct, numpy as np
st, tokfile = sys.argv[1], sys.argv[2]
raw = open(st,'rb').read(); n = struct.unpack('<Q', raw[:8])[0]; hdr = json.loads(raw[8:8+n]); base = 8+n
def get(k):
    h = hdr[k]; a,b = h['data_offsets']; buf = raw[base+a:base+b]
    if h['dtype']=='BF16': arr = (np.frombuffer(buf,'<u2').astype(np.uint32)<<16).view(np.float32)
    elif h['dtype']=='F16': arr = np.frombuffer(buf,'<f2').astype(np.float32)
    else: arr = np.frombuffer(buf,'<f4')
    return arr.reshape(h['shape'])
P = {k: get(k) for k in hdr if k != '__metadata__'}
C = P['emb.weight'].shape[1]; V = P['emb.weight'].shape[0]; H = P['blocks.0.att.r_k'].shape[0]; N = C//H
L = 1+max(int(k.split('.')[1]) for k in P if k.startswith('blocks.'))
src = open(__import__('os').path.join(__import__('os').path.dirname(__file__),'..','ref','rwkv7_ref.py')).read()
fns = src[src.index('def layer_norm'):src.index('state = []')]
exec(fns)
tokens = json.load(open(tokfile)); state = []
for i in range(L): state += [np.zeros(C), np.zeros((H, N, N)), np.zeros(C)]
outs = []
for t in tokens: o, state = forward_one(t, state); outs.append(o.tolist())
json.dump(outs, open(tokfile + '.ref.json','w')); print('ref done', len(tokens), 'tokens; argmax of last', int(np.argmax(outs[-1])))
