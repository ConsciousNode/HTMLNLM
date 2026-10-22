# Writes a PyTorch-format .pth (zip + pickle protocol 2, the layout torch.save uses) WITHOUT PyTorch,
# from a .safetensors file, so tests/pth-reader.js can check the browser .pth reader.
# Exercises: BF16 and F32 storages, several tensors sharing one storage (nonzero storage_offset),
# OrderedDict + _metadata (BUILD), _rebuild_tensor_v2 with backward-hooks OrderedDict.
# usage: python3 tests/ref/make_pth.py IN.safetensors OUT.pth OUT_expected.json
import sys, json, struct, types, pickle, zipfile, collections, io
import numpy as np

def read_st(path):
    b = open(path, 'rb').read(); n = struct.unpack('<Q', b[:8])[0]; h = json.loads(b[8:8 + n]); base = 8 + n; out = {}
    for k, m in h.items():
        if k == '__metadata__': continue
        s, e = m['data_offsets']; out[k] = np.frombuffer(b[base + s:base + e], dtype=np.float32).reshape(m['shape'])
    return out

torch = types.ModuleType('torch'); utils = types.ModuleType('torch._utils')
class BFloat16Storage: pass
class FloatStorage: pass
for c in (BFloat16Storage, FloatStorage): c.__module__ = 'torch'; setattr(torch, c.__name__, c)
def _rebuild_tensor_v2(*a): pass
_rebuild_tensor_v2.__module__ = 'torch._utils'; utils._rebuild_tensor_v2 = _rebuild_tensor_v2
sys.modules['torch'] = torch; sys.modules['torch._utils'] = utils; torch._utils = utils

class Storage:
    def __init__(self, cls, key, raw, numel): self.cls, self.key, self.raw, self.numel = cls, key, raw, numel
class Tensor:
    def __init__(self, st, off, shape): self.st, self.off, self.shape = st, off, tuple(shape)
    def __reduce__(self):
        stride = []; acc = 1
        for d in reversed(self.shape): stride.insert(0, acc); acc *= d
        return (_rebuild_tensor_v2, (self.st, self.off, self.shape, tuple(stride), False, collections.OrderedDict()))

def bf16_bits(a):
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)   # round to nearest even
    return r

t = read_st(sys.argv[1]); names = list(t)
sd = collections.OrderedDict(); expected = {}
# group: first 3 tensors share one F32 storage (offsets), the rest BF16, two per storage
groups, i = [], 0
groups.append(('F32', names[:3])); i = 3
while i < len(names): groups.append(('BF16', names[i:i + 2])); i += 2
storages = []
for key, (dt, grp) in enumerate(groups):
    parts, off, offs = [], 0, []
    for n in grp:
        a = t[n].ravel()
        if dt == 'F32': parts.append(a.astype('<f4').tobytes()); expected[n] = a.astype(np.float32)
        else:
            bits = bf16_bits(a); parts.append(bits.astype('<u2').tobytes()); expected[n] = (bits.astype(np.uint32) << 16).view(np.float32)
        offs.append(off); off += a.size
    st = Storage(FloatStorage if dt == 'F32' else BFloat16Storage, str(key), b''.join(parts), off); storages.append(st)
    for n, o in zip(grp, offs): sd[n] = Tensor(st, o, t[n].shape)
sd._metadata = collections.OrderedDict([('', {'version': 1})])

class P(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Storage): return ('storage', obj.cls, obj.key, 'cpu', obj.numel)
        return None
buf = io.BytesIO(); P(buf, protocol=2).dump(sd)
with zipfile.ZipFile(sys.argv[2], 'w', compression=zipfile.ZIP_STORED) as z:
    z.writestr('archive/data.pkl', buf.getvalue())
    for st in storages: z.writestr(f'archive/data/{st.key}', st.raw)
    z.writestr('archive/version', '3\n'); z.writestr('archive/byteorder', 'little')
json.dump({n: {'shape': list(t[n].shape), 'sum': float(expected[n].astype(np.float64).sum()), 'first': [float(x) for x in expected[n][:4]],
               'hash': int(np.frombuffer(expected[n].astype('<f4').tobytes(), dtype=np.uint32).astype(np.uint64).sum())} for n in names}, open(sys.argv[3], 'w'))
print(f'wrote {sys.argv[2]}: {len(names)} tensors in {len(storages)} storages')
