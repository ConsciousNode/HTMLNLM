# Random blocks of every quant type the engine reads, dequantized by the gguf Python package (llama.cpp's own
# reference), written as JSON for tests/gguf-dequant.js. Needs: pip install gguf numpy
# usage: python3 tests/ref/make_gguf_blocks.py out.json
import sys, json, numpy as np
from gguf import quants, GGMLQuantizationType as Q
rng = np.random.default_rng(7)
SIZES = {'Q4_0': (32, 18), 'Q4_1': (32, 20), 'Q5_0': (32, 22), 'Q5_1': (32, 24), 'Q8_0': (32, 34),
         'Q2_K': (256, 84), 'Q3_K': (256, 110), 'Q4_K': (256, 144), 'Q5_K': (256, 176), 'Q6_K': (256, 210)}
HALF = {'Q4_0': [0], 'Q4_1': [0, 2], 'Q5_0': [0], 'Q5_1': [0, 2], 'Q8_0': [0], 'Q2_K': [80, 82], 'Q3_K': [108],
        'Q4_K': [0, 2], 'Q5_K': [0, 2], 'Q6_K': [208]}            # fp16 fields: give them sane values, not NaN/inf
out = {}
for name, (bw, bb) in SIZES.items():
    nb, rows = 3, 4                                                # 4 rows of 3 blocks each
    raw = rng.integers(0, 256, size=(rows * nb, bb), dtype=np.uint8)
    for o in HALF[name]:
        raw[:, o:o + 2] = np.frombuffer(rng.uniform(-0.05, 0.05, rows * nb).astype(np.float16).tobytes(), np.uint8).reshape(-1, 2)
    deq = quants.dequantize(raw.reshape(-1), getattr(Q, name)).astype(np.float64).reshape(rows, nb * bw)
    x = rng.normal(size=nb * bw)
    out[name] = {'bytes': raw.reshape(-1).tolist(), 'rows': rows, 'cols': nb * bw, 'deq': deq.reshape(-1).tolist(),
                 'x': x.tolist(), 'dot': (deq @ x).tolist()}
json.dump(out, open(sys.argv[1], 'w'))
print('wrote', len(out), 'types:', ', '.join(out))
