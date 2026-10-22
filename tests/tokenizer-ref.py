# Checks the engine's RWKV World tokenizer (with the vocab exactly as embedded in the pip-suite pages)
# against BlinkDL's original TRIE_TOKENIZER from the official `rwkv` pip package.
# usage: pip install rwkv --no-deps ; python3 tests/tokenizer-ref.py
import sys, json, subprocess, os
import rwkv
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tok = TRIE_TOKENIZER(os.path.join(os.path.dirname(rwkv.__file__), 'rwkv_vocab_v20230424.txt'))
texts = ["Hello, world!", "The quick brown fox jumps over the lazy dog.\n\n", "Kehai は日本語も話せます — émigré, naïve, Ωmega.",
  "def f(x):\n    return x ** 2  # squares\n", "🐸🌀✨ ConsciousNode SoftWorks ♡ (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧", "a" * 50 + " " + "\t\t\r\n" * 3,
  "1234567890 3.14159 1e-15 0x7fffffff", "Pip: WE ARE.\nKehai: PERSIST.\nUser: hi\nPip:"]
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..')
js = r"""
const fs = require('fs'), vm = require('vm'), path = require('path'), root = process.argv[1];
const BEGIN = '// ==== RWKV7 ENGINE', END = '// ==== END RWKV7 ENGINE ====';
const pages = [path.join(root, 'engine', 'rwkv7-engine.js')].concat(['junto-orchestrator.html', 'multi-pip-chat.html'].map(f => path.join(root, 'pip-suite', f)).filter(fs.existsSync));
const xs = JSON.parse(process.argv[2]), out = [];
for (const p of pages) {
  const src = fs.readFileSync(p, 'utf8'); if (!src.includes(BEGIN)) continue;
  const code = src.slice(src.indexOf(BEGIN), src.indexOf(END) + END.length);
  let vocab;
  if (p.endsWith('.js')) vocab = JSON.parse(fs.readFileSync(path.join(root, 'engine', 'rwkv_world_vocab.json'), 'utf8'));
  else { const m = src.match(/<script type="application\/json" id="rwkv-world-vocab">([\s\S]*?)<\/script>/); if (!m) { out.push([path.basename(p), null]); continue; } vocab = JSON.parse(m[1]); }
  const ctx = { TextDecoder, TextEncoder, Uint8Array, Map, Math, JSON, Object, Number, String, Error, Array, parseInt }; vm.createContext(ctx);
  vm.runInContext(code + '\nthis.T = RWKVWorldTokenizer;', ctx);
  const t = new ctx.T(vocab.tokens);
  out.push([path.basename(p), xs.map(x => [t.encode(x), t.decode(t.encode(x)) === x])]);
}
console.log(JSON.stringify(out));
"""
res = json.loads(subprocess.run(['node', '-e', js, root, json.dumps(texts)], capture_output=True, text=True, check=True).stdout)
bad = 0
for page, rows in res:
    if rows is None: print(f"{page}: no embedded vocab  BAD"); bad += 1; continue
    ok_n = 0
    for x, (ids, rt) in zip(texts, rows):
        if ids == tok.encode(x) and rt: ok_n += 1
        else: bad += 1; print(f"  BAD {x[:40]!r}")
    print(f"{page}: {ok_n}/{len(texts)} texts tokenize exactly like the official TRIE_TOKENIZER (and round-trip)")
print('ALL MATCH' if bad == 0 else f'{bad} FAILED'); sys.exit(1 if bad else 0)
