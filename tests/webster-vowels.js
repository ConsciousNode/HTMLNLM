// Webster v4: render steady vowels with the Klatt core and measure their formants (LPC, numpy) against the
// Peterson & Barney targets the engine aims at. `node tests/webster-vowels.js` (needs python3 + numpy)
const fs = require('fs'), os = require('os'), path = require('path'), { execFileSync } = require('child_process');
const html = fs.readFileSync(path.join(__dirname, '..', 'webster', 'webster.html'), 'utf8');
const core = html.slice(html.indexOf('// ==== WEBSTER KLATT CORE (v4) ===='), html.indexOf('// ==== END WEBSTER KLATT CORE ===='));
const K = new Function(core + '; return { kRender, KV };')();
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'webster-')), T = { i: 0, 'æ': 0, a: 0, u: 0, 'ɛ': 0, 'ɔ': 0 };
for (const v of Object.keys(T)) { const r = K.kRender(['_', v, v, v, '.'], { pitch: 110, rate: 0.6 });
  fs.writeFileSync(path.join(dir, v + '.f32'), Buffer.from(r.samples.buffer)); }
const py = `
import sys, numpy as np, json
sr=22050; out={}
for v in json.loads(sys.argv[2]):
    x=np.fromfile(sys.argv[1]+'/'+v+'.f32',np.float32).astype(float); mid=len(x)//2; s=x[mid-1500:mid+1500]*np.hamming(3000)
    s=np.append(s[0],s[1:]-0.9*s[:-1]); o=18; r=np.array([np.dot(s[:len(s)-k],s[k:]) for k in range(o+1)])
    a=np.zeros(o+1);a[0]=1;e=r[0]
    for i in range(1,o+1):
        k=-(r[i]+np.dot(a[1:i],r[i-1:0:-1]))/e; a[1:i+1]=a[1:i+1]+k*a[i-1::-1][:i]; e*=1-k*k
    fb=sorted((np.angle(z)*sr/(2*np.pi),-np.log(abs(z))*sr/np.pi) for z in np.roots(a) if z.imag>0)
    out[v]=[round(f) for f,b in fb if 150<f<4000 and b<800][:2]
print(json.dumps(out))`;
const got = JSON.parse(execFileSync('python3', ['-c', py, dir, JSON.stringify(Object.keys(T))]).toString());
let bad = 0;
for (const v of Object.keys(T)) { const [t1, t2] = K.KV[v], [m1, m2] = got[v] || [];
  const ok = m1 && m2 && Math.abs(m1 - t1) / t1 < 0.2 && Math.abs(m2 - t2) / t2 < 0.2; if (!ok) bad++;
  console.log(`/${v}/  target F1 ${t1} F2 ${t2}   measured F1 ${m1} F2 ${m2}   ${ok ? 'ok' : 'FAIL'}`); }
fs.rmSync(dir, { recursive: true, force: true });
console.log(bad ? `FAIL (${bad})` : 'PASS'); process.exit(bad ? 1 : 0);
