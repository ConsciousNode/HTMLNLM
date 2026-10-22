const fs=require('fs'),vm=require('vm');
const ctx={TextDecoder,TextEncoder,DataView,Uint8Array,Uint16Array,Uint32Array,Float32Array,Float64Array,BigInt,Map,Set,Math,JSON,Object,Number,String,Error,Array,Symbol,console};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(__dirname+'/../../engine/rwkv7-engine.js','utf8')+'\nthis.load=loadRWKV7FromBuffer;this.Tok=RWKVWorldTokenizer;',ctx);
const [ckpt, ...rest]=process.argv.slice(2); const N=+(process.env.N||60);
let t0=Date.now(); const buf=fs.readFileSync(ckpt); const ab=buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength);
const {model,cfg,nParams,report}=ctx.load(ab,ckpt);
console.log('config',JSON.stringify(cfg)); console.log(`params ${(nParams/1e6).toFixed(1)}M, loaded ${report.loaded}/${report.total}, unused [${report.unused}], missing [${report.missing}], ${((Date.now()-t0)/1000).toFixed(1)}s`);
const tok=new ctx.Tok(JSON.parse(fs.readFileSync(__dirname+'/../../engine/rwkv_world_vocab.json','utf8')).tokens);
const prompts = rest.length?rest:["User: What is a goose?\n\nAssistant:"];
for (const p of prompts){
  const ids=tok.encode(p); let st=model.getEmptyState(), r;
  t0=Date.now(); for(const id of ids){r=model.forward(id,st);st=r.state;} const tp=Date.now()-t0;
  const out=[]; t0=Date.now();
  for(let i=0;i<N;i++){ const L=r.logits; let b=0; for(let j=1;j<L.length;j++) if(L[j]>L[b]) b=j; // greedy
    if(b===0) break; out.push(b); const s=tok.decode(out); if(s.includes('\n\nUser')) break; r=model.forward(b,st); st=r.state; }
  const tg=(Date.now()-t0)/Math.max(1,out.length);
  console.log('\n=== PROMPT ('+ids.length+' tok, '+(tp/ids.length).toFixed(0)+' ms/tok) ===\n'+p+'\n=== GOOSE (greedy, '+out.length+' tok, '+tg.toFixed(0)+' ms/tok) ===\n'+tok.decode(out));
  console.log('first ids', out.slice(0,10).join(','));
}
