const fs=require('fs'),vm=require('vm');const ctx={TextDecoder,TextEncoder,DataView,Uint8Array,Uint16Array,Uint32Array,Float32Array,Float64Array,BigInt,Map,Set,Math,JSON,Object,Number,String,Error,Array,Symbol,console};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(__dirname+'/../../engine/rwkv7-engine.js','utf8')+'\nthis.load=loadRWKV7FromBuffer;this.Tok=RWKVWorldTokenizer;',ctx);
const [ck,mode,tf]=process.argv.slice(2);
const tok=new ctx.Tok(JSON.parse(fs.readFileSync(__dirname+'/../../engine/rwkv_world_vocab.json','utf8')).tokens);
if(mode==='tok'){fs.writeFileSync(tf,JSON.stringify(tok.encode('User: What is a goose?\n\nAssistant: A goose is')));process.exit();}
const b=fs.readFileSync(ck);const {model}=ctx.load(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),ck);
const ids=JSON.parse(fs.readFileSync(tf)),ref=JSON.parse(fs.readFileSync(tf+'.ref.json'));let st=model.getEmptyState(),worst=0,agree=0;
ids.forEach((id,i)=>{const r=model.forward(id,st);st=r.state;let m=0,am=0,ar=0;for(let j=0;j<r.logits.length;j++){m=Math.max(m,Math.abs(r.logits[j]-ref[i][j]));if(r.logits[j]>r.logits[am])am=j;if(ref[i][j]>ref[i][ar])ar=j;}worst=Math.max(worst,m);agree+=am===ar;});
console.log(`engine vs official-math numpy on real weights: ${ids.length} tokens, max |dlogit| = ${worst.toExponential(2)}, argmax agree ${agree}/${ids.length}`);
