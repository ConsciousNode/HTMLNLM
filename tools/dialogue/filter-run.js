const { filterDialogue } = require('./filter.js');
const fs = require('fs');
for (const f of process.argv.slice(2)) {
  const d = JSON.parse(fs.readFileSync(f, 'utf8'));
  const r = filterDialogue(d.dialogue, d.pips.map(p => p.name));
  console.log('\n==', f.split('/').pop(), JSON.stringify(r.filter.counts), r.filter.verdict, `${r.filter.flaggedLines}/${r.filter.totalLines}`);
  for (const l of r.lines) if (l.flags.length) console.log(`  t${l.turn} ${l.speaker}: ${l.flags.map(x => x.type + (x.detail ? `(${x.detail})` : '')).join(' ')}`);
}
