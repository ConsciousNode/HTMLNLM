const { filterDialogue } = require('./filter.js'); const fs = require('fs');
for (const f of process.argv.slice(2)) {
  const d = JSON.parse(fs.readFileSync(f, 'utf8')); const r = filterDialogue(d.dialogue, d.pips.map(p => p.name));
  d.dialogue = r.lines; d.filter = { ...r.filter, appliedAfterExport: true };
  fs.writeFileSync(f.replace(/\.json$/, '.filtered.json'), JSON.stringify(d, null, 2));
}
