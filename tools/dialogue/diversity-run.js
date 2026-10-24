const { diversityReport } = require('./diversity.js'); const fs = require('fs');
const runs = process.argv.slice(2).map(f => ({ label: f.split('/').pop().replace(/^dialogue_|\.json$/g, ''), data: JSON.parse(fs.readFileSync(f, 'utf8')) }));
console.log(JSON.stringify(diversityReport(runs), null, 1));
