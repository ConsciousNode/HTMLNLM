// Run from the repo root: node tests/dialogue-tools.js   (no dependencies)
// Checks filter-1 on made-up lines (every flag type fires where it should) and on the two real
// seed-420 twins, and diversity-1 on the twins (6 replayed opening lines, one shared phrase after them).
const { filterDialogue } = require('../tools/dialogue/filter.js');
const { diversityReport } = require('../tools/dialogue/diversity.js');
const fs = require('fs'); let bad = 0;
const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) bad++; };
const L = (speaker, text, gen) => ({ speaker, role: 'pip', text, gen });
const d = [L('Wren', 'Hi Moss! User: what is up'), L('Moss', '"Hello there," says Wren with a grin.'),
  L('Wren', 'Check @FaithLove and https://x.com now'), L('Moss', 'ok ok ok 짱 shots ;_; |ON TOTAL'),
  L('Wren', 'Bye! See you later!'), L('Moss', 'Goodbye friend, take care'), L('Wren', 'Bye bye!'),
  L('Moss', 'I like the pond and the frogs and the rain', { tokens: 120, endedBy: 'cap' }),
  L('Wren', 'I like the pond and the frogs and the rain'), L('Moss', 'the frogs sing the frogs sing the frogs sing'),
  L('Wren', 'Moss (dreaming): I saw a heron'), L('Moss', "I'm not planning on taking a stab at any of it")].map((x, i) => ({ ...x, turn: i + 1 }));
const r = filterDialogue(d, ['Wren', 'Moss']); const has = (t, f) => r.lines[t - 1].flags.some(x => x.type === f);
ok(has(1, 'speaker-slip'), 'speaker slip "User:" mid-line'); ok(has(2, 'dialogue-tag') && has(2, 'quote-opener'), 'novel-style dialogue tag');
ok(has(3, 'handle') && has(3, 'url-or-markup'), 'invented @handle and URL'); ok(has(4, 'script-switch'), 'script switch');
ok([5, 6, 7].every(t => has(t, 'goodbye-spiral')), 'goodbye spiral marked on every line of it'); ok(has(8, 'hit-cap'), 'hit the token cap');
ok(has(9, 'repeat'), 'exact repeat'); ok(has(10, 'stutter'), 'stutter inside a line'); ok(has(11, 'speaker-slip'), 'loop label "Moss (dreaming):"');
ok(has(12, 'heavy-theme'), 'idiom "taking a stab" is flagged (that is why it flags and never deletes)');
ok(r.lines.length === d.length && r.lines.every((l, i) => l.text === d[i].text), 'nothing deleted or changed');
const A = JSON.parse(fs.readFileSync('tests/fixtures/seed420-stateful.json', 'utf8'));
const B = JSON.parse(fs.readFileSync('tests/fixtures/seed420-stateless.json', 'utf8'));
const fa = filterDialogue(A.dialogue, A.pips.map(p => p.name)).filter, fb = filterDialogue(B.dialogue, B.pips.map(p => p.name)).filter;
ok(fa.counts['fragment-run'] === 6, `stateful twin: one six-line fragment chain (turns 16–21, Ed's catch), got ${fa.counts['fragment-run']}`);
ok(fb.counts['heavy-theme'] === 4, `stateless twin: 4 heavy-theme lines (8, 11, 16, 33), got ${fb.counts['heavy-theme']}`);
const rep = diversityReport([{ label: 'stateful', data: A }, { label: 'stateless', data: B }]);
ok(rep.sharedOpenings[0] && rep.sharedOpenings[0].identicalLines === 6 && /window 5 \+ 1/.test(rep.sharedOpenings[0].note), 'twins replay 6 opening lines (= window 5 + 1), then split');
ok(rep.sharedPhrases.length === 1 && rep.sharedPhrases[0].phrase === 'i don’t think so', 'after the split they share one phrase');
ok(rep.recurringNames.length === 0, 'no invented name recurs across the twins');
console.log(bad ? `\n${bad} FAILED` : '\nALL OK'); process.exit(bad ? 1 : 0);
