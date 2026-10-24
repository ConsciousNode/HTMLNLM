# Dialogue tools
These are the command-line twins of what runs inside the Pip Suite pages. Node, no dependencies.

- `filter.js` (**filter-1**): the same filter Multi-Pip Chat runs on Export. It **flags lines for a human to look at and never deletes or changes anything.** Flags: empty, hit-cap (maybe-cut on exports made before factory-3), speaker-slip, dialogue-tag, quote-opener, handle, url-or-markup, script-switch, garble, stutter, repeat, near-repeat, heavy-theme (violence, abuse, self-harm, crime, drugs, sexual, death), goodbye-spiral, fragment-run. It doesn't check tone drift; that needs a reader.
  - `node filter-run.js export.json` prints the flags
  - `node filter-apply.js export.json` writes `export.filtered.json`
- `diversity.js` (**diversity-1**): how alike a pile of exports are. It reports distinct-1/2/3, pairwise overlap, replayed openings, recurring four-word phrases (counted after any replayed opening) and recurring invented names. The phone version is `pip-suite/dialogue-shelf.html`.
  - `node diversity-run.js a.json b.json …`
