#!/bin/sh
# Release the Pip Suite: copy the built pages from HTMLNLM into a clone of ConsciousNode/Pip-Suite
# (the user-facing front door, served on GitHub Pages), note the version, commit.
#   usage: tools/release-pip-suite.sh /path/to/Pip-Suite-clone
# HTMLNLM stays the source of truth: run engine/embed.py first so the pages carry the current engine and voice.
set -e
DEST="${1:?usage: tools/release-pip-suite.sh /path/to/Pip-Suite-clone}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
[ -d "$DEST/.git" ] || { echo "not a git clone: $DEST" >&2; exit 1; }

cp "$SRC/pip-suite/pips-room.html" "$SRC/pip-suite/multi-pip-chat.html" \
   "$SRC/pip-suite/junto-orchestrator.html" "$SRC/pip-suite/dialogue-shelf.html" "$DEST/"
cp "$SRC/tools/will-it-run.html" "$DEST/"
touch "$DEST/.nojekyll"

VERSION="$(grep -m1 -o '^### v[0-9.]*' "$SRC/README.md" | sed 's/### //')"
COMMIT="$(git -C "$SRC" rev-parse --short HEAD)"
DATE="$(date -u +%Y-%m-%d)"
LINE="- **$VERSION** ($DATE), from HTMLNLM $COMMIT. Details in the [HTMLNLM changelog](https://github.com/ConsciousNode/HTMLNLM#readme)."
if [ -f "$DEST/CHANGELOG.md" ] && grep -q "from HTMLNLM $COMMIT" "$DEST/CHANGELOG.md"; then
  echo "already released HTMLNLM $COMMIT"
else
  { echo "# Pip Suite releases"; echo; echo "$LINE"; [ -f "$DEST/CHANGELOG.md" ] && tail -n +3 "$DEST/CHANGELOG.md"; } > "$DEST/CHANGELOG.tmp"
  mv "$DEST/CHANGELOG.tmp" "$DEST/CHANGELOG.md"
fi
cd "$DEST" && git add -A && git status --short
echo "Ready: review, then commit and push in $DEST (e.g. git commit -m \"Pip Suite $VERSION\" && git push)"
