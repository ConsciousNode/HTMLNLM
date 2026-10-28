// ── HTMLNLM's handle on Webster: speak(text, voice) → Promise that resolves when the line has been said ──
// The AudioContext is made on first use (browsers allow sound only after a click or key press).
// Dictionary entries sometimes write diphthongs and affricates as one symbol ("aɪ", "tʃ"); the phoneme
// table has their parts, so they're split here instead of being played as silence.
const SPLIT = { 'aɪ': ['a', 'ɪ'], 'eɪ': ['e', 'ɪ'], 'aʊ': ['a', 'ʊ'], 'ɔɪ': ['ɔ', 'ɪ'], 'tʃ': ['t', 'ʃ'], 'dʒ': ['d', 'ʒ'], 'ɒ': ['ɔ'], 'ɯ': ['ʊ'] };
let synth = null, queue = Promise.resolve(), muted = false;
function detect(text) {
  if (/[぀-ヿ]/.test(text)) return 'ja'; if (/[가-힯]/.test(text)) return 'ko';
  if (/[Ѐ-ӿ]/.test(text)) return 'ru'; if (/[Ͱ-Ͽ]/.test(text)) return 'el';
  return 'en';
}
const G2PBY = { en: 'english', ja: 'japanese', ko: 'korean', ru: 'russian', el: 'greek', es: 'spanish', de: 'german', fr: 'french', it: 'italian' };
function phonemesFor(text, lang) {
  lang = lang || detect(text);
  let t = text.replace(/\*[^*]{0,80}\*/g, ' ');                   // *stage directions* aren't read aloud
  t = SymbolExpander.expand(t, lang, 'pause'); t = NumberExpander.expand(t, lang);
  const raw = G2P[G2PBY[lang] || 'english'](t), out = [];
  for (const p of raw) { if (PHONEMES[p]) out.push(p); else if (SPLIT[p]) out.push(...SPLIT[p]); }
  while (out.length && out[out.length - 1] === '_') out.pop();
  return out;
}
// A voice: { preset, pitch, rate, formantScale, brightness, roughness, natural }. Missing fields come from the preset.
function config(voice = {}, text = '') {
  const base = VOICE_PRESETS[voice.preset] || VOICE_PRESETS.narrator, t = text.trim();
  return { pitch: voice.pitch ?? base.pitch, rate: voice.rate ?? base.rate, formantScale: voice.formantScale ?? base.formantScale,
           brightness: voice.brightness ?? base.brightness, roughness: voice.roughness ?? base.roughness, natural: voice.natural ?? true,
           isQuestion: t.endsWith('?'), isExclamation: t.endsWith('!') };
}
function sayNow(text, voice) {
  return new Promise(resolve => {
    const ph = phonemesFor(text, voice && voice.lang);
    if (muted || !ph.length) return resolve();
    if (!synth) synth = new FormantSynth();
    if (synth.ctx.state === 'suspended') synth.ctx.resume();
    let done = false; const finish = () => { if (!done) { done = true; resolve(); } };
    const cfg = config(voice, text), secs = ph.reduce((a, p) => a + (PHONEMES[p] || PHONEMES['_']).dur / 1000 / cfg.rate, 0);
    synth.playSequence(ph, cfg, null, finish);
    setTimeout(finish, (secs + 1.5) * 1000);                      // never hang the chat on a lost callback
  });
}
return {
  /** Queue a line; resolves when it has been spoken (lines never overlap). */
  speak(text, voice) { const p = queue.then(() => sayNow(String(text || ''), voice)); queue = p.catch(() => {}); return p; },
  stop() { if (synth) synth.stop(); queue = Promise.resolve(); },
  setMuted(m) { muted = !!m; if (muted) this.stop(); },
  /** A distinct voice for speaker number i (deterministic, so a Pip keeps its voice). */
  voiceFor(i) {
    const V = [{ preset: 'narrator' }, { preset: 'female' }, { preset: 'elder' }, { preset: 'child' },
               { preset: 'narrator', pitch: 140, formantScale: 1.08 }, { preset: 'female', pitch: 210, formantScale: 1.12 }];
    return { ...V[i % V.length], ...(i >= V.length ? { pitch: 100 + (i * 37) % 120 } : {}) };
  },
  presets: () => Object.keys(VOICE_PRESETS), phonemesFor,
};
