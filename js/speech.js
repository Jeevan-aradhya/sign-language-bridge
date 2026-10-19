// Thin wrapper around the browser's built-in Web Speech API (text-to-speech).
// Runs fully on-device in most browsers; no audio is sent anywhere by this app.

export class Speaker {
  constructor() {
    this.synth = typeof window !== 'undefined' && 'speechSynthesis' in window ? window.speechSynthesis : null;
    this.voiceURI = null;
    this.rate = 1;
    this.onVoices = () => {};
    if (this.synth) {
      this.synth.addEventListener?.('voiceschanged', () => this.onVoices(this.voices()));
    }
  }

  get supported() { return !!this.synth; }

  voices() {
    if (!this.synth) return [];
    const all = this.synth.getVoices();
    const en = all.filter((v) => /^en/i.test(v.lang));
    return en.length ? en : all;
  }

  speak(text) {
    const t = (text || '').trim();
    if (!this.synth || !t) return false;
    this.synth.cancel();
    const u = new SpeechSynthesisUtterance(t);
    u.rate = this.rate;
    u.lang = 'en-US';
    const v = this.synth.getVoices().find((x) => x.voiceURI === this.voiceURI);
    if (v) { u.voice = v; u.lang = v.lang; }
    this.synth.speak(u);
    return true;
  }

  stop() { this.synth?.cancel(); }
}
