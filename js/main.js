import { classify, featureVector, KNN } from './classifier.js';
import { MotionTracker } from './motion.js';
import { Stabilizer } from './stabilizer.js';
import { suggest } from './words.js';
import { Speaker } from './speech.js';
import { GUIDE } from './guide.js';

const VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const TRAIN_LABELS = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('').concat('SPACE');
const MAX_PER_LABEL = 60;
const RING_LEN = 326.7;

const $ = (id) => document.getElementById(id);
const el = {
  video: $('video'), overlay: $('overlay'), trace: $('trace'), cta: $('cta'), startBtn: $('startBtn'),
  stopBtn: $('stopBtn'), status: $('status'), stageBar: $('stageBar'), srcChip: $('srcChip'), fps: $('fps'),
  ring: $('ring'), ringProg: $('ringProg'), letter: $('letter'), hint: $('hint'), sentence: $('sentence'),
  wordNow: $('wordNow'), suggest: $('suggest'), speakBtn: $('speakBtn'), backBtn: $('backBtn'), clearBtn: $('clearBtn'),
  holdMs: $('holdMs'), holdOut: $('holdOut'), autoWord: $('autoWord'), autoLetter: $('autoLetter'),
  rulesOnly: $('rulesOnly'), voice: $('voice'), rate: $('rate'), rateOut: $('rateOut'),
  chips: $('chips'), recBtn: $('recBtn'), delLetterBtn: $('delLetterBtn'), trainStatus: $('trainStatus'),
  exportBtn: $('exportBtn'), importFile: $('importFile'), wipeBtn: $('wipeBtn'), guideList: $('guideList'),
  repoLink: $('repoLink'),
};

/* ---------------- storage ---------------- */
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode / full */ } },
};

const settings = Object.assign(
  { holdMs: 900, autoWord: true, autoLetter: false, rulesOnly: false, voice: '', rate: 1 },
  store.get('slr.settings', {}),
);
const saveSettings = () => store.set('slr.settings', settings);

/* ---------------- state ---------------- */
let landmarker = null;
let stream = null;
let running = false;
let rafId = 0;
let lastVideoTime = -1;
let text = '';
let shipped = [];                 // samples from data/samples.json
let local = store.get('slr.samples.v1', []); // samples recorded in this browser
let selectedLabel = null;
let recording = null;             // { label, startAt, endAt, got, lastAt }
let lastFrame = null;             // { wl, hand } latest hand, for recording
let speed = 0, lastPalm = null, lastT = 0;
let frames = 0, fpsT = performance.now();
let committedFlash = 0;

const knn = new KNN();
const motion = new MotionTracker();
const stabilizer = new Stabilizer({ holdMs: settings.holdMs });
const speaker = new Speaker();

/* ---------------- sentence ---------------- */
function currentWord() { return text.slice(text.lastIndexOf(' ') + 1); }

function renderSentence() {
  el.sentence.textContent = '';
  if (!text) {
    const s = document.createElement('span'); s.className = 'empty'; s.textContent = 'Your sentence appears here';
    el.sentence.append(s);
  } else {
    const cut = text.lastIndexOf(' ') + 1;
    el.sentence.append(document.createTextNode(text.slice(0, cut)));
    if (cut < text.length) {
      const s = document.createElement('span'); s.className = 'cur'; s.textContent = text.slice(cut);
      el.sentence.append(s);
    }
  }
  const w = currentWord();
  el.wordNow.textContent = w ? `Current word: ${w}` : '';
  renderSuggestions();
}

function renderSuggestions() {
  el.suggest.textContent = '';
  suggest(currentWord(), 4).forEach((w, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'btn';
    const k = document.createElement('kbd'); k.textContent = String(i + 1);
    b.append(k, document.createTextNode(w.toUpperCase()));
    b.addEventListener('click', () => acceptSuggestion(w));
    el.suggest.append(b);
  });
}

function acceptSuggestion(word) {
  text = text.slice(0, text.lastIndexOf(' ') + 1) + word.toUpperCase() + ' ';
  renderSentence();
  if (settings.autoWord) speaker.speak(word);
}

function addLetter(ch) {
  text += ch;
  renderSentence();
  if (settings.autoLetter) speaker.speak(ch);
}

function addSpace() {
  if (!text || text.endsWith(' ')) return;
  const w = currentWord();
  text += ' ';
  renderSentence();
  if (settings.autoWord) speaker.speak(w.toLowerCase());
}

function commit(label) {
  if (label === 'SPACE') addSpace(); else addLetter(label);
  el.ring.classList.add('committed');
  committedFlash = performance.now() + 450;
  el.hint.textContent = label === 'SPACE'
    ? 'Space added. Spell the next word.'
    : `Added ${label}. Change the sign or lower your hand for the next letter.`;
}

/* ---------------- drawing ---------------- */
function drawOverlay(lm) {
  const c = el.overlay, ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!lm) return;
  const w = c.width, h = c.height;
  const css = getComputedStyle(document.documentElement);
  const signal = css.getPropertyValue('--signal').trim() || '#3ddc97';
  const lw = Math.max(2, w / 220);
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) { ctx.moveTo(lm[a].x * w, lm[a].y * h); ctx.lineTo(lm[b].x * w, lm[b].y * h); }
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)'; ctx.lineWidth = lw + 4; ctx.stroke(); // halo keeps it visible on any background
  ctx.strokeStyle = signal; ctx.lineWidth = lw; ctx.stroke();
  for (const p of lm) {
    ctx.beginPath(); ctx.arc(p.x * w, p.y * h, Math.max(3, w / 190), 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)'; ctx.stroke();
  }
}

function drawTrace(lm) {
  const c = el.trace, ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!lm) return;
  const vw = el.video.videoWidth || 4, vh = el.video.videoHeight || 3;
  const pts = lm.map((p) => ({ x: (1 - p.x) * vw, y: p.y * vh })); // mirrored like the camera view
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const size = Math.max(maxX - minX, maxY - minY) || 1;
  const pad = 26, k = (c.width - pad * 2) / size;
  const ox = (c.width - (maxX - minX) * k) / 2 - minX * k;
  const oy = (c.height - (maxY - minY) * k) / 2 - minY * k;
  const map = (p) => [p.x * k + ox, p.y * k + oy];
  ctx.strokeStyle = '#129a68'; ctx.lineWidth = 4; ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) { ctx.moveTo(...map(pts[a])); ctx.lineTo(...map(pts[b])); }
  ctx.stroke();
  ctx.fillStyle = '#0a6c85';
  for (const p of pts) { const [x, y] = map(p); ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); }
}

function setLetter(label, dim) {
  const t = label === 'SPACE' ? 'space' : label || '–';
  if (el.letter.textContent !== t) el.letter.textContent = t;
  el.letter.classList.toggle('word', label === 'SPACE');
  el.letter.classList.toggle('none', !label || dim);
}

function setProgress(p) {
  el.ringProg.style.strokeDashoffset = String(RING_LEN * (1 - p));
}

/* ---------------- per-frame pipeline ---------------- */
function onResult(res, now) {
  const hasHand = res && res.landmarks && res.landmarks.length > 0;
  if (!hasHand) {
    lastFrame = null; lastPalm = null; speed = 0;
    drawOverlay(null); drawTrace(null);
    stabilizer.update(null, now, false);
    motion.update(now, null, null, 0);
    setLetter(null); setProgress(0);
    if (now > committedFlash) el.ring.classList.remove('committed');
    el.srcChip.textContent = 'no hand';
    handleRecording(now, null, null);
    if (!recording) el.hint.textContent = 'No hand in view. Hold your hand up with the palm facing the camera.';
    return;
  }

  const lm = res.landmarks[0];
  const wl = (res.worldLandmarks && res.worldLandmarks[0]) || lm;
  const hand = res.handedness?.[0]?.[0]?.categoryName || 'Right';
  lastFrame = { wl, hand };

  const vw = el.video.videoWidth, vh = el.video.videoHeight;
  const palmPx = Math.hypot((lm[0].x - lm[9].x) * vw, (lm[0].y - lm[9].y) * vh) || 1;

  // hand travel speed in palm-lengths per second (smoothed)
  if (lastPalm) {
    const dt = Math.max(1, now - lastT) / 1000;
    const inst = Math.hypot((lm[9].x - lastPalm.x) * vw, (lm[9].y - lastPalm.y) * vh) / palmPx / dt;
    speed = speed * 0.7 + inst * 0.3;
  }
  lastPalm = { x: lm[9].x, y: lm[9].y }; lastT = now;
  const moving = speed > 1.3;

  const { feats, result } = classify(wl, hand, settings.rulesOnly ? null : knn);
  const label = result && result.conf >= 0.5 ? result.label : null;
  el.srcChip.textContent = result ? result.source : 'unsure';

  // J / Z: tracked from the fingertip path
  const tip = feats.pattern === '0001' ? lm[20] : lm[8];
  const motionLetter = motion.update(now, feats.pattern, { x: tip.x * vw, y: tip.y * vh }, palmPx);

  const st = stabilizer.update(label, now, moving);
  if (motionLetter) {
    stabilizer.suppress(now);
    commit(motionLetter);
    setLetter(motionLetter); setProgress(0);
  } else {
    if (st.commit) commit(st.commit);
    setLetter(st.stable || label, !st.stable);
    setProgress(st.progress);
    if (now > committedFlash) {
      el.ring.classList.remove('committed');
      if (!recording) {
        el.hint.textContent = !label ? 'Hand found, but the shape is not recognised yet. Try a clearer sign.'
          : moving ? 'Hold your hand still to confirm the letter.'
          : `Reading ${label === 'SPACE' ? 'space' : label}. Keep it steady to add it.`;
      }
    }
  }

  handleRecording(now, wl, hand);
  drawOverlay(lm); drawTrace(lm);
}

function loop() {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  const v = el.video;
  if (v.readyState < 2 || v.currentTime === lastVideoTime) return;
  lastVideoTime = v.currentTime;
  const now = performance.now();
  let res = null;
  try { res = landmarker.detectForVideo(v, now); } catch (e) { console.error(e); }
  onResult(res, now);

  frames++;
  if (now - fpsT >= 1000) { el.fps.textContent = `${frames} fps`; frames = 0; fpsT = now; }
}

/* ---------------- camera + model ---------------- */
function setStatus(msg) { el.status.textContent = msg; }

async function createLandmarker() {
  const { HandLandmarker, FilesetResolver } = await import(`${VISION}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${VISION}/wasm`);
  const make = (delegate) => HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: 'VIDEO', numHands: 1,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { return await make('GPU'); } catch { return await make('CPU'); }
}

async function startCamera() {
  if (running) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('This browser cannot open the camera here. Use https:// (or localhost) in a current browser.');
    return;
  }
  el.startBtn.disabled = true;
  setStatus('Asking for camera access…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (e) {
    el.startBtn.disabled = false;
    setStatus(e && e.name === 'NotAllowedError'
      ? 'Camera access was blocked. Allow the camera in the address bar, then press Start camera again.'
      : e && e.name === 'NotFoundError'
        ? 'No camera was found on this device.'
        : 'The camera could not be started. Close other apps that use it and try again.');
    return;
  }

  try {
    setStatus('Loading the hand-tracking model…');
    if (!landmarker) landmarker = await createLandmarker();
  } catch (e) {
    console.error(e);
    stream.getTracks().forEach((t) => t.stop()); stream = null;
    el.startBtn.disabled = false;
    setStatus('The hand-tracking model could not be loaded. Check your internet connection and reload the page.');
    return;
  }

  el.video.srcObject = stream;
  await el.video.play().catch(() => {});
  const resize = () => { el.overlay.width = el.video.videoWidth || 960; el.overlay.height = el.video.videoHeight || 720; };
  if (el.video.videoWidth) resize(); else el.video.addEventListener('loadedmetadata', resize, { once: true });

  running = true; lastVideoTime = -1; lastPalm = null; speed = 0;
  stabilizer.reset(); motion.reset();
  el.cta.hidden = true; el.stageBar.hidden = false; el.startBtn.disabled = false; setStatus('');
  el.hint.textContent = 'Camera on. Show a letter.';
  loop();
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((t) => t.stop()); stream = null;
  el.video.srcObject = null;
  drawOverlay(null); drawTrace(null);
  setLetter(null); setProgress(0);
  el.cta.hidden = false; el.stageBar.hidden = true;
  el.hint.textContent = 'Camera is off.';
  recording = null; updateRecUi();
}

/* ---------------- calibration ---------------- */
const allSamples = () => shipped.concat(local);
const countFor = (label) => allSamples().filter((s) => s.label === label).length;

function rebuildModel() { knn.setSamples(allSamples()); renderChips(); }

function renderChips() {
  el.chips.textContent = '';
  for (const label of TRAIN_LABELS) {
    const n = countFor(label);
    const b = document.createElement('button');
    b.type = 'button';
    b.className = (n ? 'has ' : '') + (label === 'SPACE' ? 'wide' : '');
    b.setAttribute('aria-pressed', String(label === selectedLabel));
    const big = document.createElement('b'); big.textContent = label === 'SPACE' ? 'Space' : label;
    const small = document.createElement('small'); small.textContent = n ? `${n} samples` : 'none';
    b.append(big, small);
    b.addEventListener('click', () => { selectedLabel = label; renderChips(); updateRecUi(); });
    el.chips.append(b);
  }
}

function updateRecUi() {
  const name = selectedLabel === 'SPACE' ? 'space' : selectedLabel;
  el.recBtn.disabled = !selectedLabel || !!recording;
  el.recBtn.textContent = recording ? 'Recording…' : selectedLabel ? `Record samples for ${name}` : 'Pick a letter first';
  el.delLetterBtn.disabled = !selectedLabel || !local.some((s) => s.label === selectedLabel);
}

function trainMsg(msg, warn = false) { el.trainStatus.textContent = msg; el.trainStatus.classList.toggle('warn', warn); }

function startRecording() {
  if (!selectedLabel || recording) return;
  if (!running) { trainMsg('Start the camera first (button above the video), then record.', true); return; }
  const now = performance.now();
  recording = { label: selectedLabel, startAt: now + 3000, endAt: now + 5500, got: [], lastAt: 0 };
  updateRecUi();
}

function handleRecording(now, wl, hand) {
  if (!recording) return;
  const r = recording;
  if (now < r.startAt) {
    const s = Math.ceil((r.startAt - now) / 1000);
    trainMsg(`Get ready to show ${r.label === 'SPACE' ? 'space' : r.label}… ${s}`);
    el.hint.textContent = `Recording ${r.label === 'SPACE' ? 'space' : r.label} starts in ${s}…`;
    return;
  }
  if (now < r.endAt) {
    trainMsg(`Recording ${r.label === 'SPACE' ? 'space' : r.label}. Hold the sign and turn your hand slightly. ${r.got.length} samples`);
    if (now - r.lastAt >= 60 && wl) {
      r.lastAt = now;
      r.got.push({ label: r.label, vec: featureVector(wl, hand).map((v) => Math.round(v * 1000) / 1000) });
    }
    return;
  }
  finishRecording();
}

function finishRecording() {
  const r = recording; recording = null;
  if (!r) return;
  if (r.got.length < 10) {
    trainMsg('Not enough frames with a visible hand. Keep your hand in view and try again.', true);
  } else {
    // keep earlier samples of this letter (more variety), but cap the total per letter
    const prior = local.filter((s) => s.label === r.label);
    const others = local.filter((s) => s.label !== r.label);
    local = others.concat(prior.concat(r.got).slice(-MAX_PER_LABEL));
    store.set('slr.samples.v1', local);
    rebuildModel();
    trainMsg(`Saved ${r.got.length} samples for ${r.label === 'SPACE' ? 'space' : r.label}. Record again from a slightly different angle for better results.`);
  }
  updateRecUi();
}

function exportSamples() {
  const blob = new Blob([JSON.stringify({ version: 1, samples: local })], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'samples.json';
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  trainMsg(`Exported ${local.length} samples.`);
}

async function importSamples(file) {
  try {
    const data = JSON.parse(await file.text());
    const ok = (data.samples || []).filter((s) => TRAIN_LABELS.includes(s.label) && Array.isArray(s.vec) && s.vec.length === 60 && s.vec.every(Number.isFinite));
    if (!ok.length) throw new Error('empty');
    const byLabel = new Map();
    for (const s of local.concat(ok)) { if (!byLabel.has(s.label)) byLabel.set(s.label, []); byLabel.get(s.label).push(s); }
    local = [...byLabel.values()].flatMap((arr) => arr.slice(-MAX_PER_LABEL));
    store.set('slr.samples.v1', local);
    rebuildModel(); updateRecUi();
    trainMsg(`Imported ${ok.length} samples.`);
  } catch {
    trainMsg('That file is not a valid samples export.', true);
  }
}

async function loadShipped() {
  try {
    const r = await fetch('data/samples.json', { cache: 'no-cache' });
    if (!r.ok) return;
    const data = await r.json();
    shipped = (data.samples || []).filter((s) => TRAIN_LABELS.includes(s.label) && Array.isArray(s.vec) && s.vec.length === 60);
  } catch { /* none shipped */ }
}

/* ---------------- UI wiring ---------------- */
function setupTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const select = (tab) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1;
      $(t.getAttribute('aria-controls')).hidden = !on;
    });
  };
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(t));
    t.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      select(next); next.focus();
    });
  });
}

function setupSettings() {
  el.holdMs.value = settings.holdMs; el.holdOut.textContent = `${(settings.holdMs / 1000).toFixed(1)} s`;
  el.autoWord.checked = settings.autoWord; el.autoLetter.checked = settings.autoLetter;
  el.rulesOnly.checked = settings.rulesOnly;
  el.rate.value = settings.rate; el.rateOut.textContent = `${Number(settings.rate).toFixed(1)}×`;
  speaker.rate = Number(settings.rate);

  el.holdMs.addEventListener('input', () => {
    settings.holdMs = Number(el.holdMs.value); stabilizer.holdMs = settings.holdMs;
    el.holdOut.textContent = `${(settings.holdMs / 1000).toFixed(1)} s`; saveSettings();
  });
  el.autoWord.addEventListener('change', () => { settings.autoWord = el.autoWord.checked; saveSettings(); });
  el.autoLetter.addEventListener('change', () => { settings.autoLetter = el.autoLetter.checked; saveSettings(); });
  el.rulesOnly.addEventListener('change', () => { settings.rulesOnly = el.rulesOnly.checked; saveSettings(); });
  el.rate.addEventListener('input', () => {
    settings.rate = Number(el.rate.value); speaker.rate = settings.rate;
    el.rateOut.textContent = `${settings.rate.toFixed(1)}×`; saveSettings();
  });

  const fillVoices = () => {
    const vs = speaker.voices();
    el.voice.textContent = '';
    if (!speaker.supported || !vs.length) {
      const o = document.createElement('option');
      o.textContent = speaker.supported ? 'Default voice' : 'Speech is not supported in this browser';
      o.value = ''; el.voice.append(o); el.voice.disabled = !speaker.supported; return;
    }
    for (const v of vs) {
      const o = document.createElement('option'); o.value = v.voiceURI; o.textContent = `${v.name} (${v.lang})`;
      if (v.voiceURI === settings.voice) o.selected = true;
      el.voice.append(o);
    }
    speaker.voiceURI = el.voice.value;
  };
  speaker.onVoices = fillVoices; fillVoices();
  el.voice.addEventListener('change', () => { settings.voice = el.voice.value; speaker.voiceURI = settings.voice; saveSettings(); });
}

function setupGuide() {
  for (const [k, v] of GUIDE) {
    const row = document.createElement('div');
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    row.append(dt, dd); el.guideList.append(row);
  }
}

function setupRepoLink() {
  const h = location.hostname;
  if (!h.endsWith('.github.io')) return;
  const repo = location.pathname.split('/')[1];
  if (!repo) return;
  el.repoLink.href = `https://github.com/${h.split('.')[0]}/${repo}`;
  el.repoLink.hidden = false;
}

function init() {
  setupTabs(); setupSettings(); setupGuide(); setupRepoLink();
  renderSentence(); renderChips(); updateRecUi(); setLetter(null); setProgress(0);

  el.startBtn.addEventListener('click', startCamera);
  el.stopBtn.addEventListener('click', stopCamera);
  el.speakBtn.addEventListener('click', () => {
    if (!text.trim()) { el.hint.textContent = 'Nothing to speak yet. Spell a word first.'; return; }
    if (!speaker.speak(text.toLowerCase())) el.hint.textContent = 'Speech is not available in this browser.';
  });
  el.backBtn.addEventListener('click', () => { text = text.slice(0, -1); renderSentence(); });
  el.clearBtn.addEventListener('click', () => { text = ''; speaker.stop(); renderSentence(); });
  el.recBtn.addEventListener('click', startRecording);
  el.delLetterBtn.addEventListener('click', () => {
    local = local.filter((s) => s.label !== selectedLabel); store.set('slr.samples.v1', local);
    rebuildModel(); updateRecUi(); trainMsg('Samples deleted for that letter.');
  });
  el.exportBtn.addEventListener('click', exportSamples);
  el.importFile.addEventListener('change', () => { const f = el.importFile.files[0]; if (f) importSamples(f); el.importFile.value = ''; });
  el.wipeBtn.addEventListener('click', () => {
    if (!local.length) { trainMsg('There are no samples in this browser.'); return; }
    if (!confirm('Delete all samples recorded in this browser?')) return;
    local = []; store.set('slr.samples.v1', local); rebuildModel(); updateRecUi(); trainMsg('All local samples deleted.');
  });

  document.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (/^[1-4]$/.test(e.key)) {
      const w = suggest(currentWord(), 4)[Number(e.key) - 1];
      if (w) { e.preventDefault(); acceptSuggestion(w); }
    } else if (e.key === 'Backspace' && t === document.body) {
      e.preventDefault(); text = text.slice(0, -1); renderSentence();
    }
  });

  window.addEventListener('pagehide', () => stream?.getTracks().forEach((tr) => tr.stop()));
  loadShipped().then(rebuildModel);
  rebuildModel();
}

init();

// Small hook so automated tests can read the sentence without scraping the DOM.
window.__signbridge = { getText: () => text, getSampleCount: () => allSamples().length };
