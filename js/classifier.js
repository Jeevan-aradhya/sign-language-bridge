// Hand-shape classifier for ASL fingerspelling.
//
// Input  : 21 MediaPipe *world* landmarks (metres, hand-centred, rotation preserved).
// Stage 1: geometric features (finger curl, thumb position, spread, orientation).
// Stage 2: rule-based letter decision (works out of the box, no training data).
// Stage 3: optional k-NN over samples recorded in the Calibrate tab. When a trained
//          sample is close enough it overrides the rules, so accuracy improves with data.
//
// J and Z need motion; they are handled in motion.js.

const FINGERS = [
  [5, 6, 7, 8],     // index
  [9, 10, 11, 12],  // middle
  [13, 14, 15, 16], // ring
  [17, 18, 19, 20], // pinky
];

// Total curl (sum of three joint bends, degrees) → finger state.
export const EXT_MAX = 75;   // below: straight  (E)
export const FIST_MIN = 180; // above: fully curled (F); in between: hooked/curved (H)

const sub = (a, b) => ({ x: b.x - a.x, y: b.y - a.y, z: b.z - a.z });
const len = (a) => Math.hypot(a.x, a.y, a.z);
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });
const dist = (a, b) => len(sub(a, b));
const angle = (a, b) => {
  const l = len(a) * len(b);
  if (l < 1e-9) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, dot(a, b) / l))) * 180) / Math.PI;
};

export function extractFeatures(lm) {
  const s = dist(lm[0], lm[9]) || 1e-6; // palm length = scale unit
  const d = (i, j) => dist(lm[i], lm[j]) / s;
  const dm = (i, a, b) => dist(lm[i], mid(lm[a], lm[b])) / s;
  const palm = sub(lm[0], lm[9]);

  const fingers = FINGERS.map(([m, p, q, t]) => {
    const bendM = angle(sub(lm[0], lm[m]), sub(lm[m], lm[p]));
    const bendP = angle(sub(lm[m], lm[p]), sub(lm[p], lm[q]));
    const bendD = angle(sub(lm[p], lm[q]), sub(lm[q], lm[t]));
    const curl = bendM + bendP + bendD;
    return { curl, state: curl < EXT_MAX ? 'E' : curl >= FIST_MIN ? 'F' : 'H' };
  });

  const thumbStraight = angle(sub(lm[2], lm[3]), sub(lm[3], lm[4])) < 40;
  const thumbExt = thumbStraight && d(4, 9) > 0.85;

  const up = -palm.y / (len(palm) || 1e-6);
  const orient = up > 0.55 ? 'up' : up < -0.55 ? 'down' : 'side';

  // lateral position of the thumb tip across the knuckle line (0 = index, 1 = pinky)
  const across = sub(lm[5], lm[17]);
  const u = dot(sub(lm[5], lm[4]), across) / (dot(across, across) || 1e-6);

  // > 0 when the index tip has crossed over the middle tip (letter R)
  const cross = dot(sub(lm[12], lm[8]), across) / ((len(across) || 1e-6) * s);

  return {
    d, dm, fingers, orient, u, cross,
    pattern: fingers.map((f) => (f.state === 'E' ? '1' : '0')).join(''),
    states: fingers.map((f) => f.state),
    avgCurl: fingers.reduce((a, f) => a + f.curl, 0) / 4,
    thumbExt, thumbStraight,
    angThumbIndex: angle(sub(lm[2], lm[4]), sub(lm[5], lm[8])),
    spread: d(8, 20),
    gaps: (d(8, 12) + d(12, 16) + d(16, 20)) / 3,
  };
}

// Returns { label, conf } or null. conf is a rough prior, not a probability.
export function ruleClassify(f) {
  const { d, dm, states, pattern, orient } = f;
  const [I, M, R, P] = states;

  switch (pattern) {
    case '1111':
      if (f.spread > 1.0 || f.gaps > 0.55) return { label: 'SPACE', conf: 0.85 };
      return { label: 'B', conf: 0.85 };

    case '0111': // F: index+thumb ring, other three up
      if (d(4, 8) < 0.6) return { label: 'F', conf: 0.85 };
      return null;

    case '1110':
      return { label: 'W', conf: 0.85 };

    case '1100': {
      const gap = d(8, 12);
      if (f.cross > -0.04 && gap < 0.4) return { label: 'R', conf: 0.7 };
      if (d(4, 10) < 0.55 && gap > 0.3) {
        return { label: orient === 'down' ? 'P' : 'K', conf: 0.7 };
      }
      if (gap > 0.42) return { label: 'V', conf: 0.85 };
      if (orient === 'side') return { label: 'H', conf: 0.8 };
      return { label: 'U', conf: 0.85 };
    }

    case '1000':
      if (f.thumbExt) {
        if (f.angThumbIndex >= 50) return { label: 'L', conf: 0.9 };
        if (orient === 'down') return { label: 'Q', conf: 0.7 };
        return { label: 'G', conf: 0.75 };
      }
      if (d(4, 12) < 0.6) return { label: 'D', conf: 0.8 }; // thumb touches the curled fingers
      if (f.thumbStraight && f.angThumbIndex < 35 && orient !== 'up') {
        return { label: orient === 'down' ? 'Q' : 'G', conf: 0.65 }; // thumb parallel to index
      }
      return { label: 'D', conf: 0.65 };

    case '0001':
      return f.thumbExt ? { label: 'Y', conf: 0.9 } : { label: 'I', conf: 0.85 };

    case '0000': {
      if (I === 'H' && M === 'F' && R === 'F' && P === 'F') return { label: 'X', conf: 0.7 };

      const touch = d(4, 8);
      if (touch < 0.45 && d(4, 12) < 0.65 && f.avgCurl < FIST_MIN + 10) {
        return { label: 'O', conf: 0.75 };
      }
      const hooked = states.filter((s) => s === 'H').length;
      if (hooked >= 3 && touch >= 0.45) return { label: 'C', conf: 0.75 };

      // Fist family: A, S, E, T, N, M. These differ only in where the thumb sits, so the
      // rules are the least reliable here. Calibrate these letters for best results.
      if (f.u < 0.12) return { label: 'A', conf: 0.6 }; // thumb beside the index knuckle
      const tT = dm(4, 6, 10), tN = dm(4, 10, 14), tM = dm(4, 14, 18);
      if (tT < 0.38 && f.u < 0.4) return { label: 'T', conf: 0.5 };
      if (tM < 0.38 && f.u > 0.55) return { label: 'M', conf: 0.5 };
      if (tN < 0.38 && f.u >= 0.35 && f.u <= 0.65) return { label: 'N', conf: 0.5 };
      if (d(4, 12) < 0.5 || d(4, 16) < 0.5) return { label: 'E', conf: 0.5 };
      if (f.u < 0.8 && d(4, 10) < 0.7) return { label: 'S', conf: 0.55 };
      return { label: 'A', conf: 0.5 };
    }
    default:
      return null;
  }
}

// ---- trainable layer -------------------------------------------------------

// 60-number vector: landmarks relative to the wrist, scaled by palm length.
// Left hands are mirrored so one set of samples serves both hands.
export function featureVector(lm, handedness) {
  const s = dist(lm[0], lm[9]) || 1e-6;
  const flip = handedness === 'Left' ? -1 : 1;
  const out = new Array(60);
  for (let i = 1; i < 21; i++) {
    out[(i - 1) * 3] = (flip * (lm[i].x - lm[0].x)) / s;
    out[(i - 1) * 3 + 1] = (lm[i].y - lm[0].y) / s;
    out[(i - 1) * 3 + 2] = (lm[i].z - lm[0].z) / s;
  }
  return out;
}

const vdist = (a, b) => {
  let t = 0;
  for (let i = 0; i < a.length; i++) { const q = a[i] - b[i]; t += q * q; }
  return Math.sqrt(t);
};

export class KNN {
  constructor() { this.items = []; this.threshold = 1; }

  setSamples(items) { this.items = items.slice(); this._calibrate(); }
  get size() { return this.items.length; }
  get labels() { return [...new Set(this.items.map((i) => i.label))]; }

  // Accept radius = 1.6 × the 90th-percentile same-class nearest-neighbour distance.
  _calibrate() {
    const byLabel = new Map();
    for (const it of this.items) {
      if (!byLabel.has(it.label)) byLabel.set(it.label, []);
      byLabel.get(it.label).push(it.vec);
    }
    const nn = [];
    for (const vecs of byLabel.values()) {
      if (vecs.length < 2) continue;
      for (let i = 0; i < vecs.length; i++) {
        let best = Infinity;
        for (let j = 0; j < vecs.length; j++) if (i !== j) best = Math.min(best, vdist(vecs[i], vecs[j]));
        nn.push(best);
      }
    }
    if (!nn.length) { this.threshold = 1; return; }
    nn.sort((a, b) => a - b);
    const p90 = nn[Math.min(nn.length - 1, Math.floor(nn.length * 0.9))];
    this.threshold = Math.max(0.6, p90 * 1.6);
  }

  predict(vec, k = 5) {
    if (this.items.length < 2) return null;
    const scored = this.items.map((it) => ({ label: it.label, dist: vdist(vec, it.vec) }));
    scored.sort((a, b) => a.dist - b.dist);
    const top = scored.slice(0, Math.min(k, scored.length));
    const votes = new Map();
    for (const t of top) votes.set(t.label, (votes.get(t.label) || 0) + 1 / (t.dist + 0.05));
    let label = null, w = -1, total = 0;
    for (const [l, v] of votes) { total += v; if (v > w) { w = v; label = l; } }
    const nearest = top.find((t) => t.label === label).dist;
    return { label, dist: nearest, share: w / total, accepted: nearest <= this.threshold && w / total >= 0.55 };
  }
}

// ---- public entry ----------------------------------------------------------

export function classify(worldLm, handedness, knn) {
  const feats = extractFeatures(worldLm);
  const rule = ruleClassify(feats);
  let result = rule ? { ...rule, source: 'rules' } : null;

  if (knn && knn.size >= 2) {
    const p = knn.predict(featureVector(worldLm, handedness));
    if (p && p.accepted) {
      result = { label: p.label, conf: Math.min(0.99, 0.6 + 0.4 * p.share), source: 'trained' };
    }
  }
  return { feats, result };
}
