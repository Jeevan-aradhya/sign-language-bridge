// J and Z are drawn in the air, so a single frame cannot tell them apart from I and D.
// We track the fingertip path, simplify it (Ramer–Douglas–Peucker) and look for the
// stroke sequence: Z = across, diagonal back, across again; J = down, then a hook.
// All distances are in "hand sizes" (palm length) so it works at any distance.

const WINDOW_MS = 1600;
const COOLDOWN_MS = 1200;

function perpDist(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const l = Math.hypot(dx, dy);
  if (l < 1e-9) return Math.hypot(p.x - a.x, p.y - a.y);
  return Math.abs(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) / l;
}

export function simplify(pts, eps) {
  if (pts.length < 3) return pts.slice();
  let maxD = 0, idx = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const dd = perpDist(pts[i], pts[0], pts[pts.length - 1]);
    if (dd > maxD) { maxD = dd; idx = i; }
  }
  if (maxD <= eps) return [pts[0], pts[pts.length - 1]];
  const left = simplify(pts.slice(0, idx + 1), eps);
  const right = simplify(pts.slice(idx), eps);
  return left.slice(0, -1).concat(right);
}

function segments(pts, minLen) {
  const segs = [];
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i].x - pts[i - 1].x, dy = pts[i].y - pts[i - 1].y;
    const l = Math.hypot(dx, dy);
    if (l >= minLen) segs.push({ dx, dy, l });
    else if (segs.length) { // merge tiny jitters into the previous stroke
      segs[segs.length - 1].dx += dx; segs[segs.length - 1].dy += dy;
      segs[segs.length - 1].l = Math.hypot(segs[segs.length - 1].dx, segs[segs.length - 1].dy);
    }
  }
  return segs;
}

export function detectZ(pts) {
  if (pts.length < 8) return false;
  const segs = segments(simplify(pts, 0.28), 0.35);
  for (let i = 0; i + 2 < segs.length; i++) {
    const [a, b, c] = [segs[i], segs[i + 1], segs[i + 2]];
    const flat = (s) => Math.abs(s.dx) > 0.6 && Math.abs(s.dy) < 0.6 * Math.abs(s.dx);
    if (!flat(a) || !flat(c)) continue;
    if (Math.sign(a.dx) !== Math.sign(c.dx)) continue;
    if (Math.sign(b.dx) !== -Math.sign(a.dx) || Math.abs(b.dx) < 0.5) continue;
    if (b.dy < 0.3) continue; // middle stroke must go downward
    return true;
  }
  return false;
}

export function detectJ(pts) {
  if (pts.length < 8) return false;
  const segs = segments(simplify(pts, 0.25), 0.3);
  for (let i = 0; i + 1 < segs.length; i++) {
    const [a, b] = [segs[i], segs[i + 1]];
    const down = a.dy > 0.8 && Math.abs(a.dx) < 0.6 * a.dy;
    const hook = b.l >= 0.3 && b.dy < 0.25 * b.l; // sideways or upward, not further down
    if (down && hook) return true;
  }
  return false;
}

export class MotionTracker {
  constructor() { this.reset(); }

  reset() { this.mode = null; this.pts = []; this.cool = 0; this.lastSeen = 0; }

  // pattern: finger pattern from features ("1000" → Z candidate, "0001" → J candidate)
  // tip: {x, y} in pixels of the relevant fingertip; scale: palm length in pixels
  update(now, pattern, tip, scale) {
    const mode = pattern === '1000' ? 'Z' : pattern === '0001' ? 'J' : null;
    if (mode) this.lastSeen = now;
    if (mode !== this.mode) {
      if (!mode && now - this.lastSeen < 300) return null; // brief flicker: keep the path
      this.mode = mode;
      this.pts = [];
    }
    if (!mode || !tip || !scale) return null;

    this.pts.push({ t: now, x: tip.x / scale, y: tip.y / scale });
    while (this.pts.length && now - this.pts[0].t > WINDOW_MS) this.pts.shift();
    if (now < this.cool) return null;

    const hit = mode === 'Z' ? detectZ(this.pts) : detectJ(this.pts);
    if (hit) { this.cool = now + COOLDOWN_MS; this.pts = []; return mode; }
    return null;
  }
}
