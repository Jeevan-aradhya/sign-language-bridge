// Synthetic MediaPipe-style world landmarks for idealised hand shapes.
// Right hand, palm toward the camera, fingers pointing up (-y). Units: metres.
// These are approximations used to test the decision logic, not real recordings.

const BASE_X = [-0.030, -0.010, 0.010, 0.028];
const BASE_Y = [-0.092, -0.095, -0.090, -0.080];
const SEG = [
  [0.040, 0.024, 0.020],
  [0.045, 0.028, 0.022],
  [0.042, 0.026, 0.021],
  [0.034, 0.020, 0.018],
];

const rad = (d) => (d * Math.PI) / 180;

// curl: 0 straight … 1 fist. `bends` overrides with explicit [mcp, pip, dip] degrees.
function finger(i, { curl = 0, bends = null, spread = 0 }) {
  const [m, p, d] = bends || [80 * curl, 100 * curl, 70 * curl];
  const sp = rad(spread);
  const d0 = [Math.sin(sp), -Math.cos(sp), 0];
  let pos = [BASE_X[i], BASE_Y[i], 0];
  const pts = [pos];
  let cum = 0;
  [m, p, d].forEach((b, k) => {
    cum += rad(b);
    const dir = [d0[0] * Math.cos(cum), d0[1] * Math.cos(cum), Math.sin(cum)];
    pos = [pos[0] + dir[0] * SEG[i][k], pos[1] + dir[1] * SEG[i][k], pos[2] + dir[2] * SEG[i][k]];
    pts.push(pos);
  });
  return pts; // mcp, pip, dip, tip
}

const THUMB = {
  // each: [cmc, mcp, ip, tip]
  out:   [[-0.020, -0.020, 0.005], [-0.038, -0.036, 0.010], [-0.068, -0.050, 0.010], [-0.094, -0.058, 0.010]],
  tuck:  [[-0.020, -0.020, 0.005], [-0.030, -0.040, 0.020], [-0.018, -0.055, 0.032], [-0.006, -0.066, 0.036]],
  // thumb tip touching the tip of a curled index (O / F / C-ish)
  pinch: [[-0.020, -0.020, 0.005], [-0.034, -0.040, 0.020], [-0.034, -0.065, 0.040], [-0.030, -0.088, 0.050]],
  // extended and parallel to the index finger (G / Q)
  par:   [[-0.020, -0.020, 0.005], [-0.036, -0.042, 0.010], [-0.047, -0.085, 0.010], [-0.052, -0.125, 0.010]],
  // alongside the fist, pointing up (A)
  side:  [[-0.020, -0.020, 0.005], [-0.036, -0.038, 0.012], [-0.040, -0.062, 0.016], [-0.040, -0.085, 0.018]],
};

export function makeHand({ fingers = [0, 0, 0, 0], thumb = 'tuck', spread = [0, 0, 0, 0], bends = [null, null, null, null], thumbPts = null, rotate = 0 }) {
  const lm = new Array(21);
  lm[0] = [0, 0, 0];
  const t = thumbPts || THUMB[thumb];
  lm[1] = t[0]; lm[2] = t[1]; lm[3] = t[2]; lm[4] = t[3];
  for (let i = 0; i < 4; i++) {
    const pts = finger(i, { curl: fingers[i], spread: spread[i], bends: bends[i] });
    for (let k = 0; k < 4; k++) lm[5 + i * 4 + k] = pts[k];
  }
  // rotate about z (in the image plane); positive = clockwise on screen
  const c = Math.cos(rad(rotate)), s = Math.sin(rad(rotate));
  return lm.map(([x, y, z]) => ({ x: x * c - y * s, y: x * s + y * c, z }));
}

export const THUMBS = THUMB;
