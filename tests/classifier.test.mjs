import test from 'node:test';
import assert from 'node:assert/strict';
import { extractFeatures, ruleClassify, classify, featureVector, KNN } from '../js/classifier.js';
import { makeHand, THUMBS } from './synth.mjs';

const label = (lm) => ruleClassify(extractFeatures(lm))?.label ?? null;
const touch = (lm, a, b, off = 0.004) => { lm[a] = { x: lm[b].x + off, y: lm[b].y, z: lm[b].z + off }; return lm; };

// Idealised poses. Fist-family letters (S, E, T, N, M) are deliberately not covered here:
// they differ by a few millimetres of thumb position and need real calibration data.
test('B: flat hand, fingers together', () => {
  assert.equal(label(makeHand({ fingers: [0, 0, 0, 0], thumb: 'tuck' })), 'B');
});
test('space: open hand, fingers spread', () => {
  assert.equal(label(makeHand({ fingers: [0, 0, 0, 0], thumb: 'out', spread: [-22, -8, 8, 22] })), 'SPACE');
});
test('U vs V by finger gap', () => {
  assert.equal(label(makeHand({ fingers: [0, 0, 1, 1] })), 'U');
  assert.equal(label(makeHand({ fingers: [0, 0, 1, 1], spread: [-12, 12, 0, 0] })), 'V');
});
test('H: U shape pointing sideways', () => {
  assert.equal(label(makeHand({ fingers: [0, 0, 1, 1], rotate: 90 })), 'H');
});
test('W: three fingers up', () => {
  assert.equal(label(makeHand({ fingers: [0, 0, 0, 1], spread: [-12, 0, 12, 0] })), 'W');
});
test('I and Y: little finger up, thumb tucked vs out', () => {
  assert.equal(label(makeHand({ fingers: [1, 1, 1, 0], thumb: 'tuck' })), 'I');
  assert.equal(label(makeHand({ fingers: [1, 1, 1, 0], thumb: 'out' })), 'Y');
});
test('L: index up, thumb out', () => {
  assert.equal(label(makeHand({ fingers: [0, 1, 1, 1], thumb: 'out' })), 'L');
});
test('D: index up, thumb touching curled fingers', () => {
  const lm = makeHand({ fingers: [0, 1, 1, 1], thumb: 'tuck' });
  assert.equal(label(touch(lm, 4, 12)), 'D');
});
test('G and Q: index and thumb parallel, sideways vs downward', () => {
  assert.equal(label(makeHand({ fingers: [0, 1, 1, 1], thumb: 'par', rotate: 90 })), 'G');
  assert.equal(label(makeHand({ fingers: [0, 1, 1, 1], thumb: 'par', rotate: 180 })), 'Q');
});
test('F: index and thumb touch, other three up', () => {
  const lm = makeHand({ fingers: [0.8, 0, 0, 0], thumb: 'tuck' });
  assert.equal(label(touch(lm, 4, 8)), 'F');
});
test('O: curved fingers meeting the thumb', () => {
  const lm = makeHand({ fingers: [0.6, 0.6, 0.6, 0.6], thumb: 'tuck' });
  lm[4] = { x: (lm[8].x + lm[12].x) / 2, y: (lm[8].y + lm[12].y) / 2, z: (lm[8].z + lm[12].z) / 2 };
  assert.equal(label(lm), 'O');
});
test('C: curved fingers, thumb open', () => {
  assert.equal(label(makeHand({ fingers: [0.45, 0.45, 0.45, 0.45], thumb: 'out' })), 'C');
});
test('X: hooked index, other fingers closed', () => {
  assert.equal(label(makeHand({
    fingers: [0, 1, 1, 1], bends: [[20, 85, 25], null, null, null], thumb: 'tuck',
  })), 'X');
});
test('A: fist with thumb beside it', () => {
  const lm = makeHand({ fingers: [1, 1, 1, 1], thumbPts: [[-0.020, -0.020, 0.005], [-0.040, -0.036, 0.010], [-0.052, -0.060, 0.012], [-0.056, -0.082, 0.014]] });
  assert.equal(label(lm), 'A');
});
test('classification does not depend on hand rotation about the camera axis for finger-count letters', () => {
  for (const r of [-25, -10, 10, 25]) {
    assert.equal(label(makeHand({ fingers: [0, 0, 1, 1], spread: [-12, 12, 0, 0], rotate: r })), 'V', `rotate ${r}`);
    assert.equal(label(makeHand({ fingers: [1, 1, 1, 0], thumb: 'tuck', rotate: r })), 'I', `rotate ${r}`);
  }
});
test('R: crossed index and middle', () => {
  const lm = makeHand({ fingers: [0, 0, 1, 1] });
  // swap fingertip x so the index tip lies on the pinky side of the middle tip
  const t8 = lm[8].x; lm[8] = { ...lm[8], x: lm[12].x + 0.004 }; lm[12] = { ...lm[12], x: t8 - 0.004 };
  assert.equal(label(lm), 'R');
});

test('noise: jitter of 2 mm does not flip clear letters', () => {
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.004;
  for (const [args, want] of [
    [{ fingers: [0, 0, 0, 0], thumb: 'tuck' }, 'B'],
    [{ fingers: [0, 0, 1, 1], spread: [-12, 12, 0, 0] }, 'V'],
    [{ fingers: [0, 1, 1, 1], thumb: 'out' }, 'L'],
    [{ fingers: [1, 1, 1, 0], thumb: 'out' }, 'Y'],
  ]) {
    let ok = 0;
    for (let i = 0; i < 50; i++) {
      const lm = makeHand(args).map((p) => ({ x: p.x + rnd(), y: p.y + rnd(), z: p.z + rnd() }));
      if (label(lm) === want) ok++;
    }
    assert.ok(ok >= 45, `${want}: ${ok}/50 under jitter`);
  }
});

test('unknown shapes return null instead of a wrong guess', () => {
  assert.equal(label(makeHand({ fingers: [1, 0, 1, 0] })), null);
});

test('k-NN: trained samples override rules for the same hand shape', () => {
  const knn = new KNN();
  const items = [];
  // teach it that the "B" shape is the letter K (silly, but proves the override path)
  for (let i = 0; i < 20; i++) {
    const lm = makeHand({ fingers: [0, 0, 0, 0], thumb: 'tuck' }).map((p) => ({ x: p.x + i * 1e-4, y: p.y, z: p.z }));
    items.push({ label: 'K', vec: featureVector(lm, 'Right') });
  }
  knn.setSamples(items);
  const lm = makeHand({ fingers: [0, 0, 0, 0], thumb: 'tuck' });
  const { result } = classify(lm, 'Right', knn);
  assert.equal(result.label, 'K');
  assert.equal(result.source, 'trained');
});

test('k-NN: far-away shapes fall back to rules', () => {
  const knn = new KNN();
  const items = [];
  for (let i = 0; i < 20; i++) {
    const lm = makeHand({ fingers: [0, 0, 0, 0], thumb: 'tuck' }).map((p) => ({ x: p.x + i * 1e-4, y: p.y, z: p.z }));
    items.push({ label: 'B', vec: featureVector(lm, 'Right') });
  }
  knn.setSamples(items);
  const { result } = classify(makeHand({ fingers: [0, 1, 1, 1], thumb: 'out' }), 'Right', knn);
  assert.equal(result.label, 'L');
  assert.equal(result.source, 'rules');
});

test('left hands are mirrored into the same feature space', () => {
  const right = makeHand({ fingers: [0, 0, 1, 1], spread: [-12, 12, 0, 0] });
  const left = right.map((p) => ({ ...p, x: -p.x }));
  const a = featureVector(right, 'Right'), b = featureVector(left, 'Left');
  const maxDiff = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  assert.ok(maxDiff < 1e-9);
});
