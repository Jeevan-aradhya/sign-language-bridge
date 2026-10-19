import test from 'node:test';
import assert from 'node:assert/strict';
import { MotionTracker, detectZ, detectJ } from '../js/motion.js';
import { Stabilizer } from '../js/stabilizer.js';

let seed = 3;
const rnd = (a = 0.04) => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * a;

// polyline through key points, sampled n points per leg, in "hand sizes"
function path(keys, per = 10) {
  const out = [];
  for (let i = 1; i < keys.length; i++) {
    for (let k = 0; k < per; k++) {
      const t = k / per;
      out.push({ x: keys[i - 1][0] + (keys[i][0] - keys[i - 1][0]) * t + rnd(), y: keys[i - 1][1] + (keys[i][1] - keys[i - 1][1]) * t + rnd() });
    }
  }
  out.push({ x: keys.at(-1)[0], y: keys.at(-1)[1] });
  return out;
}

test('Z: across, diagonal down-back, across', () => {
  assert.ok(detectZ(path([[0, 0], [1.6, 0], [0, 1.3], [1.6, 1.3]])));
});
test('Z drawn right-to-left (mirrored) is also accepted', () => {
  assert.ok(detectZ(path([[1.6, 0], [0, 0], [1.6, 1.3], [0, 1.3]])));
});
test('J: down then hook', () => {
  assert.ok(detectJ(path([[0, 0], [0, 1.5], [-0.7, 1.3]])));
});
test('no false Z/J from a still hand, a straight swipe, or a circle', () => {
  const still = Array.from({ length: 40 }, () => ({ x: rnd(0.05), y: rnd(0.05) }));
  const swipe = path([[0, 0], [2, 0]]);
  const circle = Array.from({ length: 40 }, (_, i) => ({ x: Math.cos(i / 6), y: Math.sin(i / 6) }));
  for (const p of [still, swipe, circle]) { assert.ok(!detectZ(p)); assert.ok(!detectJ(p)); }
});
test('a plain downward swipe is not a J', () => {
  assert.ok(!detectJ(path([[0, 0], [0, 1.6]])));
});

test('tracker emits Z once, then cools down', () => {
  const tr = new MotionTracker();
  const pts = path([[0, 0], [1.6, 0], [0, 1.3], [1.6, 1.3]], 8);
  let hits = [];
  pts.forEach((p, i) => { const r = tr.update(i * 33, '1000', { x: p.x * 100, y: p.y * 100 }, 100); if (r) hits.push(r); });
  assert.deepEqual(hits, ['Z']);
});
test('tracker ignores other finger patterns', () => {
  const tr = new MotionTracker();
  const pts = path([[0, 0], [1.6, 0], [0, 1.3], [1.6, 1.3]], 8);
  const hits = pts.map((p, i) => tr.update(i * 33, '0000', { x: p.x * 100, y: p.y * 100 }, 100)).filter(Boolean);
  assert.equal(hits.length, 0);
});

test('stabilizer: commits after the hold time, once', () => {
  const s = new Stabilizer({ holdMs: 900 });
  const commits = [];
  for (let t = 0; t < 3000; t += 33) { const r = s.update('A', t, false); if (r.commit) commits.push([t, r.commit]); }
  assert.equal(commits.length, 1);
  assert.equal(commits[0][1], 'A');
  assert.ok(commits[0][0] >= 900 && commits[0][0] < 1100, `committed at ${commits[0][0]}`);
});
test('stabilizer: a flicker does not commit; moving hand pauses the timer', () => {
  const s = new Stabilizer({ holdMs: 900 });
  const got = [];
  for (let t = 0; t < 2000; t += 33) {
    const label = t % 200 < 100 ? 'B' : 'C';
    const r = s.update(label, t, false); if (r.commit) got.push(r.commit);
  }
  assert.equal(got.length, 0);
  const s2 = new Stabilizer({ holdMs: 900 });
  for (let t = 0; t < 3000; t += 33) { const r = s2.update('A', t, true); assert.equal(r.commit, null); }
});
test('stabilizer: same letter twice needs the hand to leave or change', () => {
  const s = new Stabilizer({ holdMs: 600 });
  const got = [];
  let t = 0;
  const run = (label, ms) => { for (const end = t + ms; t < end; t += 33) { const r = s.update(label, t, false); if (r.commit) got.push(r.commit); } };
  run('L', 1500); run(null, 600); run('L', 1500); // hand lowered between → second L
  assert.deepEqual(got, ['L', 'L']);
  run('L', 1500);                                  // still holding → no third L
  assert.deepEqual(got, ['L', 'L']);
});
test('stabilizer: after a motion letter the static pose does not type itself', () => {
  const s = new Stabilizer({ holdMs: 600 });
  let t = 0; const got = [];
  for (; t < 300; t += 33) s.update('D', t, false);
  s.suppress(t);
  for (const end = t + 2000; t < end; t += 33) { const r = s.update('D', t, false); if (r.commit) got.push(r.commit); }
  assert.deepEqual(got, []);
});
