// Turns noisy per-frame predictions into deliberate key-presses.
//  1. Majority vote over the last N frames removes flicker.
//  2. The stable sign must be held (and the hand must be still) for `holdMs`.
//  3. A committed sign is locked until the hand changes sign or leaves the frame,
//     so holding one sign doesn't type the same letter repeatedly.

export class Stabilizer {
  constructor({ holdMs = 900, windowSize = 9, quorum = 0.65, releaseMs = 350 } = {}) {
    this.holdMs = holdMs;
    this.n = windowSize;
    this.quorum = quorum;
    this.releaseMs = releaseMs;
    this.reset();
  }

  reset() {
    this.win = [];
    this.cur = null;     // current stable label
    this.since = 0;      // when `cur` became stable (or last moved)
    this.locked = null;  // last committed label
    this.noneSince = null;
  }

  // label: string|null (null = no hand / no confident sign). moving: hand is travelling.
  update(label, now, moving = false) {
    this.win.push(label);
    if (this.win.length > this.n) this.win.shift();

    const counts = new Map();
    for (const l of this.win) counts.set(l, (counts.get(l) || 0) + 1);
    let best = null, bestC = 0;
    for (const [l, c] of counts) if (l !== null && c > bestC) { best = l; bestC = c; }
    const enough = this.win.length >= Math.min(5, this.n);
    const stable = enough && best !== null && bestC / this.win.length >= this.quorum ? best : null;

    if (stable !== this.cur) { this.cur = stable; this.since = now; }
    if (moving) this.since = now;

    if (stable === null) {
      if (this.noneSince === null) this.noneSince = now;
      if (now - this.noneSince >= this.releaseMs) this.locked = null;
    } else {
      this.noneSince = null;
    }

    let progress = 0, commit = null;
    if (stable && stable !== this.locked) {
      progress = Math.min(1, (now - this.since) / this.holdMs);
      if (progress >= 1) { commit = stable; this.locked = stable; progress = 0; }
    }
    return { stable, progress, commit };
  }

  // Called after a motion letter (J/Z) is emitted so the static pose doesn't also fire.
  suppress(now) {
    const pose = this.cur ?? this.locked;
    this.win = []; this.cur = null; this.since = now; this.noneSince = null;
    this.locked = pose; // the I/D pose that was just used to draw J/Z must not also type itself
  }
}
