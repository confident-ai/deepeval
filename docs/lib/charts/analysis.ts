import type { Fn } from "./expression";

export const safe = (f: Fn, x: number): number => {
  try {
    const v = f(x);
    return typeof v === "number" ? v : NaN;
  } catch {
    return NaN;
  }
};

const sample = (f: Fn, a: number, b: number, n: number) => {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    xs.push(x);
    ys.push(safe(f, x));
  }
  return { xs, ys };
};

const dedupe = (values: number[], tolerance: number) => {
  const out: number[] = [];
  for (const v of values.filter(Number.isFinite).sort((p, q) => p - q)) {
    if (!out.length || v - out[out.length - 1] > tolerance) out.push(v);
  }
  return out;
};

const golden = (g: Fn, a: number, b: number) => {
  const r = (Math.sqrt(5) - 1) / 2;
  let c = b - r * (b - a);
  let d = a + r * (b - a);
  let fc = g(c);
  let fd = g(d);
  for (let i = 0; i < 90; i++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - r * (b - a);
      fc = g(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + r * (b - a);
      fd = g(d);
    }
  }
  return (a + b) / 2;
};

const extrema = (f: Fn, sign: 1 | -1, a: number, b: number, span: number) => {
  const n = 600;
  const { xs, ys } = sample(f, a, b, n);
  const out: number[] = [];
  for (let i = 1; i < n; i++) {
    const [y0, y1, y2] = [ys[i - 1] * sign, ys[i] * sign, ys[i + 1] * sign];
    if (![y0, y1, y2].every(Number.isFinite)) continue;
    if (y1 < y0 && y1 <= y2) {
      const x = golden(t => {
        const v = safe(f, t) * sign;
        return Number.isFinite(v) ? v : Infinity;
      }, xs[i - 1], xs[i + 1]);
      const v = safe(f, x);
      if (Number.isFinite(v) && Math.abs(v) < 50 * span) out.push(x);
    }
  }
  return dedupe(out, (b - a) * 1e-4);
};

export const minima = (f: Fn, a: number, b: number, span: number) => extrema(f, 1, a, b, span);
export const maxima = (f: Fn, a: number, b: number, span: number) => extrema(f, -1, a, b, span);

export const roots = (f: Fn, a: number, b: number, span: number) => {
  const n = 1000;
  const { xs, ys } = sample(f, a, b, n);
  const out: number[] = [];
  for (let i = 0; i <= n; i++) if (ys[i] === 0) out.push(xs[i]);
  for (let i = 0; i < n; i++) {
    const y0 = ys[i];
    const y1 = ys[i + 1];
    if (!Number.isFinite(y0) || !Number.isFinite(y1) || y0 === 0 || y1 === 0) continue;
    if (Math.sign(y0) === Math.sign(y1) || Math.abs(y1 - y0) > 2 * span) continue;
    let lo = xs[i];
    let hi = xs[i + 1];
    let flo = y0;
    for (let k = 0; k < 64; k++) {
      const m = (lo + hi) / 2;
      const fm = safe(f, m);
      if (!Number.isFinite(fm)) break;
      if (Math.sign(fm) === Math.sign(flo)) {
        lo = m;
        flo = fm;
      } else hi = m;
    }
    out.push((lo + hi) / 2);
  }
  return dedupe(out, (b - a) * 1e-4);
};

export const crossings = (f: Fn, g: Fn, a: number, b: number, span: number) =>
  roots(x => f(x) - g(x), a, b, span);

export const asymptotes = (f: Fn, a: number, b: number, span: number) => {
  const big = 20 * span;
  const n = 2000;
  const { xs, ys } = sample(f, a, b, n);
  const out: number[] = [];
  for (let i = 0; i <= n; i++) if (ys[i] === Infinity || ys[i] === -Infinity) out.push(xs[i]);
  for (let i = 0; i < n; i++) {
    const y0 = ys[i];
    const y1 = ys[i + 1];
    if (!Number.isFinite(y0) || !Number.isFinite(y1)) continue;
    if (Math.sign(y0) === Math.sign(y1) || Math.abs(y1 - y0) <= 2 * span) continue;
    let lo = xs[i];
    let hi = xs[i + 1];
    let flo = y0;
    for (let k = 0; k < 80; k++) {
      const m = (lo + hi) / 2;
      const fm = safe(f, m);
      if (!Number.isFinite(fm)) {
        lo = hi = m;
        break;
      }
      if (Math.sign(fm) === Math.sign(flo)) {
        lo = m;
        flo = fm;
      } else hi = m;
    }
    const x = (lo + hi) / 2;
    const eps = 1e-9 * (1 + Math.abs(x));
    if (Math.abs(safe(f, x - eps)) > big || !Number.isFinite(safe(f, x))) out.push(x);
  }
  for (let i = 1; i < n; i++) {
    const [y0, y1, y2] = [Math.abs(ys[i - 1]), Math.abs(ys[i]), Math.abs(ys[i + 1])];
    if (![y0, y1, y2].every(Number.isFinite)) continue;
    if (y1 > big && y1 >= y0 && y1 >= y2) {
      const x = golden(t => {
        const v = Math.abs(safe(f, t));
        return Number.isFinite(v) ? -v : -Infinity;
      }, xs[i - 1], xs[i + 1]);
      const v = safe(f, x);
      if (!Number.isFinite(v) || Math.abs(v) > big * 10) out.push(x);
    }
  }
  return dedupe(out, (b - a) * 1e-3);
};
