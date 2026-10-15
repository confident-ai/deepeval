export type Fn = (x: number) => number;

const FUNCTIONS: Record<string, { arity: number; fn: (...args: number[]) => number }> = {
  sin: { arity: 1, fn: Math.sin },
  cos: { arity: 1, fn: Math.cos },
  tan: { arity: 1, fn: Math.tan },
  asin: { arity: 1, fn: Math.asin },
  acos: { arity: 1, fn: Math.acos },
  atan: { arity: 1, fn: Math.atan },
  exp: { arity: 1, fn: Math.exp },
  ln: { arity: 1, fn: Math.log },
  log: { arity: 1, fn: Math.log10 },
  sqrt: { arity: 1, fn: Math.sqrt },
  abs: { arity: 1, fn: Math.abs },
  floor: { arity: 1, fn: Math.floor },
  ceil: { arity: 1, fn: Math.ceil },
  sign: { arity: 1, fn: Math.sign },
  min: { arity: 2, fn: Math.min },
  max: { arity: 2, fn: Math.max },
  pow: { arity: 2, fn: Math.pow },
};

const CONSTANTS: Record<string, number> = { pi: Math.PI, e: Math.E };

type Token = { kind: "num"; value: number } | { kind: "name"; value: string } | { kind: "op"; value: string };

const tokenize = (src: string): Token[] => {
  const tokens: Token[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (/\s/.test(ch)) {
      i++;
    } else if (/[0-9.]/.test(ch)) {
      const m = /^(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/i.exec(src.slice(i));
      if (!m) throw new Error(`unexpected "${ch}" at position ${i + 1}`);
      tokens.push({ kind: "num", value: Number(m[0]) });
      i += m[0].length;
    } else if (/[a-z_]/i.test(ch)) {
      const m = /^[a-z_][a-z0-9_]*/i.exec(src.slice(i))!;
      tokens.push({ kind: "name", value: m[0].toLowerCase() });
      i += m[0].length;
    } else if ((ch === "<" || ch === ">") && src[i + 1] === "=") {
      tokens.push({ kind: "op", value: `${ch}=` });
      i += 2;
    } else if ("+-*/^(),<>?:".includes(ch)) {
      tokens.push({ kind: "op", value: ch });
      i++;
    } else {
      throw new Error(`unexpected "${ch}" at position ${i + 1}`);
    }
  }
  return tokens;
};

export const compileExpression = (src: string): Fn => {
  const tokens = tokenize(src);
  let pos = 0;
  const peek = () => tokens[pos];
  const isOp = (value: string) => peek()?.kind === "op" && peek()!.value === value;
  const expect = (value: string) => {
    if (!isOp(value)) throw new Error(`expected "${value}"${peek() ? ` before "${peek()!.value}"` : " at the end"}`);
    pos++;
  };

  const conditional = (): Fn => {
    const test = comparison();
    if (!isOp("?")) return test;
    pos++;
    const yes = conditional();
    expect(":");
    const no = conditional();
    return x => (test(x) ? yes(x) : no(x));
  };

  const comparison = (): Fn => {
    const left = sum();
    const next = peek();
    const op = next?.kind === "op" ? next.value : "";
    if (!["<", ">", "<=", ">="].includes(op)) return left;
    pos++;
    const right = sum();
    if (op === "<") return x => (left(x) < right(x) ? 1 : 0);
    if (op === ">") return x => (left(x) > right(x) ? 1 : 0);
    if (op === "<=") return x => (left(x) <= right(x) ? 1 : 0);
    return x => (left(x) >= right(x) ? 1 : 0);
  };

  const sum = (): Fn => {
    let left = product();
    while (isOp("+") || isOp("-")) {
      const op = tokens[pos++].value;
      const a = left;
      const b = product();
      left = op === "+" ? x => a(x) + b(x) : x => a(x) - b(x);
    }
    return left;
  };

  const product = (): Fn => {
    let left = unary();
    while (isOp("*") || isOp("/")) {
      const op = tokens[pos++].value;
      const a = left;
      const b = unary();
      left = op === "*" ? x => a(x) * b(x) : x => a(x) / b(x);
    }
    return left;
  };

  const unary = (): Fn => {
    if (isOp("-")) {
      pos++;
      const a = unary();
      return x => -a(x);
    }
    if (isOp("+")) {
      pos++;
      return unary();
    }
    return power();
  };

  const power = (): Fn => {
    const base = atom();
    if (!isOp("^")) return base;
    pos++;
    const exponent = unary();
    return x => base(x) ** exponent(x);
  };

  const atom = (): Fn => {
    const token = tokens[pos++];
    if (!token) throw new Error("ends too early");
    if (token.kind === "num") {
      const v = token.value;
      if (peek()?.kind === "name" || isOp("(")) throw new Error(`write multiplication out: "${v} * …"`);
      return () => v;
    }
    if (token.kind === "op") {
      if (token.value !== "(") throw new Error(`unexpected "${token.value}"`);
      const inner = conditional();
      expect(")");
      return inner;
    }
    const name = token.value;
    if (name === "x") return x => x;
    if (Object.hasOwn(CONSTANTS, name)) {
      const v = CONSTANTS[name];
      return () => v;
    }
    const f = Object.hasOwn(FUNCTIONS, name) ? FUNCTIONS[name] : undefined;
    if (!f) throw new Error(`unknown name "${name}" (use x, pi, e, or ${Object.keys(FUNCTIONS).join(", ")})`);
    expect("(");
    const args: Fn[] = [conditional()];
    while (isOp(",")) {
      pos++;
      args.push(conditional());
    }
    expect(")");
    if (args.length !== f.arity) throw new Error(`${name}() takes ${f.arity} argument${f.arity > 1 ? "s" : ""}`);
    return args.length === 1
      ? x => f.fn(args[0](x))
      : x => f.fn(args[0](x), args[1](x));
  };

  const fn = conditional();
  if (pos < tokens.length) throw new Error(`unexpected "${tokens[pos].value}"`);
  return fn;
};

export const expressionError = (src: string): string | null => {
  try {
    compileExpression(src);
    return null;
  } catch (err) {
    return (err as Error).message;
  }
};
