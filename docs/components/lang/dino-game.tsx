"use client";

import { useEffect, useRef } from "react";
import styles from "./dino-game.module.scss";

/** Sprites are row strings; `#` is a filled pixel. */
const DINO_BODY = [
  "............######..",
  "...........########.",
  "...........###.####.",
  "...........########.",
  "...........########.",
  "...........#####....",
  "...........######...",
  "...........#####....",
  ".#.........#####....",
  ".##.......#######...",
  ".###.....#########..",
  ".####...###########.",
  ".#####.############.",
  "..#################.",
  "...################.",
  "....##############..",
  ".....############...",
  ".....###...######...",
  ".....##.....#####...",
];

const DINO_LEGS = [
  [".....##......###....", ".....##......###....", ".....####....###...."],
  [".....##......###....", ".....###.....###....", ".....###....####...."],
];

const DINO_FRAMES = DINO_LEGS.map((legs) => [...DINO_BODY, ...legs]);

const CACTUS = [
  "...##...",
  "...##...",
  "...##...",
  "#..##...",
  "#..##..#",
  "#..##..#",
  "##.##..#",
  ".#.##.##",
  ".#####.#",
  "...##.##",
  "...#####",
  "...##...",
  "...##...",
  "...##...",
  "...##...",
  "...##...",
];

const CLOUD = [
  "....######....",
  "..##########..",
  ".############.",
  "##############",
  ".############.",
];

const HEIGHT = 190;
const GROUND_INSET = 34;
const PX = 3;
const DINO_X = 28;
const GRAVITY = 0.62;
const JUMP_VELOCITY = -11.2;
const BASE_SPEED = 6;
const MAX_SPEED = 14;
const HITBOX_INSET = 5;

type Obstacle = { x: number; width: number; height: number; scale: number };
type Cloud = { x: number; y: number };
type Status = "idle" | "running" | "over";

const spriteSize = (sprite: string[], scale: number) => ({
  width: sprite[0].length * scale,
  height: sprite.length * scale,
});

const drawSprite = (
  ctx: CanvasRenderingContext2D,
  sprite: string[],
  x: number,
  y: number,
  scale: number,
) => {
  for (let row = 0; row < sprite.length; row++) {
    const line = sprite[row];
    let col = 0;
    while (col < line.length) {
      if (line[col] !== "#") {
        col++;
        continue;
      }
      // Coalesce each run of filled pixels into one rect.
      let run = 1;
      while (line[col + run] === "#") run++;
      ctx.fillRect(x + col * scale, y + row * scale, run * scale, scale);
      col += run;
    }
  }
};

export const DinoGame = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = canvas.clientWidth || 600;
    let colour = "#535353";
    let raf = 0;
    let last = 0;

    const dino = spriteSize(DINO_FRAMES[0], PX);
    const groundY = HEIGHT - GROUND_INSET;

    let status: Status = "idle";
    let dinoY = 0;
    let velocity = 0;
    let speed = BASE_SPEED;
    let distance = 0;
    let frames = 0;
    let spawnIn = 60;
    let obstacles: Obstacle[] = [];
    let clouds: Cloud[] = [];
    let best = 0;
    try {
      best = Number(localStorage.getItem("deepeval:dino-best") ?? 0) || 0;
    } catch {
      best = 0;
    }

    const readColour = () => {
      colour = getComputedStyle(canvas).color || colour;
    };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth || 600;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(HEIGHT * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.imageSmoothingEnabled = false;
    };

    const reset = () => {
      status = "running";
      dinoY = 0;
      velocity = 0;
      speed = BASE_SPEED;
      distance = 0;
      frames = 0;
      spawnIn = 60;
      obstacles = [];
      clouds = [
        { x: width * 0.5, y: 34 },
        { x: width * 0.9, y: 62 },
      ];
    };

    const jump = () => {
      if (status === "running") {
        if (dinoY === 0) velocity = JUMP_VELOCITY;
        return;
      }
      reset();
    };

    const score = () => Math.floor(distance / 18);

    const spawn = () => {
      const scale = Math.random() < 0.35 ? PX + 1 : PX;
      // A wide cactus is airborne-time-limited, so it is hardest when it
      // crawls past slowly. Hold clusters back until the speed has ramped,
      // otherwise the opening obstacles are the meanest ones.
      const cluster = speed > 8 && Math.random() < 0.25 ? 2 : 1;
      const size = spriteSize(CACTUS, scale);
      obstacles.push({
        x: width + 20,
        width: size.width * cluster,
        height: size.height,
        scale,
      });
      spawnIn = Math.round((80 + Math.random() * 70) * (BASE_SPEED / speed));
    };

    const collides = (o: Obstacle) => {
      const dinoLeft = DINO_X + HITBOX_INSET;
      const dinoRight = DINO_X + dino.width - HITBOX_INSET;
      // Only the feet matter vertically: clearing a cactus means being above
      // it, and the dino's head is always above the ground anyway.
      const dinoFeet = groundY - dinoY - HITBOX_INSET;
      const obsLeft = o.x + HITBOX_INSET;
      const obsRight = o.x + o.width - HITBOX_INSET;
      const obsTop = groundY - o.height;
      return dinoRight > obsLeft && dinoLeft < obsRight && dinoFeet > obsTop;
    };

    const update = (step: number) => {
      if (status !== "running") return;
      frames += step;
      distance += speed * step;
      speed = Math.min(MAX_SPEED, BASE_SPEED + distance / 900);

      velocity += GRAVITY * step;
      dinoY = Math.max(0, dinoY - velocity * step);
      if (dinoY === 0) velocity = 0;

      spawnIn -= step;
      if (spawnIn <= 0) spawn();

      for (const o of obstacles) o.x -= speed * step;
      obstacles = obstacles.filter((o) => o.x + o.width > -10);

      for (const c of clouds) c.x -= speed * 0.35 * step;
      clouds = clouds.filter((c) => c.x > -60);
      if (clouds.length < 2 && Math.random() < 0.01) {
        clouds.push({ x: width + 40, y: 24 + Math.random() * 50 });
      }

      if (obstacles.some(collides)) {
        status = "over";
        best = Math.max(best, score());
        try {
          localStorage.setItem("deepeval:dino-best", String(best));
        } catch {
          // Private mode — high score just does not persist.
        }
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, HEIGHT);
      ctx.fillStyle = colour;
      ctx.globalAlpha = 0.3;
      for (const c of clouds) drawSprite(ctx, CLOUD, c.x, c.y, 2);
      ctx.globalAlpha = 1;

      ctx.fillRect(0, groundY, width, 2);
      for (let x = -((distance * 1) % 40); x < width; x += 40) {
        ctx.fillRect(x + 12, groundY + 5, 6, 2);
      }

      const running = status === "running" && dinoY === 0;
      const frame = running ? Math.floor(frames / 6) % DINO_FRAMES.length : 0;
      drawSprite(
        ctx,
        DINO_FRAMES[frame],
        DINO_X,
        groundY - dino.height - dinoY,
        PX,
      );

      for (const o of obstacles) {
        const size = spriteSize(CACTUS, o.scale);
        for (let i = 0; i * size.width < o.width; i++) {
          drawSprite(
            ctx,
            CACTUS,
            o.x + i * size.width,
            groundY - size.height,
            o.scale,
          );
        }
      }

      ctx.font = "600 13px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "right";
      ctx.globalAlpha = 0.5;
      if (best > 0) {
        ctx.fillText(`HI ${String(best).padStart(5, "0")}`, width - 84, 22);
      }
      ctx.globalAlpha = 1;
      ctx.fillText(String(score()).padStart(5, "0"), width - 8, 22);

      ctx.textAlign = "center";
      if (status === "idle") {
        ctx.globalAlpha = 0.65;
        ctx.fillText("PRESS SPACE TO PLAY", width / 2, groundY + 26);
        ctx.globalAlpha = 1;
      } else if (status === "over") {
        ctx.font = "600 15px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText("G A M E   O V E R", width / 2, 62);
        ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.globalAlpha = 0.65;
        ctx.fillText("press space to retry", width / 2, groundY + 26);
        ctx.globalAlpha = 1;
      }
    };

    const loop = (now: number) => {
      const step = last ? Math.min((now - last) / (1000 / 60), 3) : 1;
      last = now;
      update(step);
      draw();
      raf = requestAnimationFrame(loop);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code !== "Space" && event.code !== "ArrowUp") return;
      event.preventDefault();
      jump();
    };

    const onPointerDown = (event: PointerEvent) => {
      event.preventDefault();
      canvas.focus();
      jump();
    };

    readColour();
    resize();
    raf = requestAnimationFrame(loop);

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    const themeObserver = new MutationObserver(readColour);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "style"],
    });
    window.addEventListener("keydown", onKeyDown);
    canvas.addEventListener("pointerdown", onPointerDown);

    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      themeObserver.disconnect();
      window.removeEventListener("keydown", onKeyDown);
      canvas.removeEventListener("pointerdown", onPointerDown);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={styles.canvas}
      tabIndex={0}
      role="img"
      aria-label="Dinosaur jumping game. Press space or tap to play."
    />
  );
};
