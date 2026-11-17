# Fluid Cursor Hero

An interactive WebGL hero background: move your cursor and it stirs a glowing,
colorful fluid over pure black. It runs a 2D Eulerian Navier-Stokes simulation
on the GPU (React Three Fiber + Three.js), with a multi-pass bloom and a
silk-shaded display shader.

## Run it

Requires Node.js 18+ and npm.

```bash
git clone https://github.com/ashwid11/fluid-cursor-hero.git
cd fluid-cursor-hero
npm install
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173) and move your mouse.

Other scripts:

| Command           | What it does                               |
| ----------------- | ------------------------------------------ |
| `npm run build`   | Type-check and build to `dist/`            |
| `npm run preview` | Serve the production build locally         |
| `npm run lint`    | Lint with Oxlint                           |

## Tuning the look

- `src/components/fluid/useFluidSimulation.ts`: `FLUID_CONFIG` (resolution,
  splat size and force, vorticity `curl`, dissipation, dye intensity, color
  cycling) and `BLOOM_CONFIG` (bloom threshold and knee).
- `src/components/fluid/FluidHeroScene.tsx`: the display shader (silk shading,
  bloom weight, tone mapping).

## Notes

- Needs a browser with WebGL2 and half-float render targets.
- The simulation runs at 1024px dye resolution; on slower GPUs, lower
  `dyeResolution` in `FLUID_CONFIG`.
