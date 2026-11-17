import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

export const FLUID_CONFIG = {
  simResolution: 128,
  dyeResolution: 1024,
  velocityRadius: 0.0064,
  dyeRadius: 0.0068,
  splatForce: 3400,
  curl: 8,
  pressureIterations: 20,
  pressureDissipation: 0.8,
  densityDissipation: 1.45,
  velocityDissipation: 1.15,
  dyeFloor: 0.0005,
  // Vivid progressive HSV color cycling.
  colorUpdateInterval: 0.18,
  colorLerpTime: 0.08,
  dyeIntensity: 0.11,
}

export const BLOOM_CONFIG = { levels: 4, threshold: 0.16, softKnee: 0.85 }

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6)
  const f = h * 6 - i
  const p = v * (1 - s)
  const q = v * (1 - f * s)
  const t = v * (1 - (1 - f) * s)
  switch (((i % 6) + 6) % 6) {
    case 0: return [v, t, p]
    case 1: return [q, v, p]
    case 2: return [p, v, t]
    case 3: return [p, q, v]
    case 4: return [t, p, v]
    default: return [v, p, q]
  }
}

const VERT = /* glsl */ `
  varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB;
  uniform vec2 texelSize;
  void main() {
    vUv = position.xy * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`

const HEAD = /* glsl */ `
  precision highp float;
  varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB;
`

const FRAG = {
  splat: HEAD + /* glsl */ `
    uniform sampler2D uTarget; uniform float aspectRatio; uniform vec3 color; uniform vec2 point; uniform float radius;
    void main() {
      vec2 p = vUv - point; p.x *= aspectRatio;
      vec3 splat = exp(-dot(p, p) / radius) * color;
      vec3 base = texture2D(uTarget, vUv).xyz;
      gl_FragColor = vec4(base + splat, 1.0);
    }`,
  curl: HEAD + /* glsl */ `
    uniform sampler2D uVelocity;
    void main() {
      float L = texture2D(uVelocity, vL).y; float R = texture2D(uVelocity, vR).y;
      float T = texture2D(uVelocity, vT).x; float B = texture2D(uVelocity, vB).x;
      gl_FragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
    }`,
  vorticity: HEAD + /* glsl */ `
    uniform sampler2D uVelocity; uniform sampler2D uCurl; uniform float curl; uniform float dt;
    void main() {
      float L = texture2D(uCurl, vL).x; float R = texture2D(uCurl, vR).x;
      float T = texture2D(uCurl, vT).x; float B = texture2D(uCurl, vB).x;
      float C = texture2D(uCurl, vUv).x;
      vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
      force /= length(force) + 0.0001;
      force *= curl * C; force.y *= -1.0;
      vec2 v = texture2D(uVelocity, vUv).xy + force * dt;
      gl_FragColor = vec4(clamp(v, -1000.0, 1000.0), 0.0, 1.0);
    }`,
  divergence: HEAD + /* glsl */ `
    uniform sampler2D uVelocity;
    void main() {
      float L = texture2D(uVelocity, vL).x; float R = texture2D(uVelocity, vR).x;
      float T = texture2D(uVelocity, vT).y; float B = texture2D(uVelocity, vB).y;
      vec2 C = texture2D(uVelocity, vUv).xy;
      if (vL.x < 0.0) L = -C.x; if (vR.x > 1.0) R = -C.x;
      if (vT.y > 1.0) T = -C.y; if (vB.y < 0.0) B = -C.y;
      gl_FragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
    }`,
  clear: HEAD + /* glsl */ `
    uniform sampler2D uTexture; uniform float value;
    void main() { gl_FragColor = value * texture2D(uTexture, vUv); }`,
  pressure: HEAD + /* glsl */ `
    uniform sampler2D uPressure; uniform sampler2D uDivergence;
    void main() {
      float L = texture2D(uPressure, vL).x; float R = texture2D(uPressure, vR).x;
      float T = texture2D(uPressure, vT).x; float B = texture2D(uPressure, vB).x;
      float div = texture2D(uDivergence, vUv).x;
      gl_FragColor = vec4((L + R + B + T - div) * 0.25, 0.0, 0.0, 1.0);
    }`,
  gradSubtract: HEAD + /* glsl */ `
    uniform sampler2D uPressure; uniform sampler2D uVelocity;
    void main() {
      float L = texture2D(uPressure, vL).x; float R = texture2D(uPressure, vR).x;
      float T = texture2D(uPressure, vT).x; float B = texture2D(uPressure, vB).x;
      vec2 v = texture2D(uVelocity, vUv).xy - vec2(R - L, T - B);
      gl_FragColor = vec4(v, 0.0, 1.0);
    }`,
  advection: HEAD + /* glsl */ `
    uniform sampler2D uVelocity; uniform sampler2D uSource; uniform vec2 velTexel;
    uniform float dt; uniform float dissipation; uniform float floorValue;
    void main() {
      vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * velTexel;
      vec4 result = texture2D(uSource, coord);
      vec4 decayed = result / (1.0 + dissipation * dt);
      // Linear floor (dye only) so faint haze vanishes completely.
      // Applied to magnitude so signed velocity is never clamped.
      gl_FragColor = sign(decayed) * max(abs(decayed) - vec4(floorValue), vec4(0.0));
    }`,
  bloomPrefilter: HEAD + /* glsl */ `
    uniform sampler2D uSource; uniform vec3 curve; uniform float threshold;
    void main() {
      vec3 c = max(texture2D(uSource, vUv).rgb, vec3(0.0));
      float br = max(c.r, max(c.g, c.b));
      float rq = clamp(br - curve.x, 0.0, curve.y);
      rq = curve.z * rq * rq;
      c *= max(rq, br - threshold) / max(br, 1e-4);
      gl_FragColor = vec4(c, 1.0);
    }`,
  bloomDown: HEAD + /* glsl */ `
    uniform sampler2D uSource; uniform vec2 srcTexel;
    void main() {
      vec3 s = texture2D(uSource, vUv + srcTexel * vec2(-1.0, -1.0)).rgb
             + texture2D(uSource, vUv + srcTexel * vec2( 1.0, -1.0)).rgb
             + texture2D(uSource, vUv + srcTexel * vec2(-1.0,  1.0)).rgb
             + texture2D(uSource, vUv + srcTexel * vec2( 1.0,  1.0)).rgb;
      gl_FragColor = vec4(s * 0.25, 1.0);
    }`,
  bloomUp: HEAD + /* glsl */ `
    uniform sampler2D uLow; uniform sampler2D uBase; uniform vec2 lowTexel; uniform float gain;
    void main() {
      vec3 s = texture2D(uLow, vUv + lowTexel * vec2(-1.0, 0.0)).rgb
             + texture2D(uLow, vUv + lowTexel * vec2( 1.0, 0.0)).rgb
             + texture2D(uLow, vUv + lowTexel * vec2(0.0,  1.0)).rgb
             + texture2D(uLow, vUv + lowTexel * vec2(0.0, -1.0)).rgb
             + 2.0 * texture2D(uLow, vUv + lowTexel * vec2(-0.7, -0.7)).rgb
             + 2.0 * texture2D(uLow, vUv + lowTexel * vec2( 0.7,  0.7)).rgb
             + 2.0 * texture2D(uLow, vUv + lowTexel * vec2(-0.7,  0.7)).rgb
             + 2.0 * texture2D(uLow, vUv + lowTexel * vec2( 0.7, -0.7)).rgb;
      gl_FragColor = vec4(texture2D(uBase, vUv).rgb + s * (gain / 12.0), 1.0);
    }`,
}

type Pass = keyof typeof FRAG

const makeRT = (w: number, h: number) =>
  new THREE.WebGLRenderTarget(w, h, {
        type: THREE.HalfFloatType,
        format: THREE.RGBAFormat,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        wrapS: THREE.ClampToEdgeWrapping,
        wrapT: THREE.ClampToEdgeWrapping,
        depthBuffer: false,
        stencilBuffer: false,
  })

class DoubleFBO {
  read: THREE.WebGLRenderTarget
  write: THREE.WebGLRenderTarget
  texel: THREE.Vector2
  constructor(w: number, h: number) {
    this.read = makeRT(w, h)
    this.write = makeRT(w, h)
    this.texel = new THREE.Vector2(1 / w, 1 / h)
  }
  swap() {
    const t = this.read
    this.read = this.write
    this.write = t
  }
  dispose() {
    this.read.dispose()
    this.write.dispose()
  }
}

interface Sim {
  mats: Record<Pass, THREE.ShaderMaterial>
  quad: THREE.Mesh
  scene: THREE.Scene
  camera: THREE.Camera
  velocity: DoubleFBO
  density: DoubleFBO
  pressure: DoubleFBO
  divergence: THREE.WebGLRenderTarget
  curl: THREE.WebGLRenderTarget
  geometry: THREE.BufferGeometry
  bloomDown: THREE.WebGLRenderTarget[]
  bloomUp: THREE.WebGLRenderTarget[]
}

/**
 * Runs an Eulerian Navier-Stokes fluid simulation. Returns a getter for the
 * current dye texture (stable to call every frame; the texture swaps each step).
 */
export function useFluidSimulation() {
  const gl = useThree((s) => s.gl)
  const width = useThree((s) => s.size.width)
  const height = useThree((s) => s.size.height)
  const aspect = width / Math.max(height, 1)
  const simH = FLUID_CONFIG.simResolution
  const simW = Math.max(2, Math.round(simH * aspect))
  const dyeH = FLUID_CONFIG.dyeResolution
  const dyeW = Math.max(2, Math.round(dyeH * aspect))

  const simRef = useRef<Sim | null>(null)
  const fallback = useRef<THREE.Texture | null>(null)
  const pointer = useRef({ x: 0.5, y: 0.5, dx: 0, dy: 0, moved: false, has: false })
  const hue = useRef(Math.random())
  const colorTimer = useRef(0)
  const currentColor = useRef(new THREE.Vector3())
  const targetColor = useRef(new THREE.Vector3(...hsvToRgb(hue.current, 1, 1)))
  const prevDelta = useRef({ x: 0, y: 0 })

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const x = e.clientX / window.innerWidth
      const y = 1 - e.clientY / window.innerHeight
      const p = pointer.current
      if (p.has) {
        p.dx += x - p.x
        p.dy += y - p.y
        p.moved = true
      }
      p.x = x
      p.y = y
      p.has = true
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => window.removeEventListener('pointermove', onMove)
  }, [])

  useEffect(() => {
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3))
    const mats = {} as Record<Pass, THREE.ShaderMaterial>
    ;(Object.keys(FRAG) as Pass[]).forEach((k) => {
      mats[k] = new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG[k],
        uniforms: { texelSize: { value: new THREE.Vector2() } },
        depthTest: false,
        depthWrite: false,
      })
    })
    const bloomDown: THREE.WebGLRenderTarget[] = []
    for (let i = 0; i < BLOOM_CONFIG.levels; i++) {
      const d = 2 ** (i + 1)
      bloomDown.push(makeRT(Math.max(2, Math.floor(dyeW / d)), Math.max(2, Math.floor(dyeH / d))))
    }
    // bloomUp[i] accumulates level i with the blurred result of level i + 1.
    const bloomUp = bloomDown.slice(0, -1).map((t) => makeRT(t.width, t.height))
    const quad = new THREE.Mesh(geometry, mats.splat)
    quad.frustumCulled = false
    const scene = new THREE.Scene()
    scene.add(quad)
    const sim: Sim = {
      mats,
      quad,
      scene,
      camera: new THREE.Camera(),
      velocity: new DoubleFBO(simW, simH),
      density: new DoubleFBO(dyeW, dyeH),
      pressure: new DoubleFBO(simW, simH),
      divergence: makeRT(simW, simH),
      curl: makeRT(simW, simH),
      geometry,
      bloomDown,
      bloomUp,
    }
    simRef.current = sim
    return () => {
      simRef.current = null
      sim.velocity.dispose()
      sim.density.dispose()
      sim.pressure.dispose()
      sim.divergence.dispose()
      sim.curl.dispose()
      bloomDown.forEach((t) => t.dispose())
      bloomUp.forEach((t) => t.dispose())
      geometry.dispose()
      Object.values(mats).forEach((m) => m.dispose())
    }
  }, [simW, simH, dyeW, dyeH])

  useEffect(
    () => () => {
      fallback.current?.dispose()
    },
    [],
  )

  useFrame((_state, delta) => {
    const sim = simRef.current
    if (!sim) return
    const dt = Math.min(delta, 0.033)
    const { mats, quad, scene, camera } = sim
    const cfg = FLUID_CONFIG

    const prevTarget = gl.getRenderTarget()
    const prevAuto = gl.autoClear
    gl.autoClear = false

    const run = (pass: Pass, target: THREE.WebGLRenderTarget, texel: THREE.Vector2, u: Record<string, unknown>) => {
      const m = mats[pass]
      m.uniforms.texelSize.value.copy(texel)
      for (const k in u) {
        if (!m.uniforms[k]) m.uniforms[k] = { value: u[k] }
        else m.uniforms[k].value = u[k]
      }
      quad.material = m
      gl.setRenderTarget(target)
      gl.render(scene, camera)
    }

    const vel = sim.velocity
    const den = sim.density
    const prs = sim.pressure
    const vt = vel.texel

    // Progressive analogous-plus hue step.
    colorTimer.current += dt
    if (colorTimer.current >= cfg.colorUpdateInterval) {
      colorTimer.current %= cfg.colorUpdateInterval
      hue.current = (hue.current + 0.14 + Math.random() * 0.06) % 1.0
      targetColor.current.set(...hsvToRgb(hue.current, 1, 1))
    }
    currentColor.current.lerp(targetColor.current, Math.min(1, dt / cfg.colorLerpTime))

    const p = pointer.current
    if (p.moved) {
      const c = currentColor.current
      // Light exponential smoothing of pointer delta to suppress trackpad jitter.
      const pd = prevDelta.current
      const dx = 0.65 * p.dx + 0.35 * pd.x
      const dy = 0.65 * p.dy + 0.35 * pd.y
      pd.x = dx
      pd.y = dy
      const distance = Math.hypot(p.dx * aspect, p.dy)
      const steps = Math.min(24, Math.max(1, Math.ceil(distance / 0.003)))
      const sx = p.x - p.dx
      const sy = p.y - p.dy
      const k = cfg.dyeIntensity / steps
      for (let i = 1; i <= steps; i++) {
        const t = i / steps
        const point = new THREE.Vector2(sx + p.dx * t, sy + p.dy * t)
        run('splat', vel.write, vt, {
          uTarget: vel.read.texture,
          aspectRatio: aspect,
          point,
          radius: cfg.velocityRadius,
          color: new THREE.Vector3((dx / steps) * cfg.splatForce, (dy / steps) * cfg.splatForce, 0),
        })
        vel.swap()
        run('splat', den.write, den.texel, {
          uTarget: den.read.texture,
          aspectRatio: aspect,
          point,
          radius: cfg.dyeRadius,
          color: new THREE.Vector3(c.x * k, c.y * k, c.z * k),
        })
        den.swap()
      }
      p.dx = 0
      p.dy = 0
      p.moved = false
    } else {
      prevDelta.current.x = 0
      prevDelta.current.y = 0
    }

    if (cfg.curl > 0) {
      run('curl', sim.curl, vt, { uVelocity: vel.read.texture })
      run('vorticity', vel.write, vt, { uVelocity: vel.read.texture, uCurl: sim.curl.texture, curl: cfg.curl, dt })
      vel.swap()
    }
    run('divergence', sim.divergence, vt, { uVelocity: vel.read.texture })
    run('clear', prs.write, vt, { uTexture: prs.read.texture, value: cfg.pressureDissipation })
    prs.swap()
    for (let i = 0; i < cfg.pressureIterations; i++) {
      run('pressure', prs.write, vt, { uPressure: prs.read.texture, uDivergence: sim.divergence.texture })
      prs.swap()
    }
    run('gradSubtract', vel.write, vt, { uPressure: prs.read.texture, uVelocity: vel.read.texture })
    vel.swap()
    run('advection', vel.write, vt, {
      uVelocity: vel.read.texture,
      uSource: vel.read.texture,
      velTexel: vt,
      dt,
      dissipation: cfg.velocityDissipation,
      floorValue: 0,
    })
    vel.swap()
    run('advection', den.write, den.texel, {
      uVelocity: vel.read.texture,
      uSource: den.read.texture,
      velTexel: vt,
      dt,
      dissipation: cfg.densityDissipation,
      floorValue: cfg.dyeFloor,
    })
    den.swap()

    // Multi-pass bloom: soft-knee bright pass, downsample chain, progressive upsample.
    const { threshold, softKnee } = BLOOM_CONFIG
    const knee = threshold * softKnee + 1e-4
    const down = sim.bloomDown
    const up = sim.bloomUp
    const tex = (t: THREE.WebGLRenderTarget) => new THREE.Vector2(1 / t.width, 1 / t.height)
    run('bloomPrefilter', down[0], den.texel, {
      uSource: den.read.texture,
      curve: new THREE.Vector3(threshold - knee, knee * 2, 0.25 / knee),
      threshold,
    })
    for (let i = 1; i < down.length; i++) {
      run('bloomDown', down[i], den.texel, { uSource: down[i - 1].texture, srcTexel: tex(down[i - 1]) })
    }
    for (let i = down.length - 2; i >= 0; i--) {
      const low = i === down.length - 2 ? down[i + 1] : up[i + 1]
      run('bloomUp', up[i], den.texel, {
        uLow: low.texture,
        uBase: down[i].texture,
        lowTexel: tex(low),
        gain: 0.8,
      })
    }

    gl.setRenderTarget(prevTarget)
    gl.autoClear = prevAuto
  }, -1)

  const getDensity = (): THREE.Texture => {
    const sim = simRef.current
    if (sim) return sim.density.read.texture
    if (!fallback.current) fallback.current = new THREE.Texture()
    return fallback.current
  }
  const getBloom = (): THREE.Texture => {
    const sim = simRef.current
    if (sim) return sim.bloomUp[0].texture
    if (!fallback.current) fallback.current = new THREE.Texture()
    return fallback.current
  }
  return { getDensity, getBloom, aspect }
}
