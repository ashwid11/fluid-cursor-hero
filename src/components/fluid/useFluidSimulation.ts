import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

export const FLUID_CONFIG = {
  simResolution: 128,
  dyeResolution: 1024,
  splatRadius: 0.0022,
  splatForce: 6000,
  curl: 32,
  pressureIterations: 20,
  pressure: 0.8,
  densityDissipation: 1.4,
  velocityDissipation: 0.35,
  dyeIntensity: 0.4,
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
      gl_FragColor = vec4(texture2D(uTarget, vUv).xyz + splat, 1.0);
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
    uniform float dt; uniform float dissipation;
    void main() {
      vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * velTexel;
      vec4 result = texture2D(uSource, coord);
      gl_FragColor = result / (1.0 + dissipation * dt);
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
  const hue = useRef(0)

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
    }
    simRef.current = sim
    return () => {
      simRef.current = null
      sim.velocity.dispose()
      sim.density.dispose()
      sim.pressure.dispose()
      sim.divergence.dispose()
      sim.curl.dispose()
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
    const dt = Math.min(delta, 0.0166)
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

    const p = pointer.current
    if (p.moved) {
      hue.current = (hue.current + 0.004 + Math.hypot(p.dx, p.dy) * 0.5) % 1
      const c = new THREE.Color().setHSL(hue.current, 1, 0.5, THREE.LinearSRGBColorSpace)
      const point = new THREE.Vector2(p.x, p.y)
      run('splat', vel.write, vt, {
        uTarget: vel.read.texture,
        aspectRatio: aspect,
        point,
        radius: cfg.splatRadius,
        color: new THREE.Vector3(p.dx * cfg.splatForce, p.dy * cfg.splatForce, 0),
      })
      vel.swap()
      run('splat', den.write, den.texel, {
        uTarget: den.read.texture,
        aspectRatio: aspect,
        point,
        radius: cfg.splatRadius,
        color: new THREE.Vector3(c.r, c.g, c.b).multiplyScalar(cfg.dyeIntensity),
      })
      den.swap()
      p.dx = 0
      p.dy = 0
      p.moved = false
    }

    run('curl', sim.curl, vt, { uVelocity: vel.read.texture })
    run('vorticity', vel.write, vt, { uVelocity: vel.read.texture, uCurl: sim.curl.texture, curl: cfg.curl, dt })
    vel.swap()
    run('divergence', sim.divergence, vt, { uVelocity: vel.read.texture })
    run('clear', prs.write, vt, { uTexture: prs.read.texture, value: cfg.pressure })
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
    })
    vel.swap()
    run('advection', den.write, den.texel, {
      uVelocity: vel.read.texture,
      uSource: den.read.texture,
      velTexel: vt,
      dt,
      dissipation: cfg.densityDissipation,
    })
    den.swap()

    gl.setRenderTarget(prevTarget)
    gl.autoClear = prevAuto
  }, -1)

  const getDensity = (): THREE.Texture => {
    const sim = simRef.current
    if (sim) return sim.density.read.texture
    if (!fallback.current) fallback.current = new THREE.Texture()
    return fallback.current
  }
  return { getDensity, aspect }
}
