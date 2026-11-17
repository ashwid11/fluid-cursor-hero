import { useEffect, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { FLUID_CONFIG, useFluidSimulation } from './useFluidSimulation'

const BG_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }
`
const BG_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uDye;
  uniform sampler2D uBloom;
  uniform vec2 uTexel;

  float lumaAt(vec2 uv) {
    return dot(max(texture2D(uDye, uv).rgb, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
  }

  void main() {
    vec3 c = max(texture2D(uDye, vUv).rgb, vec3(0.0));

    // Liquid-silk shading from finite-difference luminance gradients.
    float lumaL = lumaAt(vUv - vec2(uTexel.x, 0.0));
    float lumaR = lumaAt(vUv + vec2(uTexel.x, 0.0));
    float lumaT = lumaAt(vUv + vec2(0.0, uTexel.y));
    float lumaB = lumaAt(vUv - vec2(0.0, uTexel.y));
    float dx = lumaR - lumaL;
    float dy = lumaT - lumaB;
    vec3 n = normalize(vec3(dx, dy, length(uTexel) * 1.4));
    float diffuse = clamp(dot(n, normalize(vec3(0.55, 0.75, 0.36))), 0.0, 1.0);
    c += c * (diffuse * 0.42);

    vec3 bloom = max(texture2D(uBloom, vUv).rgb, vec3(0.0));
    c += bloom * 0.62;

    // Subtract shared white floor so overlapping hues stay pure glowing color instead of turning white
    float minChan = min(c.r, min(c.g, c.b));
    c = max(c - vec3(minChan * 0.62), vec3(0.0));

    // Luminous HDR roll-off letting dominant color channels glow up to 0.95
    float peak = max(c.r, max(c.g, c.b));
    float mappedPeak = 1.0 - exp(-peak * 1.55);
    vec3 mapped = c * (mappedPeak / max(peak, 1e-4));

    // Gentle midtone gamma lift for radiant neon-silk presence without white clipping
    mapped = pow(clamp(mapped, 0.0, 0.95), vec3(0.84));
    gl_FragColor = vec4(mapped, 1.0);
  }
`

type Getter = () => THREE.Texture

function FluidBackground({ getDensity, getBloom, aspect }: { getDensity: Getter; getBloom: Getter; aspect: number }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3))
    return g
  }, [])
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: BG_VERT,
        fragmentShader: BG_FRAG,
        uniforms: {
          uDye: { value: null as THREE.Texture | null },
          uBloom: { value: null as THREE.Texture | null },
          uTexel: { value: new THREE.Vector2() },
        },
        depthTest: false,
        depthWrite: false,
      }),
    [],
  )
  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )
  useFrame(() => {
    const tex = getDensity()
    material.uniforms.uDye.value = tex
    material.uniforms.uBloom.value = getBloom()
    const dyeH = FLUID_CONFIG.dyeResolution
    const dyeW = Math.max(2, Math.round(dyeH * aspect))
    material.uniforms.uTexel.value.set(1 / dyeW, 1 / dyeH)
  })
  return <mesh geometry={geometry} material={material} renderOrder={-1} frustumCulled={false} />
}

function Scene() {
  const { getDensity, getBloom, aspect } = useFluidSimulation()
  return (
    <>
      <color attach="background" args={['#000000']} />
      <FluidBackground getDensity={getDensity} getBloom={getBloom} aspect={aspect} />
    </>
  )
}

export default function FluidHeroScene() {
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [0, 0, 4.5], fov: 40 }}
      gl={{ antialias: true, alpha: false }}
      style={{ position: 'fixed', inset: 0, zIndex: 0, background: '#000000', pointerEvents: 'none' }}
    >
      <Scene />
    </Canvas>
  )
}
